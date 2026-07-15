#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Prospective Test Sistemi
=========================================
Her hafta/gun otomatik tahmin kaydeder.
Hicbir sonucu degistirmeden biriktirir.
Gelecekte gerceklesen depremlerle karsilastirir.

Kullanim:
  python scripts/prospective_tracker.py --record
  python scripts/prospective_tracker.py --evaluate
  python scripts/prospective_tracker.py --status
"""

import json
import sqlite3
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta
import argparse

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dual_risk_framework import analyze_zone, EXTENDED_ZONES
    HAS_DR = True
except Exception:
    HAS_DR = False

DB_PATH = Path("output/prospective/prospective.db")
DB_PATH.parent.mkdir(exist_ok=True)


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            zone_key TEXT NOT NULL,
            zone_name TEXT,
            lt_score REAL,
            st_score REAL,
            combined_score REAL,
            combined_level TEXT,
            h30 REAL, h90 REAL, h1y REAL, h5y REAL,
            pattern TEXT,
            b_value REAL,
            quiescence REAL,
            n_events INTEGER,
            frozen INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS actual_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            lat REAL, lon REAL,
            magnitude REAL,
            zone_key TEXT,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_date TEXT,
            window_days INTEGER,
            n_zones INTEGER,
            n_hits INTEGER,
            n_misses INTEGER,
            n_false_alarms INTEGER,
            auc REAL,
            details TEXT
        )
    """)
    conn.commit()
    return conn


def record_predictions():
    """Tum bolgelerin guncel tahminlerini kaydet."""
    if not HAS_DR:
        print("Dual risk modulu yok")
        return

    conn = init_db()
    ts = datetime.utcnow().isoformat()
    n = 0

    print(f"Tahminler kaydediliyor: {ts[:16]}")

    for key in EXTENDED_ZONES:
        try:
            res = analyze_zone(zone_key=key)
            lt = res.get("long_term", {})
            st = res.get("short_term", {})
            cb = res.get("combined", {})
            hz = res.get("horizons", {})
            z = res.get("zone", {})

            st_feats = st.get("raw_features", {}) if not st.get("error") else {}

            conn.execute("""
                INSERT INTO predictions
                (timestamp, zone_key, zone_name,
                 lt_score, st_score, combined_score, combined_level,
                 h30, h90, h1y, h5y,
                 pattern, b_value, quiescence, n_events)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ts, key, z.get("name"),
                lt.get("score"),
                st.get("score") if not st.get("error") else None,
                cb.get("score"),
                cb.get("level"),
                hz.get("30d"), hz.get("90d"), hz.get("1y"), hz.get("5y"),
                st.get("pattern") if not st.get("error") else None,
                st_feats.get("w1_b_value"),
                st_feats.get("quiescence_ratio"),
                st_feats.get("w3_n_events"),
            ))
            n += 1
            print(f"  {key}: COMB={cb.get('score',0)*100:.1f}%")

        except Exception as e:
            print(f"  {key}: HATA {e}")

        import time
        time.sleep(2)

    conn.commit()
    conn.close()
    print(f"\n{n} bolge kaydedildi. Tum tahminler frozen=1 (degistirilemez).")


def check_actual_events():
    """USGS'den son buyuk depremleri cek ve kaydet."""
    import requests

    conn = init_db()
    end = datetime.utcnow()
    start = end - timedelta(days=30)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 6.0,
        "orderby": "magnitude",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"USGS hatasi: {e}")
        return

    events = data.get("features", [])
    print(f"Son 30 gunde {len(events)} adet Mw6+ deprem")

    import math
    for ev in events:
        p = ev["properties"]
        g = ev["geometry"]["coordinates"]
        lat, lon, mag = g[1], g[0], p.get("mag")
        ev_time = datetime.utcfromtimestamp(p["time"]/1000).isoformat()

        # En yakin zone bul
        best_key = None
        best_dist = 999999
        for key, z in EXTENDED_ZONES.items():
            dlat = lat - z["lat"]
            dlon = lon - z["lon"]
            d = math.sqrt(dlat**2 + dlon**2) * 111
            if d < best_dist and d < 500:
                best_dist = d
                best_key = key

        existing = conn.execute(
            "SELECT id FROM actual_events WHERE timestamp=? AND magnitude=?",
            (ev_time, mag)
        ).fetchone()

        if not existing:
            conn.execute("""
                INSERT INTO actual_events
                (timestamp, lat, lon, magnitude, zone_key, source)
                VALUES (?,?,?,?,?,?)
            """, (ev_time, lat, lon, mag, best_key, "USGS"))
            print(f"  Mw{mag:.1f} {p.get('place','')} -> zone={best_key}")

    conn.commit()
    conn.close()


def evaluate(window_days=365):
    """Tahminleri gercek olaylarla karsilastir."""
    conn = init_db()

    preds = pd.read_sql(
        "SELECT * FROM predictions ORDER BY timestamp", conn
    )
    events = pd.read_sql(
        "SELECT * FROM actual_events ORDER BY timestamp", conn
    )

    if len(preds) == 0:
        print("Henuz tahmin yok. Once --record calistirin.")
        conn.close()
        return

    print(f"Tahmin sayisi: {len(preds)}")
    print(f"Gercek olay sayisi: {len(events)}")

    if len(events) == 0:
        print("Henuz gercek olay kaydedilmemis.")
        print("Once --record ile tahmin kaydedin.")
        print("Sonra bekleyin (gunler/haftalar/aylar).")
        print("Sonra --evaluate ile degerlendirin.")
        conn.close()
        return

    # Her tahmin icin: sonraki window_days gun icinde o zone'da Mw6+ oldu mu?
    results = []
    for _, pred in preds.iterrows():
        pred_time = pd.to_datetime(pred["timestamp"])
        zone = pred["zone_key"]
        cutoff = pred_time + timedelta(days=window_days)

        hit = events[
            (events["zone_key"] == zone) &
            (pd.to_datetime(events["timestamp"]) > pred_time) &
            (pd.to_datetime(events["timestamp"]) <= cutoff) &
            (events["magnitude"] >= 6.0)
        ]

        results.append({
            "zone": zone,
            "pred_time": str(pred_time),
            "combined_score": pred["combined_score"],
            "actual_event": len(hit) > 0,
            "n_events": len(hit),
        })

    df = pd.DataFrame(results)

    if df["actual_event"].nunique() < 2:
        print("Yeterli cesiitlilik yok (hem hit hem miss gerekli)")
        conn.close()
        return

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(df["actual_event"].astype(int),
                         df["combined_score"].fillna(0))

    n_hits = df["actual_event"].sum()
    n_miss = (~df["actual_event"]).sum()

    print(f"\nPROSPECTIVE DEGERLENDIRME ({window_days} gun penceresi):")
    print(f"  AUC: {auc:.4f}")
    print(f"  Hit: {n_hits}, Miss: {n_miss}")
    print(f"  Toplam tahmin: {len(df)}")

    conn.execute("""
        INSERT INTO evaluations
        (eval_date, window_days, n_zones, n_hits, n_misses, auc)
        VALUES (?,?,?,?,?,?)
    """, (datetime.utcnow().isoformat(), window_days,
          len(df), int(n_hits), int(n_miss), round(float(auc), 4)))

    conn.commit()
    conn.close()


def show_status():
    if not DB_PATH.exists():
        print("Veritabani yok. Once --record calistirin.")
        return

    conn = sqlite3.connect(str(DB_PATH))

    n_pred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_events = conn.execute("SELECT COUNT(*) FROM actual_events").fetchone()[0]
    n_eval = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]

    print(f"Prospective Tracker Durumu:")
    print(f"  Kayitli tahmin: {n_pred}")
    print(f"  Gercek olay: {n_events}")
    print(f"  Degerlendirme: {n_eval}")

    if n_pred > 0:
        first = conn.execute(
            "SELECT MIN(timestamp) FROM predictions"
        ).fetchone()[0]
        last = conn.execute(
            "SELECT MAX(timestamp) FROM predictions"
        ).fetchone()[0]
        print(f"  Ilk tahmin: {first[:16]}")
        print(f"  Son tahmin: {last[:16]}")

    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                    help="Guncel tahminleri kaydet")
    ap.add_argument("--evaluate", action="store_true",
                    help="Tahminleri gercek olaylarla karsilastir")
    ap.add_argument("--check-events", action="store_true",
                    help="Son buyuk depremleri USGS'den cek")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--window", type=int, default=365,
                    help="Degerlendirme penceresi (gun)")
    args = ap.parse_args()

    if args.record:
        record_predictions()
    elif args.evaluate:
        evaluate(args.window)
    elif args.check_events:
        check_actual_events()
    elif args.status:
        show_status()
    else:
        print("Kullanim:")
        print("  python scripts/prospective_tracker.py --record")
        print("  python scripts/prospective_tracker.py --check-events")
        print("  python scripts/prospective_tracker.py --evaluate")
        print("  python scripts/prospective_tracker.py --status")


if __name__ == "__main__":
    main()