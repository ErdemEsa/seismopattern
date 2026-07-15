#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 4.1: Otomatik Izleme Motoru
====================================================
Bilinen sismik bölgeleri periyodik olarak tarar,
risk degisimlerini loglar, uyari uretir.

Kullanim:
  python scripts/auto_monitor.py --once        (tek sefer tara)
  python scripts/auto_monitor.py --daemon      (arka planda calistir)
  python scripts/auto_monitor.py --report      (son raporu goster)
"""

import json
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from isc_fetch_v2 import fetch_and_analyze
    HAS_ISC = True
except Exception:
    HAS_ISC = False

try:
    from fault_distance import get_fault_db
    HAS_FAULTS = True
except Exception:
    HAS_FAULTS = False

try:
    from coulomb_simple import compute_cff_for_app
    HAS_CFF = True
except Exception:
    HAS_CFF = False

try:
    from gps_velocity import get_gps_info_for_app
    HAS_GPS = True
except Exception:
    HAS_GPS = False

try:
    from nlp_scanner import get_nlp_info_for_app
    HAS_NLP = True
except Exception:
    HAS_NLP = False

# Model yukle
MODEL_DIR = Path("output/models")
try:
    import joblib
    MODELS = {}
    FL = {}
    fp = MODEL_DIR / "feature_lists.json"
    if fp.exists():
        with open(fp) as f:
            FL = json.load(f)
        for pt in ["TIP_A", "TIP_B", "TIP_C"]:
            p = MODEL_DIR / f"model_{pt}.joblib"
            if p.exists():
                MODELS[pt] = (joblib.load(p), FL.get(pt, []))
    HAS_MODEL = len(MODELS) > 0
except Exception:
    HAS_MODEL = False
    MODELS = {}
    FL = {}


# Predict fonksiyonlari (app.py'den kopyalandi)
def rule_based_type(r):
    qr = r.get("quiescence_ratio")
    acc = r.get("accel_90d")
    n3 = r.get("w3_n_events", 0) or 0
    if qr is None or pd.isna(qr) or n3 < 3: return "TIP_C"
    if qr < 0.5: return "TIP_B"
    if qr >= 1.0: return "TIP_A"
    if 0.5 <= qr < 0.8:
        if acc and not pd.isna(acc) and acc >= 1.5: return "TIP_A"
        return "TIP_B"
    return "TIP_A"

def add_derived(r):
    r = r.copy()
    c0 = float(r.get("count_0_1y", 0) or 0)
    c1 = float(r.get("count_1_2y", 0) or 0)
    c2 = float(r.get("count_2_3y", 0) or 0)
    r["count_linear_trend"] = c0 - c2
    r["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)
    def sf(a, b):
        try: return float(a) - float(b)
        except: return float('nan')
    r["b_drop_w3_w1"] = sf(r.get("w3_b_value"), r.get("w1_b_value"))
    r["spatial_focus_change"] = sf(r.get("w3_mean_dist_km"), r.get("w1_mean_dist_km"))
    r["depth_change_km"] = sf(r.get("w1_mean_depth_km"), r.get("w3_mean_depth_km"))
    if not r.get("quiescence_ratio"):
        prev = (c1 + c2) / 2.0
        r["quiescence_ratio"] = c0 / prev if prev > 0 else float('nan')
    if not r.get("w3_n_events"): r["w3_n_events"] = c0 + c1 + c2
    if not r.get("w1_n_events"): r["w1_n_events"] = c0
    return r

def predict_risk_simple(features):
    if not HAS_MODEL: return None
    row = add_derived(features)
    pt = rule_based_type(row)
    comp = {}
    for tip, (pipe, feats) in MODELS.items():
        try:
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
            comp[tip] = round(float(pipe.predict_proba(x)[0, 1]), 4)
        except: comp[tip] = None
    valid = {t: s for t, s in comp.items() if s is not None}
    if not valid: return None
    W = {"TIP_A": 1.0, "TIP_B": 1.2, "TIP_C": 0.5}
    primary = comp.get(pt)
    ws = sum(W.get(t, 1) * s for t, s in valid.items())
    wt = sum(W.get(t, 1) for t in valid)
    ens = ws / wt if wt > 0 else 0.5
    fin = 0.7 * primary + 0.3 * ens if primary is not None else ens
    if fin >= 0.75: lv = "KRITIK"
    elif fin >= 0.60: lv = "YUKSEK"
    elif fin >= 0.45: lv = "ORTA"
    elif fin >= 0.30: lv = "DIKKAT"
    else: lv = "DUSUK"
    return {"score": round(fin, 4), "level": lv, "pattern": pt, "components": comp}


# =========================================================
# VERITABANI
# =========================================================

DB_PATH = Path("output/monitor.db")

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            zone_key TEXT,
            zone_name TEXT,
            lat REAL, lon REAL,
            risk_score REAL,
            risk_level TEXT,
            pattern_type TEXT,
            n_events INTEGER,
            b_value REAL,
            quiescence REAL,
            cff_total REAL,
            gps_vh REAL,
            source TEXT,
            raw_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            zone_key TEXT,
            zone_name TEXT,
            alert_type TEXT,
            message TEXT,
            old_score REAL,
            new_score REAL,
            acknowledged INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


# =========================================================
# IZLENEN BOLGELER
# =========================================================

ZONES = {
    "marmara": {"name": "Marmara", "lat": 40.77, "lon": 29.00, "priority": 1},
    "kahramanmaras": {"name": "Kahramanmaras", "lat": 37.22, "lon": 37.02, "priority": 2},
    "izmit": {"name": "Izmit", "lat": 40.75, "lon": 29.86, "priority": 2},
    "cascadia": {"name": "Cascadia", "lat": 45.50, "lon": -125.00, "priority": 1},
    "nankai": {"name": "Nankai", "lat": 33.00, "lon": 135.00, "priority": 1},
    "lima": {"name": "Lima", "lat": -12.00, "lon": -77.00, "priority": 1},
    "tohoku": {"name": "Tohoku", "lat": 38.30, "lon": 142.37, "priority": 2},
}


# =========================================================
# TARAMA
# =========================================================

def scan_zone(zone_key, zone_info, conn):
    """Tek bir bolgeyi tara."""
    name = zone_info["name"]
    lat = zone_info["lat"]
    lon = zone_info["lon"]
    ts = datetime.utcnow().isoformat()

    print(f"\n  [{zone_key}] {name} ({lat}, {lon})")

    # Sismik veri cek
    feats, meta = None, None
    if HAS_ISC:
        try:
            feats, meta, err = fetch_and_analyze(
                lat, lon, 300, 2.5, use_cache=False
            )
        except Exception as e:
            print(f"    ISC hata: {e}")

    if feats is None:
        print(f"    Veri alinamadi, atlaniyor")
        return None

    n_events = meta.get("n_total", 0)
    source = meta.get("source", "?")
    print(f"    {n_events} olay ({source})")

    # Risk hesapla
    risk = predict_risk_simple(feats)
    if risk is None:
        print(f"    Risk hesaplanamadi")
        return None

    score = risk["score"]
    level = risk["level"]
    pattern = risk["pattern"]
    print(f"    Risk: {level} ({score*100:.1f}%)")

    # Ek katmanlar
    cff_total = 0
    if HAS_CFF:
        try:
            cff = compute_cff_for_app(lat, lon)
            cff_total = cff.get("cff_total", 0)
        except: pass

    gps_vh = None
    if HAS_GPS:
        try:
            gps = get_gps_info_for_app(lat, lon)
            gps_vh = gps.get("gps_mean_vh_mm_yr")
        except: pass

    b_val = feats.get("w1_b_value")
    qr = feats.get("quiescence_ratio")

    # Veritabanina kaydet
    raw = json.dumps({
        "features": {k: v for k, v in feats.items()
                     if not isinstance(v, (list, dict))},
        "meta": {k: v for k, v in meta.items()
                 if not isinstance(v, (list, dict))},
        "risk": risk,
    }, default=str)

    conn.execute("""
        INSERT INTO scan_results
        (timestamp, zone_key, zone_name, lat, lon,
         risk_score, risk_level, pattern_type,
         n_events, b_value, quiescence, cff_total, gps_vh,
         source, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (ts, zone_key, name, lat, lon,
          score, level, pattern,
          n_events, b_val, qr, cff_total, gps_vh,
          source, raw))

    # Onceki skor ile karsilastir
    prev = conn.execute("""
        SELECT risk_score, risk_level FROM scan_results
        WHERE zone_key=? AND id < (SELECT MAX(id) FROM scan_results WHERE zone_key=?)
        ORDER BY id DESC LIMIT 1
    """, (zone_key, zone_key)).fetchone()

    if prev:
        old_score, old_level = prev
        delta = score - old_score

        if delta > 0.10:
            alert_msg = (f"{name}: Risk ARTTI "
                        f"({old_score*100:.1f}% -> {score*100:.1f}%)")
            print(f"    UYARI: {alert_msg}")
            conn.execute("""
                INSERT INTO alerts
                (timestamp, zone_key, zone_name, alert_type,
                 message, old_score, new_score)
                VALUES (?,?,?,?,?,?,?)
            """, (ts, zone_key, name, "RISK_INCREASE",
                  alert_msg, old_score, score))

        elif level in ["KRITIK", "YUKSEK"] and old_level not in ["KRITIK", "YUKSEK"]:
            alert_msg = f"{name}: Seviye yükseldi ({old_level} -> {level})"
            print(f"    ALARM: {alert_msg}")
            conn.execute("""
                INSERT INTO alerts
                (timestamp, zone_key, zone_name, alert_type,
                 message, old_score, new_score)
                VALUES (?,?,?,?,?,?,?)
            """, (ts, zone_key, name, "LEVEL_ESCALATION",
                  alert_msg, old_score, score))

    conn.commit()
    return {"score": score, "level": level, "n_events": n_events}


def scan_all_zones():
    """Tum bilinen bolgeleri tara."""
    print("=" * 60)
    print(f"OTOMATIK TARAMA - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    conn = init_db()

    results = {}
    sorted_zones = sorted(ZONES.items(),
                          key=lambda x: x[1].get("priority", 9))

    for key, info in sorted_zones:
        try:
            r = scan_zone(key, info, conn)
            if r:
                results[key] = r
        except Exception as e:
            print(f"  [{key}] HATA: {e}")

        time.sleep(2)

    # Ozet
    print(f"\n{'='*60}")
    print("TARAMA OZETI")
    print(f"{'='*60}")
    print(f"\n  {'Bolge':<20} {'Skor':>6} {'Seviye':<10} {'Olay':>6}")
    print(f"  {'─'*45}")
    for key, r in sorted(results.items(),
                         key=lambda x: x[1]["score"], reverse=True):
        name = ZONES[key]["name"]
        print(f"  {name:<20} {r['score']*100:>5.1f}% {r['level']:<10} "
              f"{r['n_events']:>6}")

    # Uyarilari goster
    alerts = conn.execute("""
        SELECT timestamp, zone_name, alert_type, message
        FROM alerts
        WHERE date(timestamp) = date('now')
        ORDER BY timestamp DESC
    """).fetchall()

    if alerts:
        print(f"\n  BUGUNUN UYARILARI ({len(alerts)}):")
        for ts, zn, at, msg in alerts:
            print(f"  ⚠ [{ts[:16]}] {msg}")

    conn.close()
    return results


# =========================================================
# RAPOR
# =========================================================

def show_report(days=7):
    """Son N gunun raporunu goster."""
    if not DB_PATH.exists():
        print("Veritabani bulunamadi. Once --once ile tarama yapin.")
        return

    conn = sqlite3.connect(str(DB_PATH))

    print("=" * 60)
    print(f"SON {days} GUN RAPORU")
    print("=" * 60)

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    # Her bolge icin son skor
    print(f"\n  {'Bolge':<20} {'Son Skor':>9} {'Seviye':<10} "
          f"{'Olay':>6} {'Tarih':<20}")
    print(f"  {'─'*70}")

    for key in ZONES:
        row = conn.execute("""
            SELECT risk_score, risk_level, n_events, timestamp
            FROM scan_results
            WHERE zone_key=?
            ORDER BY timestamp DESC LIMIT 1
        """, (key,)).fetchone()

        if row:
            score, level, n_ev, ts = row
            name = ZONES[key]["name"]
            print(f"  {name:<20} {score*100:>8.1f}% {level:<10} "
                  f"{n_ev:>6} {ts[:16]}")

    # Uyarilar
    alerts = conn.execute("""
        SELECT timestamp, zone_name, alert_type, message
        FROM alerts
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
    """, (cutoff,)).fetchall()

    if alerts:
        print(f"\n  UYARILAR ({len(alerts)}):")
        for ts, zn, at, msg in alerts:
            print(f"  ⚠ [{ts[:16]}] {msg}")
    else:
        print(f"\n  Son {days} gunde uyari yok.")

    # Trend analizi
    print(f"\n  TREND (son {days} gun):")
    for key in ZONES:
        rows = conn.execute("""
            SELECT risk_score, timestamp
            FROM scan_results
            WHERE zone_key=? AND timestamp >= ?
            ORDER BY timestamp
        """, (key, cutoff)).fetchall()

        if len(rows) >= 2:
            scores = [r[0] for r in rows]
            trend = scores[-1] - scores[0]
            arrow = "↑" if trend > 0.02 else ("↓" if trend < -0.02 else "→")
            name = ZONES[key]["name"]
            print(f"  {name:<20} {arrow} {trend*100:+.1f}% "
                  f"({len(rows)} olcum)")

    conn.close()


# =========================================================
# DAEMON MODU
# =========================================================

def run_daemon(interval_hours=24):
    """Arka planda periyodik tarama."""
    print(f"Daemon modu baslatildi (her {interval_hours} saatte bir)")
    print(f"Durdurmak icin Ctrl+C")

    while True:
        try:
            scan_all_zones()
            next_scan = datetime.utcnow() + timedelta(hours=interval_hours)
            print(f"\nSonraki tarama: {next_scan.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"Bekleniyor ({interval_hours} saat)...")
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            print("\nDaemon durduruldu.")
            break
        except Exception as e:
            print(f"\nHATA: {e}")
            print("60 saniye sonra tekrar deneniyor...")
            time.sleep(60)


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser(
        description="SeismoPattern Otomatik Izleme Motoru"
    )
    ap.add_argument("--once", action="store_true",
                    help="Tek sefer tum bolgeleri tara")
    ap.add_argument("--daemon", action="store_true",
                    help="Arka planda periyodik calistir")
    ap.add_argument("--report", action="store_true",
                    help="Son raporu goster")
    ap.add_argument("--interval", type=int, default=24,
                    help="Tarama araligi (saat)")
    ap.add_argument("--days", type=int, default=7,
                    help="Rapor kac gun geriye baksin")
    args = ap.parse_args()

    print(f"ISC: {'OK' if HAS_ISC else 'YOK'}")
    print(f"Model: {'OK' if HAS_MODEL else 'YOK'}")
    print(f"Fay: {'OK' if HAS_FAULTS else 'YOK'}")
    print(f"CFF: {'OK' if HAS_CFF else 'YOK'}")
    print(f"GPS: {'OK' if HAS_GPS else 'YOK'}")
    print(f"NLP: {'OK' if HAS_NLP else 'YOK'}")

    if args.once:
        scan_all_zones()
    elif args.daemon:
        run_daemon(args.interval)
    elif args.report:
        show_report(args.days)
    else:
        print("\nKullanim:")
        print("  python scripts/auto_monitor.py --once")
        print("  python scripts/auto_monitor.py --daemon --interval 24")
        print("  python scripts/auto_monitor.py --report --days 7")


if __name__ == "__main__":
    main()