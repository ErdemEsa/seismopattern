#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Prospective Test Sistemi (Immutable v2.1)
=========================================================
Her hafta/gun otomatik tahmin kaydeder.
Tahminleri degistirilemez sekilde hash'ler ve arsivler.
Gelecekte gerceklesen depremlerle karsilastirir.

Kullanim:
  python scripts/prospective_tracker.py --migrate
  python scripts/prospective_tracker.py --record
  python scripts/prospective_tracker.py --check-events
  python scripts/prospective_tracker.py --evaluate
  python scripts/prospective_tracker.py --verify
  python scripts/prospective_tracker.py --status
"""

import json
import sqlite3
import sys
import time
import hashlib
import subprocess
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse

sys.path.insert(0, str(Path(__file__).parent))

try:
    from dual_risk_framework import analyze_zone, EXTENDED_ZONES
    HAS_DR = True
except Exception:
    HAS_DR = False
    EXTENDED_ZONES = {}

DB_PATH = Path("output/prospective/prospective.db")
ARCHIVE_ROOT = Path("output/prospective/archive")
MANIFEST_PATH = Path("output/prospective/manifest.jsonl")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = 2
MIN_RECORD_INTERVAL_HOURS = 20


# =============================================
# Yardimci fonksiyonlar
# =============================================

def utc_now_str():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_naive_utc(ts_str):
    """Herhangi bir timestamp string'ini naive UTC datetime'a cevir."""
    s = str(ts_str).strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "+" in s[10:]:
        s = s[:s.index("+", 10)]
    return datetime.fromisoformat(s)


def canonical_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_value(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return v

def normalize_nullable_text(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s else None


def short_text(v, n=16):
    s = normalize_nullable_text(v)
    return s[:n] if s else "None"


def detect_code_version():
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_id_from_timestamp(ts, prefix="run"):
    dt = to_naive_utc(ts)
    return f"{prefix}-{dt.strftime('%Y%m%dT%H%M%SZ')}"


def ensure_column(conn, table, col_def):
    col_name = col_def.split()[0]
    existing = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col_name not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")


def get_last_run_row(conn):
    return conn.execute("""
        SELECT run_id, timestamp, run_hash
        FROM prediction_runs
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()


def build_prediction_payload(
    timestamp, zone_key, zone_name,
    lt_score, st_score, combined_score, combined_level,
    h30, h90, h1y, h5y,
    pattern, b_value, quiescence, n_events,
    source="dual_risk_framework.analyze_zone",
    analysis_error=None, legacy_backfill=False
):
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": timestamp,
        "zone_key": clean_value(zone_key),
        "zone_name": clean_value(zone_name),
        "lt_score": clean_value(lt_score),
        "st_score": clean_value(st_score),
        "combined_score": clean_value(combined_score),
        "combined_level": clean_value(combined_level),
        "h30": clean_value(h30),
        "h90": clean_value(h90),
        "h1y": clean_value(h1y),
        "h5y": clean_value(h5y),
        "pattern": clean_value(pattern),
        "b_value": clean_value(b_value),
        "quiescence": clean_value(quiescence),
        "n_events": clean_value(n_events),
        "source": source,
        "analysis_error": clean_value(analysis_error),
        "legacy_backfill": bool(legacy_backfill),
    }


def write_archive(run_id, timestamp, prev_run_hash, code_version,
                   payloads, legacy_backfill=False):
    dt = to_naive_utc(timestamp)
    folder = ARCHIVE_ROOT / dt.strftime("%Y") / dt.strftime("%m")
    folder.mkdir(parents=True, exist_ok=True)

    archive_obj = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "prev_run_hash": prev_run_hash,
        "code_version": code_version,
        "legacy_backfill": bool(legacy_backfill),
        "n_predictions": len(payloads),
        "predictions": payloads,
    }

    archive_path = folder / f"predictions_{run_id}.json"
    archive_path.write_text(
        json.dumps(archive_obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    run_hash = sha256_text(canonical_json(archive_obj))

    sha_path = folder / f"predictions_{run_id}.json.sha256"
    sha_path.write_text(f"{run_hash}  {archive_path.name}\n", encoding="utf-8")

    manifest_entry = {
        "timestamp_utc": timestamp,
        "run_id": run_id,
        "run_hash": run_hash,
        "prev_run_hash": prev_run_hash,
        "archive_path": str(archive_path).replace("\\", "/"),
        "legacy_backfill": bool(legacy_backfill),
    }
    with MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(canonical_json(manifest_entry) + "\n")

    return archive_path, run_hash


# =============================================
# DB init / migration
# =============================================

def init_db(verbose=False, skip_triggers=False):
    conn = sqlite3.connect(str(DB_PATH))

    # --- Tablo olusturma ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            zone_key TEXT NOT NULL,
            zone_name TEXT,
            lt_score REAL, st_score REAL,
            combined_score REAL, combined_level TEXT,
            h30 REAL, h90 REAL, h1y REAL, h5y REAL,
            pattern TEXT, b_value REAL, quiescence REAL, n_events INTEGER,
            frozen INTEGER DEFAULT 1
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS actual_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, lat REAL, lon REAL,
            magnitude REAL, zone_key TEXT, source TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_date TEXT, window_days INTEGER,
            n_zones INTEGER, n_hits INTEGER, n_misses INTEGER,
            n_false_alarms INTEGER, auc REAL, details TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            n_predictions INTEGER NOT NULL,
            run_hash TEXT NOT NULL,
            prev_run_hash TEXT,
            archive_path TEXT,
            code_version TEXT,
            schema_version INTEGER,
            legacy_backfill INTEGER DEFAULT 0,
            notes TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS integrity_checks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            check_time TEXT, run_id TEXT,
            ok INTEGER, details TEXT
        )
    """)

    # --- Yeni kolonlar ---
    ensure_column(conn, "predictions", "run_id TEXT")
    ensure_column(conn, "predictions", "record_index INTEGER")
    ensure_column(conn, "predictions", "record_hash TEXT")
    ensure_column(conn, "predictions", "prev_record_hash TEXT")
    ensure_column(conn, "predictions", "payload_json TEXT")
    ensure_column(conn, "predictions", "archive_path TEXT")
    ensure_column(conn, "predictions", "schema_version INTEGER")
    ensure_column(conn, "predictions", "legacy_record INTEGER DEFAULT 0")
    ensure_column(conn, "predictions", "code_version TEXT")
    ensure_column(conn, "actual_events", "frozen INTEGER DEFAULT 1")

    # --- Indexler ---
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_zone ON predictions(zone_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON prediction_runs(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ae_ts ON actual_events(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ae_zone ON actual_events(zone_key)")

    conn.commit()

    # --- Triggerlar ---
    if not skip_triggers:
        install_triggers(conn)

    return conn


def install_triggers(conn):
    """Immutability triggerlarini kur."""
    # Once mevcutlari sil (idempotent)
    for trig in [
        "trg_predictions_no_update", "trg_predictions_no_delete",
        "trg_prediction_runs_no_update", "trg_prediction_runs_no_delete",
        "trg_actual_events_no_update", "trg_actual_events_no_delete",
    ]:
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")

    conn.execute("""
        CREATE TRIGGER trg_predictions_no_update
        BEFORE UPDATE ON predictions
        WHEN OLD.frozen = 1
        BEGIN SELECT RAISE(ABORT, 'predictions are immutable'); END;
    """)
    conn.execute("""
        CREATE TRIGGER trg_predictions_no_delete
        BEFORE DELETE ON predictions
        BEGIN SELECT RAISE(ABORT, 'predictions cannot be deleted'); END;
    """)
    conn.execute("""
        CREATE TRIGGER trg_prediction_runs_no_update
        BEFORE UPDATE ON prediction_runs
        BEGIN SELECT RAISE(ABORT, 'prediction_runs are immutable'); END;
    """)
    conn.execute("""
        CREATE TRIGGER trg_prediction_runs_no_delete
        BEFORE DELETE ON prediction_runs
        BEGIN SELECT RAISE(ABORT, 'prediction_runs cannot be deleted'); END;
    """)
    conn.execute("""
        CREATE TRIGGER trg_actual_events_no_update
        BEFORE UPDATE ON actual_events
        WHEN OLD.frozen = 1
        BEGIN SELECT RAISE(ABORT, 'actual_events are immutable'); END;
    """)
    conn.execute("""
        CREATE TRIGGER trg_actual_events_no_delete
        BEFORE DELETE ON actual_events
        BEGIN SELECT RAISE(ABORT, 'actual_events cannot be deleted'); END;
    """)
    conn.commit()


def drop_triggers(conn):
    """Migration/backfill sirasinda triggerlari gecici olarak kaldir."""
    for trig in [
        "trg_predictions_no_update", "trg_predictions_no_delete",
        "trg_prediction_runs_no_update", "trg_prediction_runs_no_delete",
        "trg_actual_events_no_update", "trg_actual_events_no_delete",
    ]:
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
    conn.commit()


# =============================================
# Legacy backfill
# =============================================

def backfill_legacy(conn, verbose=False):
    """Eski (hash'siz) tahminleri hash'le ve arsivle."""

    # Zaten backfill yapilmis mi?
    existing_runs = conn.execute(
        "SELECT COUNT(*) FROM prediction_runs"
    ).fetchone()[0]
    if existing_runs > 0:
        if verbose:
            print("Backfill zaten yapilmis, atlaniyor.")
        return 0

    legacy = pd.read_sql("""
        SELECT * FROM predictions
        WHERE run_id IS NULL OR run_id = ''
        ORDER BY timestamp, zone_key, id
    """, conn)

    if len(legacy) == 0:
        if verbose:
            print("Legacy kayit yok.")
        return 0

    # Triggerlari gecici kaldir (UPDATE yapabilmek icin)
    drop_triggers(conn)

    total = 0
    prev_run_hash = "GENESIS"

    for ts, grp in legacy.groupby("timestamp", sort=True):
        grp = grp.sort_values(["zone_key", "id"]).reset_index(drop=True)
        run_id = run_id_from_timestamp(ts, prefix="legacy")

        payloads = []
        prev_record_hash = None

        for idx, row in grp.iterrows():
            payload = build_prediction_payload(
                timestamp=row["timestamp"],
                zone_key=row["zone_key"],
                zone_name=row["zone_name"],
                lt_score=row["lt_score"],
                st_score=row["st_score"],
                combined_score=row["combined_score"],
                combined_level=row["combined_level"],
                h30=row["h30"], h90=row["h90"],
                h1y=row["h1y"], h5y=row["h5y"],
                pattern=row["pattern"],
                b_value=row["b_value"],
                quiescence=row["quiescence"],
                n_events=row["n_events"],
                source="legacy_db_backfill",
                legacy_backfill=True
            )
            record_hash = sha256_text(canonical_json(payload))
            payloads.append(payload)

            # DB guncelle
            conn.execute("""
                UPDATE predictions SET
                    run_id=?, record_index=?,
                    record_hash=?, prev_record_hash=?,
                    payload_json=?, schema_version=?,
                    legacy_record=1, code_version=?
                WHERE id=?
            """, (
                run_id, idx + 1,
                record_hash, prev_record_hash,
                canonical_json(payload), SCHEMA_VERSION,
                "legacy-backfill", int(row["id"])
            ))

            prev_record_hash = record_hash

        # Archive yaz
        archive_path, run_hash = write_archive(
            run_id=run_id, timestamp=ts,
            prev_run_hash=prev_run_hash,
            code_version="legacy-backfill",
            payloads=payloads, legacy_backfill=True
        )

        # archive_path'i predictions'a yaz
        for _, row in grp.iterrows():
            conn.execute("""
                UPDATE predictions SET archive_path=?
                WHERE id=?
            """, (str(archive_path).replace("\\", "/"), int(row["id"])))

        # Run kaydi
        conn.execute("""
            INSERT INTO prediction_runs
            (run_id, timestamp, n_predictions, run_hash, prev_run_hash,
             archive_path, code_version, schema_version, legacy_backfill, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, ts, len(payloads), run_hash, prev_run_hash,
            str(archive_path).replace("\\", "/"),
            "legacy-backfill", SCHEMA_VERSION, 1,
            "Backfilled from pre-hash legacy predictions"
        ))

        prev_run_hash = run_hash
        total += len(grp)

    conn.commit()

    # Triggerlari tekrar kur
    install_triggers(conn)

    return total


# =============================================
# Record
# =============================================

def record_predictions(force=False):
    if not HAS_DR:
        print("Dual risk modulu yok")
        return

    conn = init_db()
    last_run = get_last_run_row(conn)
    now_ts = utc_now_str()

    if last_run and not force:
        dt_last = to_naive_utc(last_run[1])
        dt_now = to_naive_utc(now_ts)
        diff_hours = (dt_now - dt_last).total_seconds() / 3600
        if diff_hours < MIN_RECORD_INTERVAL_HOURS:
            remaining = MIN_RECORD_INTERVAL_HOURS - diff_hours
            print(f"Son kayit cok yeni: {last_run[1]}")
            print(f"Kalan bekleme: {remaining:.1f} saat")
            print("Gerekirse --force kullanin.")
            conn.close()
            return

    run_id = run_id_from_timestamp(now_ts, prefix="run")
    prev_run_hash = last_run[2] if last_run else "GENESIS"
    code_version = detect_code_version()

    payloads = []
    rows_for_db = []

    print(f"Tahminler kaydediliyor: {now_ts[:16]}  | run_id={run_id}")

    prev_record_hash = None

    for idx, key in enumerate(EXTENDED_ZONES.keys(), start=1):
        zone_name = EXTENDED_ZONES.get(key, {}).get("name", key)

        try:
            res = analyze_zone(zone_key=key)
            lt = res.get("long_term", {})
            st = res.get("short_term", {})
            cb = res.get("combined", {})
            hz = res.get("horizons", {})
            z = res.get("zone", {})
            st_feats = st.get("raw_features", {}) if not st.get("error") else {}

            payload = build_prediction_payload(
                timestamp=now_ts,
                zone_key=key,
                zone_name=z.get("name", zone_name),
                lt_score=lt.get("score"),
                st_score=st.get("score") if not st.get("error") else None,
                combined_score=cb.get("score"),
                combined_level=cb.get("level"),
                h30=hz.get("30d"), h90=hz.get("90d"),
                h1y=hz.get("1y"), h5y=hz.get("5y"),
                pattern=st.get("pattern") if not st.get("error") else None,
                b_value=st_feats.get("w1_b_value"),
                quiescence=st_feats.get("quiescence_ratio"),
                n_events=st_feats.get("w3_n_events"),
                analysis_error=st.get("error")
            )

            score = cb.get("score")
            print(f"  {key}: COMB={score*100:.1f}%" if score else f"  {key}: COMB=NA")

        except Exception as e:
            payload = build_prediction_payload(
                timestamp=now_ts,
                zone_key=key,
                zone_name=zone_name,
                lt_score=None, st_score=None,
                combined_score=None, combined_level=None,
                h30=None, h90=None, h1y=None, h5y=None,
                pattern=None, b_value=None,
                quiescence=None, n_events=None,
                analysis_error=str(e)
            )
            print(f"  {key}: HATA -> {e}")

        record_hash = sha256_text(canonical_json(payload))
        payloads.append(payload)

        rows_for_db.append({
            "ts": now_ts,
            "payload": payload,
            "record_hash": record_hash,
            "prev_record_hash": prev_record_hash,
            "record_index": idx,
        })

        prev_record_hash = record_hash
        time.sleep(2)

    # Archive yaz
    archive_path, run_hash = write_archive(
        run_id=run_id, timestamp=now_ts,
        prev_run_hash=prev_run_hash,
        code_version=code_version,
        payloads=payloads
    )

    # Run kaydi
    conn.execute("""
        INSERT INTO prediction_runs
        (run_id, timestamp, n_predictions, run_hash, prev_run_hash,
         archive_path, code_version, schema_version, legacy_backfill, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        run_id, now_ts, len(rows_for_db), run_hash, prev_run_hash,
        str(archive_path).replace("\\", "/"),
        code_version, SCHEMA_VERSION, 0, "Operational prospective record"
    ))

    # Predictions kaydi
    ap_str = str(archive_path).replace("\\", "/")
    for r in rows_for_db:
        p = r["payload"]
        conn.execute("""
            INSERT INTO predictions
            (timestamp, zone_key, zone_name,
             lt_score, st_score, combined_score, combined_level,
             h30, h90, h1y, h5y,
             pattern, b_value, quiescence, n_events,
             frozen, run_id, record_index, record_hash, prev_record_hash,
             payload_json, archive_path, schema_version, legacy_record, code_version)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?,?,?,?,?,0,?)
        """, (
            p["timestamp_utc"], p["zone_key"], p["zone_name"],
            p["lt_score"], p["st_score"],
            p["combined_score"], p["combined_level"],
            p["h30"], p["h90"], p["h1y"], p["h5y"],
            p["pattern"], p["b_value"], p["quiescence"], p["n_events"],
            run_id, r["record_index"], r["record_hash"], r["prev_record_hash"],
            canonical_json(p), ap_str, SCHEMA_VERSION, code_version
        ))

    conn.commit()
    conn.close()

    print(f"\n{len(rows_for_db)} bolge kaydedildi.")
    print(f"Archive : {archive_path}")
    print(f"Run hash: {run_hash}")


# =============================================
# Actual events
# =============================================

def check_actual_events():
    import requests
    import math

    conn = init_db()
    end = datetime.utcnow()
    start = end - timedelta(days=30)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": 6.0,
        "orderby": "time",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"USGS hatasi: {e}")
        conn.close()
        return

    events = data.get("features", [])
    print(f"Son 30 gunde {len(events)} adet Mw6+ deprem")

    added = 0
    for ev in events:
        p = ev["properties"]
        g = ev["geometry"]["coordinates"]
        lat, lon, mag = g[1], g[0], p.get("mag")
        ev_time = datetime.utcfromtimestamp(
            p["time"] / 1000
        ).replace(microsecond=0).isoformat() + "Z"

        best_key = None
        best_dist = 999999
        for key, z in EXTENDED_ZONES.items():
            dlat = lat - z["lat"]
            dlon = lon - z["lon"]
            d = math.sqrt(dlat**2 + dlon**2) * 111
            if d < best_dist and d < 500:
                best_dist = d
                best_key = key

        existing = conn.execute("""
            SELECT id FROM actual_events
            WHERE timestamp=? AND magnitude=? AND lat=? AND lon=?
        """, (ev_time, mag, lat, lon)).fetchone()

        if not existing:
            conn.execute("""
                INSERT INTO actual_events
                (timestamp, lat, lon, magnitude, zone_key, source, frozen)
                VALUES (?,?,?,?,?,?,1)
            """, (ev_time, lat, lon, mag, best_key, "USGS"))
            print(f"  Mw{mag:.1f} {p.get('place','')} -> zone={best_key}")
            added += 1

    conn.commit()
    conn.close()
    print(f"Yeni eklenen olay: {added}")


# =============================================
# Evaluate
# =============================================

def evaluate(window_days=365):
    conn = init_db()

    preds = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp", conn)
    events = pd.read_sql("SELECT * FROM actual_events ORDER BY timestamp", conn)

    if len(preds) == 0:
        print("Henuz tahmin yok.")
        conn.close()
        return

    print(f"Tahmin sayisi: {len(preds)}")
    print(f"Gercek olay sayisi: {len(events)}")

    if len(events) == 0:
        print("Henuz gercek olay kaydedilmemis.")
        conn.close()
        return

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
        })

    df = pd.DataFrame(results)

    if df["actual_event"].nunique() < 2:
        print("Yeterli cesitlilik yok (hem pozitif hem negatif gerekli)")
        conn.close()
        return

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(
        df["actual_event"].astype(int),
        df["combined_score"].fillna(0)
    )

    n_hits = int(df["actual_event"].sum())
    n_miss = int((~df["actual_event"]).sum())

    print(f"\nPROSPECTIVE DEGERLENDIRME ({window_days} gun penceresi):")
    print(f"  AUC: {auc:.4f}")
    print(f"  Hit: {n_hits}, Miss: {n_miss}")
    print(f"  Toplam: {len(df)}")

    conn.execute("""
        INSERT INTO evaluations
        (eval_date, window_days, n_zones, n_hits, n_misses, auc, details)
        VALUES (?,?,?,?,?,?,?)
    """, (
        utc_now_str(), window_days, len(df),
        n_hits, n_miss, round(float(auc), 4),
        json.dumps({"rows": len(df)})
    ))
    conn.commit()
    conn.close()


# =============================================
# Integrity verify
# =============================================

def verify_integrity(run_id_filter=None):
    conn = init_db()

    if run_id_filter:
        runs = pd.read_sql(
            "SELECT * FROM prediction_runs WHERE run_id=? ORDER BY timestamp",
            conn, params=[run_id_filter]
        )
    else:
        runs = pd.read_sql(
            "SELECT * FROM prediction_runs ORDER BY timestamp", conn
        )

    if len(runs) == 0:
        print("Dogrulanacak run yok.")
        conn.close()
        return

    overall_ok = True
    expected_prev_run_hash = "GENESIS"

    for _, run in runs.iterrows():
        issues = []

        rid = normalize_nullable_text(run["run_id"])
        db_run_hash = normalize_nullable_text(run["run_hash"])
        db_prev_run_hash = normalize_nullable_text(run["prev_run_hash"])
        archive_path_val = normalize_nullable_text(run["archive_path"])

        archive_path = Path(archive_path_val) if archive_path_val else None
        archive_obj = None

        # 1. Archive dosya kontrolu
        if not archive_path or not archive_path.exists():
            issues.append("archive file missing")
        else:
            try:
                archive_obj = json.loads(archive_path.read_text(encoding="utf-8"))
            except Exception as e:
                issues.append(f"archive read error: {e}")

        # 2. Run hash kontrolu
        if archive_obj is not None:
            calc_run_hash = sha256_text(canonical_json(archive_obj))

            if calc_run_hash != db_run_hash:
                issues.append(
                    f"run_hash MISMATCH (calc={short_text(calc_run_hash)}, db={short_text(db_run_hash)})"
                )

            if archive_obj.get("run_id") != rid:
                issues.append("archive run_id mismatch")

            if archive_obj.get("timestamp_utc") != run["timestamp"]:
                issues.append("archive timestamp mismatch")

            if archive_obj.get("prev_run_hash") != db_prev_run_hash:
                issues.append("archive prev_run_hash mismatch")

        # 3. Run chain kontrolu
        if not run_id_filter:
            if db_prev_run_hash != expected_prev_run_hash:
                issues.append(
                    f"chain break: expected prev={short_text(expected_prev_run_hash)}, "
                    f"got={short_text(db_prev_run_hash)}"
                )
            expected_prev_run_hash = db_run_hash

        # 4. Prediction kayitlari
        preds = pd.read_sql("""
            SELECT * FROM predictions
            WHERE run_id=?
            ORDER BY record_index, id
        """, conn, params=[rid])

        if len(preds) != int(run["n_predictions"]):
            issues.append(
                f"n_predictions mismatch: db={len(preds)} run={run['n_predictions']}"
            )

        archive_preds = archive_obj.get("predictions", []) if archive_obj else []
        prev_rh = None

        for i, (_, pred) in enumerate(preds.iterrows()):
            pid = int(pred["id"])
            payload_json = normalize_nullable_text(pred["payload_json"])
            current_rh = normalize_nullable_text(pred["record_hash"])
            stored_prev = normalize_nullable_text(pred["prev_record_hash"])

            if not payload_json:
                issues.append(f"payload_json missing pred_id={pid}")
            else:
                try:
                    payload = json.loads(payload_json)
                    calc_rh = sha256_text(canonical_json(payload))

                    if calc_rh != current_rh:
                        issues.append(f"record_hash mismatch pred_id={pid}")

                    if i < len(archive_preds):
                        if canonical_json(payload) != canonical_json(archive_preds[i]):
                            issues.append(f"archive/db mismatch pred_id={pid}")
                    else:
                        issues.append(f"archive prediction missing pred_id={pid}")

                except Exception as e:
                    issues.append(f"payload parse error pred_id={pid}: {e}")

            if stored_prev != prev_rh:
                issues.append(
                    f"record chain break pred_id={pid}: "
                    f"expected={short_text(prev_rh)}, got={short_text(stored_prev)}"
                )

            prev_rh = current_rh

        ok = len(issues) == 0
        overall_ok = overall_ok and ok

        conn.execute("""
            INSERT INTO integrity_checks
            (check_time, run_id, ok, details)
            VALUES (?,?,?,?)
        """, (
            utc_now_str(),
            rid,
            1 if ok else 0,
            json.dumps({"issues": issues}, ensure_ascii=False)
        ))

        print(f"  {rid}: {'OK' if ok else 'FAIL'}")
        for x in issues:
            print(f"    - {x}")

    conn.commit()
    conn.close()
    print(f"\nGenel durum: {'TUM HASHLER GECERLI' if overall_ok else 'TUTARSIZLIK VAR'}")


# =============================================
# Status
# =============================================

def show_status():
    if not DB_PATH.exists():
        print("Veritabani yok. Once --migrate veya --record calistirin.")
        return

    conn = init_db()

    n_pred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_hashed = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE record_hash IS NOT NULL"
    ).fetchone()[0]
    n_legacy = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE legacy_record=1"
    ).fetchone()[0]
    n_runs = conn.execute("SELECT COUNT(*) FROM prediction_runs").fetchone()[0]
    n_events = conn.execute("SELECT COUNT(*) FROM actual_events").fetchone()[0]
    n_eval = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]

    print("Prospective Tracker Durumu:")
    print(f"  Kayitli tahmin      : {n_pred}")
    print(f"  Hash'li tahmin      : {n_hashed}")
    print(f"  Prediction run      : {n_runs}")
    print(f"  Legacy backfill     : {n_legacy}")
    print(f"  Gercek olay         : {n_events}")
    print(f"  Degerlendirme       : {n_eval}")

    if n_pred > 0:
        first = conn.execute("SELECT MIN(timestamp) FROM predictions").fetchone()[0]
        last = conn.execute("SELECT MAX(timestamp) FROM predictions").fetchone()[0]
        print(f"  Ilk tahmin          : {first[:19]}")
        print(f"  Son tahmin          : {last[:19]}")

    if n_runs > 0:
        lr = conn.execute("""
            SELECT run_id, timestamp, run_hash
            FROM prediction_runs ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        print(f"  Son run_id          : {lr[0]}")
        print(f"  Son run hash        : {lr[2][:16]}...")

    lc = conn.execute("""
        SELECT check_time, run_id, ok
        FROM integrity_checks ORDER BY id DESC LIMIT 1
    """).fetchone()
    if lc:
        print(f"  Son integrity check : {lc[0][:19]} | {lc[1]} | {'OK' if lc[2] else 'FAIL'}")

    conn.close()


def migrate_only():
    print("Migration + Legacy Backfill baslatiliyor...")
    conn = init_db(skip_triggers=True)
    n = backfill_legacy(conn, verbose=True)
    install_triggers(conn)
    conn.close()

    if n > 0:
        print(f"Legacy backfill: {n} tahmin hashlenip arsivlendi.")
    print("DB migration tamam.")
    print()
    show_status()


# =============================================
# Main
# =============================================

def main():
    ap = argparse.ArgumentParser(description="SeismoPattern Prospective Tracker")
    ap.add_argument("--migrate", action="store_true",
                    help="DB migration + legacy backfill")
    ap.add_argument("--record", action="store_true",
                    help="Guncel tahminleri kaydet")
    ap.add_argument("--evaluate", action="store_true",
                    help="Tahminleri gercek olaylarla karsilastir")
    ap.add_argument("--check-events", action="store_true",
                    help="Son buyuk depremleri USGS'den cek")
    ap.add_argument("--verify", action="store_true",
                    help="Hash ve archive butunlugunu dogrula")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Cift kayit korumasini atla")
    ap.add_argument("--window", type=int, default=365,
                    help="Degerlendirme penceresi (gun)")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Belirli bir run_id dogrula")
    args = ap.parse_args()

    if args.migrate:
        migrate_only()
    elif args.record:
        record_predictions(force=args.force)
    elif args.evaluate:
        evaluate(args.window)
    elif args.check_events:
        check_actual_events()
    elif args.verify:
        verify_integrity(run_id_filter=args.run_id)
    elif args.status:
        show_status()
    else:
        print("Kullanim:")
        print("  --migrate       DB migration + legacy backfill")
        print("  --record        Guncel tahminleri kaydet")
        print("  --check-events  USGS'den son depremleri cek")
        print("  --evaluate      Prospective degerlendirme")
        print("  --verify        Hash butunluk kontrolu")
        print("  --status        Genel durum")


if __name__ == "__main__":
    main()