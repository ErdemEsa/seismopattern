#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 5.2: Segment Bazli Global Tarama
========================================================
25 bolgeyi otomatik tarar, segment risk analizi ile birlestirir,
nihai rapor uretir.

Kullanim:
  python scripts/global_scan.py --scan
  python scripts/global_scan.py --report
"""

import json
import time
import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

# Moduller
try:
    from isc_fetch_v2 import fetch_and_analyze
    HAS_ISC = True
except: HAS_ISC = False

try:
    from zone_database import EXTENDED_ZONES, compute_segment_risk
    HAS_ZONES = True
except: HAS_ZONES = False

try:
    from fault_distance import get_fault_db
    HAS_FAULTS = True
except: HAS_FAULTS = False

try:
    from coulomb_simple import compute_cff_for_app
    HAS_CFF = True
except: HAS_CFF = False

try:
    from gps_velocity import get_gps_info_for_app
    HAS_GPS = True
except: HAS_GPS = False

try:
    from nlp_scanner import get_nlp_info_for_app
    HAS_NLP = True
except: HAS_NLP = False

# Model
MODEL_DIR = Path("output/models")
try:
    import joblib
    MODELS, FL = {}, {}
    fp = MODEL_DIR / "feature_lists.json"
    if fp.exists():
        with open(fp) as f: FL = json.load(f)
        for pt in ["TIP_A","TIP_B","TIP_C"]:
            p = MODEL_DIR / f"model_{pt}.joblib"
            if p.exists(): MODELS[pt] = (joblib.load(p), FL.get(pt,[]))
    HAS_MODEL = len(MODELS) > 0
except: HAS_MODEL = False; MODELS = {}; FL = {}

OUTPUT_DIR = Path("output/global_scan")
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = OUTPUT_DIR / "global_scan.db"


def rule_based_type(r):
    qr = r.get("quiescence_ratio")
    acc = r.get("accel_90d")
    n3 = r.get("w3_n_events", 0) or 0
    if qr is None or pd.isna(qr) or n3 < 3: return "TIP_C"
    if qr < 0.5: return "TIP_B"
    if qr >= 1.0: return "TIP_A"
    if 0.5 <= qr < 0.8:
        return "TIP_A" if (acc and not pd.isna(acc) and acc >= 1.5) else "TIP_B"
    return "TIP_A"


def predict_risk(features):
    if not HAS_MODEL: return None
    row = features.copy()
    c0 = float(row.get("count_0_1y", 0) or 0)
    c1 = float(row.get("count_1_2y", 0) or 0)
    c2 = float(row.get("count_2_3y", 0) or 0)
    row["count_linear_trend"] = c0 - c2
    row["count_accel_ratio"] = c0 / ((c1+c2)/2 + 1e-6)
    try: row["b_drop_w3_w1"] = float(row.get("w3_b_value",0) or 0) - float(row.get("w1_b_value",0) or 0)
    except: row["b_drop_w3_w1"] = 0
    try: row["spatial_focus_change"] = float(row.get("w3_mean_dist_km",0) or 0) - float(row.get("w1_mean_dist_km",0) or 0)
    except: row["spatial_focus_change"] = 0
    try: row["depth_change_km"] = float(row.get("w1_mean_depth_km",0) or 0) - float(row.get("w3_mean_depth_km",0) or 0)
    except: row["depth_change_km"] = 0
    if not row.get("quiescence_ratio"):
        prev = (c1+c2)/2
        row["quiescence_ratio"] = c0/prev if prev > 0 else None
    if not row.get("w3_n_events"): row["w3_n_events"] = c0+c1+c2
    if not row.get("w1_n_events"): row["w1_n_events"] = c0

    pt = rule_based_type(row)
    comp = {}
    for tip, (pipe, feats) in MODELS.items():
        try:
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
            comp[tip] = round(float(pipe.predict_proba(x)[0,1]), 4)
        except: pass
    valid = {t:s for t,s in comp.items() if s is not None}
    if not valid: return None
    W = {"TIP_A":1.0,"TIP_B":1.2,"TIP_C":0.5}
    primary = comp.get(pt)
    ws = sum(W.get(t,1)*s for t,s in valid.items())
    wt = sum(W.get(t,1) for t in valid)
    ens = ws/wt if wt>0 else 0.5
    fin = 0.7*primary + 0.3*ens if primary else ens
    if fin >= 0.75: lv = "KRITIK"
    elif fin >= 0.60: lv = "YUKSEK"
    elif fin >= 0.45: lv = "ORTA"
    elif fin >= 0.30: lv = "DIKKAT"
    else: lv = "DUSUK"
    return {"score": round(fin,4), "level": lv, "pattern": pt, "components": comp}


def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            zone_key TEXT,
            zone_name TEXT,
            region TEXT,
            lat REAL, lon REAL,
            sismik_score REAL,
            sismik_level TEXT,
            pattern TEXT,
            segment_score REAL,
            segment_level TEXT,
            combined_score REAL,
            combined_level TEXT,
            n_events INTEGER,
            b_value REAL,
            quiescence REAL,
            cff_total REAL,
            gps_vh REAL,
            slip_deficit_m REAL,
            coupling_ratio REAL,
            population_risk TEXT,
            source TEXT,
            risk_factors INTEGER,
            total_factors INTEGER
        )
    """)
    conn.commit()
    return conn


def scan_zone(key, zone, conn):
    """Tek bolge taramasi: sismik + segment + ek katmanlar."""
    name = zone["name"]
    lat = zone["lat"]
    lon = zone["lon"]
    region = zone.get("region", "?")
    ts = datetime.utcnow().isoformat()

    print(f"\n  [{key}] {name} ({region})")

    # 1. Sismik veri
    feats, meta = None, None
    n_events = 0
    source = "?"
    if HAS_ISC:
        try:
            feats, meta, err = fetch_and_analyze(lat, lon, 300, 2.5, use_cache=True)
            if meta:
                n_events = meta.get("n_total", 0)
                source = meta.get("source", "?")
        except: pass

    if feats is None:
        print(f"    Veri alinamadi, atlaniyor")
        return None

    print(f"    {n_events} olay ({source})")

    # 2. Risk tahmini
    risk = predict_risk(feats)
    sismik_score = risk["score"] if risk else 0
    sismik_level = risk["level"] if risk else "?"
    pattern = risk["pattern"] if risk else "?"
    print(f"    Sismik: {sismik_level} ({sismik_score*100:.1f}%)")

    # 3. Segment riski
    seg = compute_segment_risk(zone) if HAS_ZONES else {}
    seg_score = seg.get("segment_risk_score", 0) or 0
    seg_level = seg.get("segment_risk_level", "?")
    slip_deficit = seg.get("slip_deficit_m", 0)
    coupling = seg.get("coupling_ratio", 0)
    print(f"    Segment: {seg_level} ({seg_score*100:.1f}%)")

    # 4. Birlesik skor: %60 sismik + %40 segment
    combined = 0.6 * sismik_score + 0.4 * seg_score
    if combined >= 0.75: comb_level = "KRITIK"
    elif combined >= 0.60: comb_level = "YUKSEK"
    elif combined >= 0.45: comb_level = "ORTA"
    elif combined >= 0.30: comb_level = "DIKKAT"
    else: comb_level = "DUSUK"
    print(f"    Birlesik: {comb_level} ({combined*100:.1f}%)")

    # 5. Ek katmanlar
    cff_total = 0
    if HAS_CFF:
        try:
            cff = compute_cff_for_app(lat, lon)
            cff_total = cff.get("cff_total", 0) or 0
        except: pass

    gps_vh = None
    if HAS_GPS:
        try:
            gps = get_gps_info_for_app(lat, lon)
            gps_vh = gps.get("gps_mean_vh_mm_yr")
        except: pass

    # Risk faktorleri say
    risk_factors = 0
    total_factors = 4
    if sismik_score >= 0.30: risk_factors += 1
    if seg_score >= 0.30: risk_factors += 1
    if cff_total > 0.1: risk_factors += 1
    if gps_vh and gps_vh > 25: risk_factors += 1

    b_val = feats.get("w1_b_value") if feats else None
    qr = feats.get("quiescence_ratio") if feats else None

    # DB kaydet
    conn.execute("""
        INSERT INTO scans
        (timestamp, zone_key, zone_name, region, lat, lon,
         sismik_score, sismik_level, pattern,
         segment_score, segment_level,
         combined_score, combined_level,
         n_events, b_value, quiescence, cff_total, gps_vh,
         slip_deficit_m, coupling_ratio, population_risk,
         source, risk_factors, total_factors)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (ts, key, name, region, lat, lon,
          sismik_score, sismik_level, pattern,
          seg_score, seg_level,
          round(combined, 4), comb_level,
          n_events, b_val, qr, cff_total, gps_vh,
          slip_deficit, coupling, zone.get("population_risk"),
          source, risk_factors, total_factors))
    conn.commit()

    return {
        "key": key, "name": name, "region": region,
        "sismik": sismik_score, "segment": seg_score,
        "combined": round(combined, 4), "level": comb_level,
        "n_events": n_events, "risk_factors": risk_factors,
        "population": zone.get("population_risk", "?"),
    }


def run_global_scan():
    """Tum 25 bolgeyi tara."""
    if not HAS_ZONES:
        print("HATA: zone_database yuklenemedi")
        return

    print("=" * 70)
    print(f"GLOBAL TARAMA - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Bolge sayisi: {len(EXTENDED_ZONES)}")
    print("=" * 70)

    print(f"\nModuller: ISC={'OK' if HAS_ISC else 'X'} "
          f"Model={'OK' if HAS_MODEL else 'X'} "
          f"Fay={'OK' if HAS_FAULTS else 'X'} "
          f"CFF={'OK' if HAS_CFF else 'X'} "
          f"GPS={'OK' if HAS_GPS else 'X'}")

    conn = init_db()
    results = []

    # Oncelik sirasina gore tara
    sorted_zones = sorted(EXTENDED_ZONES.items(),
                          key=lambda x: x[1].get("priority", 9))

    for key, zone in sorted_zones:
        try:
            r = scan_zone(key, zone, conn)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  [{key}] HATA: {e}")
        time.sleep(2)

    conn.close()

    # Rapor
    print(f"\n{'='*70}")
    print("GLOBAL TARAMA RAPORU")
    print(f"{'='*70}")

    results.sort(key=lambda x: x["combined"], reverse=True)

    print(f"\n{'Bolge':<30} {'Birlesik':>9} {'Sismik':>8} {'Segment':>9} "
          f"{'Seviye':<8} {'Olay':>6} {'RF':>4} {'Nufus':<15}")
    print("-" * 100)

    for r in results:
        print(f"  {r['name']:<28} {r['combined']*100:>8.1f}% "
              f"{r['sismik']*100:>7.1f}% {r['segment']*100:>8.1f}% "
              f"{r['level']:<8} {r['n_events']:>6} "
              f"{r['risk_factors']}/4 {r['population']:<15}")

    # Kategori ozeti
    kritik = [r for r in results if r["level"] == "KRITIK"]
    yuksek = [r for r in results if r["level"] == "YUKSEK"]
    orta = [r for r in results if r["level"] == "ORTA"]
    dikkat = [r for r in results if r["level"] == "DIKKAT"]
    dusuk = [r for r in results if r["level"] == "DUSUK"]

    print(f"\n  KRITIK : {len(kritik)} bolge")
    for r in kritik:
        print(f"    - {r['name']} ({r['combined']*100:.1f}%, {r['population']})")

    print(f"  YUKSEK : {len(yuksek)} bolge")
    for r in yuksek:
        print(f"    - {r['name']} ({r['combined']*100:.1f}%)")

    print(f"  ORTA   : {len(orta)} bolge")
    print(f"  DIKKAT : {len(dikkat)} bolge")
    print(f"  DUSUK  : {len(dusuk)} bolge")

    # JSON kaydet
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_zones": len(results),
        "results": results,
        "summary": {
            "kritik": len(kritik),
            "yuksek": len(yuksek),
            "orta": len(orta),
            "dikkat": len(dikkat),
            "dusuk": len(dusuk),
        }
    }
    with open(OUTPUT_DIR / "global_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Rapor kaydedildi: {OUTPUT_DIR / 'global_report.json'}")
    return results


def show_report():
    """Son raporu goster."""
    report_path = OUTPUT_DIR / "global_report.json"
    if not report_path.exists():
        print("Rapor bulunamadi. Once --scan calistirin.")
        return

    with open(report_path) as f:
        report = json.load(f)

    print(f"Son tarama: {report['timestamp'][:16]}")
    print(f"Bolge: {report['n_zones']}")

    results = report["results"]
    results.sort(key=lambda x: x["combined"], reverse=True)

    print(f"\n{'Bolge':<30} {'Birlesik':>9} {'Seviye':<8} {'Nufus':<15}")
    print("-" * 65)
    for r in results:
        print(f"  {r['name']:<28} {r['combined']*100:>8.1f}% "
              f"{r['level']:<8} {r.get('population','?'):<15}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.scan:
        run_global_scan()
    elif args.report:
        show_report()
    else:
        print("Kullanim:")
        print("  python scripts/global_scan.py --scan")
        print("  python scripts/global_scan.py --report")


if __name__ == "__main__":
    main()