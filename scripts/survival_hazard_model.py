#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Survival / Hazard Layer
=======================================
Gercek bir deterministic zaman tahmini degil;
ufuk bazli (30g / 90g / 1y / 5y) tehlike skoru.

Bu katman:
- long-term segment hazard
- short-term anomaly (quality-adjusted)
- CFF
- NLP tarihsel destek
ile horizon bazli hazard uretir.

Kullanim:
  python scripts/survival_hazard_model.py --zone marmara
  python scripts/survival_hazard_model.py --zone cascadia
  python scripts/survival_hazard_model.py --benchmark
"""

import json
import math
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))

# moduller
try:
    from zone_database import EXTENDED_ZONES, compute_segment_risk
    HAS_ZONES = True
except Exception:
    HAS_ZONES = False
    EXTENDED_ZONES = {}

try:
    from fault_distance import get_fault_info_for_app
    HAS_FAULTS = True
except Exception:
    HAS_FAULTS = False

try:
    from gps_velocity import get_gps_info_for_app
    HAS_GPS = True
except Exception:
    HAS_GPS = False

try:
    from coulomb_simple import compute_cff_for_app
    HAS_CFF = True
except Exception:
    HAS_CFF = False

try:
    from nlp_scanner import get_nlp_info_for_app
    HAS_NLP = True
except Exception:
    HAS_NLP = False

try:
    from isc_fetch_v2 import fetch_and_analyze
    HAS_ISC = True
except Exception:
    HAS_ISC = False

try:
    from quality_adjusted_risk import compute_quality_adjusted_risk
    HAS_QAR = True
except Exception:
    HAS_QAR = False

OUTPUT_DIR = Path("output/survival_hazard")
OUTPUT_DIR.mkdir(exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def get_zone_by_key(key):
    if key in EXTENDED_ZONES:
        z = dict(EXTENDED_ZONES[key])
        z["key"] = key
        return z
    return None


def get_zone_context(lat, lon):
    best = None
    best_dist = 999999
    for key, z in EXTENDED_ZONES.items():
        d = haversine_km(lat, lon, z["lat"], z["lon"])
        if d < best_dist and d < 500:
            best = dict(z)
            best["key"] = key
            best_dist = d
    return best


def compute_long_term_score(zone, lat=None, lon=None):
    """
    Segment risk + fault + GPS ile uzun vadeli skor.
    """
    if zone is None:
        return {"score": 0.0, "level": "DUSUK", "factors": []}

    seg = compute_segment_risk(zone)
    score = float(seg.get("segment_risk_score", 0) or 0)
    factors = list(seg.get("segment_factors", []))

    # Fay katkisi
    if HAS_FAULTS and lat is not None and lon is not None:
        try:
            fi = get_fault_info_for_app(lat, lon)
            ff = fi.get("features", {})
            dist = ff.get("nearest_fault_dist_km")
            complexity = ff.get("fault_complexity", 0) or 0

            if dist is not None:
                if dist < 10:
                    score += 0.05
                    factors.append(f"Cok yakin fay ({dist} km)")
                elif dist < 30:
                    score += 0.03
                    factors.append(f"Yakin fay ({dist} km)")

            if complexity >= 20:
                score += 0.03
                factors.append(f"Yuksek fay karmasikligi ({complexity})")
        except Exception:
            pass

    # GPS katkisi
    if HAS_GPS and lat is not None and lon is not None:
        try:
            gps = get_gps_info_for_app(lat, lon)
            vh = gps.get("gps_mean_vh_mm_yr")
            strain = gps.get("gps_strain_rate")
            nst = gps.get("gps_n_stations", 0) or 0

            if nst >= 5:
                if vh is not None:
                    if vh >= 30:
                        score += 0.05
                        factors.append(f"Yuksek GPS hiz ({vh} mm/yil)")
                    elif vh >= 20:
                        score += 0.03
                        factors.append(f"Orta-yuksek GPS hiz ({vh} mm/yil)")
                if strain is not None:
                    if strain >= 1e-4:
                        score += 0.05
                        factors.append(f"Yuksek strain rate ({strain})")
                    elif strain >= 5e-5:
                        score += 0.03
                        factors.append(f"Orta strain rate ({strain})")
        except Exception:
            pass

    score = min(1.0, round(score, 4))

    if score >= 0.75: level = "KRITIK"
    elif score >= 0.60: level = "YUKSEK"
    elif score >= 0.45: level = "ORTA"
    elif score >= 0.30: level = "DIKKAT"
    else: level = "DUSUK"

    return {
        "score": score,
        "level": level,
        "factors": factors,
        "slip_deficit_m": seg.get("slip_deficit_m"),
        "coupling_ratio": seg.get("coupling_ratio"),
    }


def compute_hazard_scores(zone, long_term, short_term, cff_info=None, nlp_info=None):
    """
    Discrete-time hazard / horizon score.
    Deterministic zaman tahmini degil.

    Formul:
      base_lambda = 1 / recurrence_years
      segment_multiplier = f(long_term)
      anomaly_multiplier = f(short_term_adjusted)
      cff_multiplier = f(CFF)
      nlp_multiplier = f(history)
      quality_multiplier = f(confidence)
    """
    if zone is None:
        return {}

    rec = zone.get("recurrence_years", 250)
    base_lambda = 1.0 / rec

    lt_score = long_term.get("score", 0) if long_term else 0
    st_score = short_term.get("st_adjusted", 0) if short_term else 0
    confidence = short_term.get("confidence", "ORTA") if short_term else "ORTA"

    # 1) Segment multiplier (0.5 .. 2.3)
    segment_mult = 0.5 + 1.8 * lt_score

    # 2) Anomaly multiplier (0.4 .. 2.6)
    anomaly_mult = 0.4 + 2.2 * st_score

    # 3) CFF multiplier
    cff_mult = 1.0
    cff_total = 0
    if cff_info:
        cff_total = cff_info.get("cff_total", 0) or 0
        if cff_total >= 5:
            cff_mult = 1.50
        elif cff_total >= 1:
            cff_mult = 1.30
        elif cff_total >= 0.1:
            cff_mult = 1.10

    # 4) NLP multiplier
    nlp_mult = 1.0
    if nlp_info:
        n_hist = len(nlp_info.get("historical", []))
        n_types = nlp_info.get("summary", {}).get("n_types", 0)
        if n_hist >= 2:
            nlp_mult += 0.10
        elif n_hist == 1:
            nlp_mult += 0.05
        if n_types >= 4:
            nlp_mult += 0.05

    # 5) Quality multiplier
    if confidence == "YUKSEK":
        quality_mult = 1.00
    elif confidence == "ORTA":
        quality_mult = 0.90
    else:
        quality_mult = 0.75

    # 6) Toplam multiplier
    total_mult = segment_mult * anomaly_mult * cff_mult * nlp_mult * quality_mult
    total_mult = max(0.15, min(4.0, total_mult))

    def p_days(days):
        years = days / 365.25
        p = 1 - math.exp(-base_lambda * years * total_mult)
        return round(p * 100, 2)

    # ufuk skorları
    h30 = p_days(30)
    h90 = p_days(90)
    h1y = p_days(365)
    h5y = p_days(365 * 5)

    return {
        "30d": h30,
        "90d": h90,
        "1y": h1y,
        "5y": h5y,
        "base_lambda": round(base_lambda, 6),
        "segment_multiplier": round(segment_mult, 3),
        "anomaly_multiplier": round(anomaly_mult, 3),
        "cff_multiplier": round(cff_mult, 3),
        "nlp_multiplier": round(nlp_mult, 3),
        "quality_multiplier": round(quality_mult, 3),
        "total_multiplier": round(total_mult, 3),
        "inputs": {
            "lt_score": round(float(lt_score), 4),
            "st_adjusted": round(float(st_score), 4),
            "cff_total": round(float(cff_total), 6),
            "confidence": confidence,
        }
    }


def analyze_hazard(zone_key=None, lat=None, lon=None, ref_date=None):
    if zone_key:
        zone = get_zone_by_key(zone_key)
        if zone is None:
            raise ValueError(f"Bolge bulunamadi: {zone_key}")
        lat = zone["lat"]
        lon = zone["lon"]
    else:
        zone = get_zone_context(lat, lon)

    # long-term
    lt = compute_long_term_score(zone, lat=lat, lon=lon)

    # short-term (quality adjusted)
    if HAS_QAR:
        st = compute_quality_adjusted_risk(
            lat=lat, lon=lon, ref_date=ref_date, lt_score=lt["score"]
        )
    else:
        st = {"st_adjusted": 0.0, "confidence": "DUSUK"}

    # cff
    cff = None
    if HAS_CFF:
        try:
            cff = compute_cff_for_app(lat, lon, ref_date)
        except Exception:
            cff = None

    # nlp
    nlp = None
    if HAS_NLP:
        try:
            nlp = get_nlp_info_for_app(lat, lon)
        except Exception:
            nlp = None

    horizons = compute_hazard_scores(zone, lt, st, cff, nlp)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "zone": zone,
        "lat": lat,
        "lon": lon,
        "ref_date": ref_date,
        "long_term": lt,
        "short_term": st,
        "cff": cff,
        "nlp": nlp,
        "hazards": horizons,
    }


def run_benchmark():
    """
    Horizon bazli benchmark.
    Pozitif orneklerde horizon skorlarinin yuksek olmasi beklenir.
    """
    print("=" * 72)
    print("SURVIVAL / HAZARD BENCHMARK")
    print("=" * 72)

    positives = [
        ("Kahramanmaras_2023", 37.22, 37.02, "2023-01-06"),
        ("Izmit_1999", 40.75, 29.86, "1999-07-17"),
        ("Tohoku_2011", 38.30, 142.37, "2011-02-11"),
        ("Maule_2010", -35.85, -72.72, "2010-01-27"),
        ("Landers_1992", 34.20, -116.44, "1992-05-28"),
        ("Ridgecrest_2019", 35.77, -117.60, "2019-06-06"),
        ("Nepal_2015", 28.23, 84.73, "2015-03-25"),
        ("Sumatra_2004", 3.30, 95.98, "2004-11-26"),
    ]

    negatives = [
        ("Norvec_stabil", 60.00, 10.00, None),
        ("Avustralya_ic", -25.00, 135.00, None),
        ("Sibirya", 60.00, 100.00, None),
        ("Guney_Afrika", -29.00, 24.00, None),
        ("Kanada_kalkan", 55.00, -95.00, None),
    ]

    rows = []

    print("\nPozitif ornekler:")
    for name, lat, lon, ref in positives:
        try:
            r = analyze_hazard(lat=lat, lon=lon, ref_date=ref)
            hz = r["hazards"]
            print(f"  {name:<22} 30d={hz['30d']:.2f}%  90d={hz['90d']:.2f}%  1y={hz['1y']:.2f}%  5y={hz['5y']:.2f}%")
            rows.append({"case": name, "label": 1, **hz})
        except Exception as e:
            print(f"  {name}: {e}")

    print("\nNegatif ornekler:")
    for name, lat, lon, ref in negatives:
        try:
            r = analyze_hazard(lat=lat, lon=lon, ref_date=ref)
            hz = r["hazards"]
            print(f"  {name:<22} 30d={hz['30d']:.2f}%  90d={hz['90d']:.2f}%  1y={hz['1y']:.2f}%  5y={hz['5y']:.2f}%")
            rows.append({"case": name, "label": 0, **hz})
        except Exception as e:
            print(f"  {name}: {e}")

    if len(rows) < 8:
        print("\nYetersiz benchmark")
        return None

    df = pd.DataFrame(rows)

    from sklearn.metrics import roc_auc_score, average_precision_score

    result = {}
    print("\nHorizon AUC Sonuclari:")
    print(f"{'Ufuk':<8} {'AUC':>8} {'AP':>8}")
    print("-" * 28)

    for key in ["30d", "90d", "1y", "5y"]:
        auc = roc_auc_score(df["label"], df[key])
        ap = average_precision_score(df["label"], df[key])
        result[key] = {"auc": round(float(auc), 4), "ap": round(float(ap), 4)}
        print(f"{key:<8} {auc:>8.4f} {ap:>8.4f}")

    out = OUTPUT_DIR / "hazard_benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "rows": rows,
            "results": result
        }, f, indent=2, default=str)

    print(f"\nKaydedildi: {out}")
    return result


def run_all_zones():
    print("=" * 72)
    print("SURVIVAL / HAZARD TUM BOLGELER")
    print("=" * 72)

    rows = []
    for key in EXTENDED_ZONES.keys():
        try:
            r = analyze_hazard(zone_key=key)
            hz = r["hazards"]
            lt = r["long_term"]["score"]
            st = r["short_term"]["st_adjusted"]
            rows.append({
                "key": key,
                "name": r["zone"]["name"],
                "lt": lt,
                "st_adj": st,
                "30d": hz["30d"],
                "90d": hz["90d"],
                "1y": hz["1y"],
                "5y": hz["5y"],
                "mult": hz["total_multiplier"],
            })
            print(f"  {r['zone']['name']:<28} LT={lt*100:>5.1f}% ST={st*100:>5.1f}% 1y=%{hz['1y']}")
        except Exception as e:
            print(f"  {key}: {e}")

    df = pd.DataFrame(rows).sort_values("1y", ascending=False)
    out = OUTPUT_DIR / "hazard_table.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\nIlk 10 bolge (1y hazard):")
    print(df.head(10)[["name", "lt", "st_adj", "30d", "1y", "5y"]].to_string(index=False))
    print(f"\nKaydedildi: {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", type=str, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--refdate", type=str, default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.all:
        run_all_zones()
    elif args.zone:
        r = analyze_hazard(zone_key=args.zone, ref_date=args.refdate)
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    elif args.lat is not None and args.lon is not None:
        r = analyze_hazard(lat=args.lat, lon=args.lon, ref_date=args.refdate)
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
    else:
        print("Kullanim:")
        print("  python scripts/survival_hazard_model.py --zone marmara")
        print("  python scripts/survival_hazard_model.py --benchmark")
        print("  python scripts/survival_hazard_model.py --all")


if __name__ == "__main__":
    main()