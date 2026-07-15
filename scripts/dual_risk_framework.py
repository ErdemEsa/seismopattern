#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Dual Risk Framework
===================================
Uzun vadeli segment riski ile kısa vadeli sismik anomali riskini ayırır.

Katmanlar:
1. Long-term Segment Risk
   - Slip deficit
   - Coupling ratio
   - Recurrence ratio
   - Unruptured segment ratio
   - GPS strain / velocity katkısı
   - Fay yakınlığı / karmaşıklığı

2. Short-term Seismic Anomaly Risk
   - Kalibre edilmiş model skoru
   - Quiescence / accel / b-value
   - CFF etkisi
   - Son 1-3 yıl anomalileri

3. Combined Operational Risk
   - Uzun + kısa vadeyi birleştirir
   - 30g / 90g / 1y / 5y ufuk skorları
   - Açıklama ve öncelik üretir

Kullanim:
  python scripts/dual_risk_framework.py --zone marmara
  python scripts/dual_risk_framework.py --zone cascadia
  python scripts/dual_risk_framework.py --all
  python scripts/dual_risk_framework.py --lat 40.77 --lon 29.00
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

# ---------------------------------------------------------
# Moduller
# ---------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))

HAS_ISC = HAS_FAULTS = HAS_CFF = HAS_GPS = HAS_NLP = HAS_ZONES = False

try:
    from isc_fetch_v2 import fetch_and_analyze
    HAS_ISC = True
except Exception:
    pass

try:
    from fault_distance import get_fault_info_for_app
    HAS_FAULTS = True
except Exception:
    pass

try:
    from coulomb_simple import compute_cff_for_app
    HAS_CFF = True
except Exception:
    pass

try:
    from gps_velocity import get_gps_info_for_app
    HAS_GPS = True
except Exception:
    pass

try:
    from nlp_scanner import get_nlp_info_for_app
    HAS_NLP = True
except Exception:
    pass

try:
    from zone_database import EXTENDED_ZONES, compute_segment_risk
    HAS_ZONES = True
except Exception:
    EXTENDED_ZONES = {}

try:
    from quality_adjusted_risk import compute_quality_adjusted_from_data
    HAS_QAR = True
except Exception:
    HAS_QAR = False

try:
    from survival_hazard_model import compute_hazard_scores
    HAS_HAZARD = True
except Exception:
    HAS_HAZARD = False

import joblib

MODEL_DIR = Path("output/models")
CAL_DIR = Path("output/calibrated_models")
OUTPUT_DIR = Path("output/dual_risk")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------
def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def risk_level(score):
    if score >= 0.75:
        return "KRITIK"
    elif score >= 0.60:
        return "YUKSEK"
    elif score >= 0.45:
        return "ORTA"
    elif score >= 0.30:
        return "DIKKAT"
    return "DUSUK"


def rule_based_type(r):
    qr = r.get("quiescence_ratio")
    acc = r.get("accel_90d")
    n3 = r.get("w3_n_events", 0) or 0

    if qr is None or pd.isna(qr) or n3 < 3:
        return "TIP_C"
    if qr < 0.5:
        return "TIP_B"
    if qr >= 1.0:
        return "TIP_A"
    if 0.5 <= qr < 0.8:
        if acc is not None and not pd.isna(acc) and acc >= 1.5:
            return "TIP_A"
        return "TIP_B"
    return "TIP_A"


def add_derived(row):
    r = dict(row)
    c0 = float(r.get("count_0_1y", 0) or 0)
    c1 = float(r.get("count_1_2y", 0) or 0)
    c2 = float(r.get("count_2_3y", 0) or 0)

    r["count_linear_trend"] = c0 - c2
    r["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)

    def safe_diff(a, b):
        try:
            if a is None or b is None:
                return np.nan
            return float(a) - float(b)
        except Exception:
            return np.nan

    r["b_drop_w3_w1"] = safe_diff(r.get("w3_b_value"), r.get("w1_b_value"))
    r["spatial_focus_change"] = safe_diff(r.get("w3_mean_dist_km"), r.get("w1_mean_dist_km"))
    r["depth_change_km"] = safe_diff(r.get("w1_mean_depth_km"), r.get("w3_mean_depth_km"))

    if not r.get("quiescence_ratio"):
        prev = (c1 + c2) / 2.0
        r["quiescence_ratio"] = c0 / prev if prev > 0 else np.nan

    if not r.get("w3_n_events"):
        r["w3_n_events"] = c0 + c1 + c2
    if not r.get("w1_n_events"):
        r["w1_n_events"] = c0

    return r


# ---------------------------------------------------------
# Model yukleme
# ---------------------------------------------------------
def load_models():
    models = {}
    feature_lists = {}

    fl_path = MODEL_DIR / "feature_lists.json"
    if fl_path.exists():
        with open(fl_path, "r", encoding="utf-8") as f:
            feature_lists = json.load(f)

    for pt in ["TIP_A", "TIP_B", "TIP_C"]:
        p = MODEL_DIR / f"model_{pt}.joblib"
        if p.exists():
            try:
                models[pt] = (joblib.load(p), feature_lists.get(pt, []))
            except Exception:
                pass

    cal_path = CAL_DIR / "calibrated_model.joblib"
    cal_feat = CAL_DIR / "calibrated_features.json"
    if cal_path.exists():
        try:
            cal_model = joblib.load(cal_path)
            if cal_feat.exists():
                with open(cal_feat, "r", encoding="utf-8") as f:
                    cal_feats = json.load(f)
            else:
                cal_feats = feature_lists.get("TIP_A", [])
            models["CALIBRATED"] = (cal_model, cal_feats)
        except Exception:
            pass

    return models, feature_lists


MODELS, FEATURE_LISTS = load_models()


# ---------------------------------------------------------
# Short-term (kisa vadeli) risk
# ---------------------------------------------------------
def predict_short_term_from_features(features):
    """
    Kisa vadeli sismik anomali riskini hesaplar.
    """
    if not MODELS:
        return None

    row = add_derived(features)
    pt = rule_based_type(row)

    component_scores = {}

    for tip, payload in MODELS.items():
        try:
            pipe, feats = payload
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
            p = float(pipe.predict_proba(x)[0, 1])
            component_scores[tip] = round(p, 4)
        except Exception:
            component_scores[tip] = None

    valid = {k: v for k, v in component_scores.items()
             if v is not None and k in ["TIP_A", "TIP_B", "TIP_C"]}

    if not valid:
        return None

    weights = {"TIP_A": 1.0, "TIP_B": 1.2, "TIP_C": 0.5}
    primary = valid.get(pt)

    ws = sum(weights.get(t, 1.0) * s for t, s in valid.items())
    wt = sum(weights.get(t, 1.0) for t in valid)
    ensemble = ws / wt if wt > 0 else 0.5

    final = 0.7 * primary + 0.3 * ensemble if primary is not None else ensemble

    calibrated = None
    if "CALIBRATED" in MODELS:
        try:
            cal_pipe, cal_feats = MODELS["CALIBRATED"]
            x_cal = pd.DataFrame([{f: row.get(f, np.nan) for f in cal_feats}])
            calibrated = float(cal_pipe.predict_proba(x_cal)[0, 1])
            final = 0.70 * final + 0.30 * calibrated
            component_scores["CALIBRATED"] = round(calibrated, 4)
        except Exception:
            pass

    final = round(float(final), 4)

    # confidence
    confidence_issues = []
    n_total = row.get("w3_n_events", 0) or 0

    if n_total < 20:
        confidence_issues.append("Az veri")
    if row.get("w1_b_value") is None:
        confidence_issues.append("b(1y) hesaplanamadi")

    z_all_zero = all(
        (row.get(k, 0) == 0 or row.get(k, 0) is None)
        for k in ["z_rate_1y", "z_b_value_1y", "z_max_mw_1y", "z_dist_1y"]
    )
    if z_all_zero:
        confidence_issues.append("Z-score eksik")

    if n_total < 50 and row.get("w1_b_value") is None:
        confidence = "DUSUK"
    elif len(confidence_issues) >= 2:
        confidence = "DUSUK"
    elif len(confidence_issues) == 1:
        confidence = "ORTA"
    else:
        confidence = "YUKSEK"

    explanation = []
    if pt == "TIP_A":
        explanation.append("Aktivasyon sablonu")
    elif pt == "TIP_B":
        explanation.append("Sismik sessizlik sablonu")
    else:
        explanation.append("Belirsiz sablon")

    qr = row.get("quiescence_ratio")
    if qr is not None and not pd.isna(qr):
        if qr < 0.5:
            explanation.append(f"aktivite % {round((1-qr)*100)} azaldi")
        elif qr > 1.5:
            explanation.append(f"aktivite % {round((qr-1)*100)} artti")

    b1 = row.get("w1_b_value")
    b3 = row.get("w3_b_value")
    if b1 is not None and b3 is not None and not pd.isna(b1) and not pd.isna(b3):
        if float(b3) > float(b1):
            explanation.append(f"b dustu ({float(b3):.2f}->{float(b1):.2f})")
        else:
            explanation.append(f"b artti ({float(b3):.2f}->{float(b1):.2f})")

    return {
        "score": final,
        "level": risk_level(final),
        "pattern": pt,
        "confidence": confidence,
        "confidence_issues": confidence_issues,
        "component_scores": component_scores,
        "explanation": " | ".join(explanation),
        "raw_features": row,
    }


# ---------------------------------------------------------
# Long-term (uzun vadeli) segment riski
# ---------------------------------------------------------
def compute_long_term_risk(zone, fault_info=None, gps_info=None):
    """
    Uzun vadeli segment riskini hesaplar.
    zone_database.py içindeki segment riskini temel alır,
    fault + gps ile küçük düzeltmeler yapar.
    """
    if zone is None:
        return None

    seg = compute_segment_risk(zone)
    base = seg.get("segment_risk_score", 0) or 0
    factors = list(seg.get("segment_factors", []))

    score = float(base)

    # Fay yakınlığı katkısı
    if fault_info and fault_info.get("features"):
        ff = fault_info["features"]
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

    # GPS katkısı
    if gps_info:
        vh = gps_info.get("gps_mean_vh_mm_yr")
        strain = gps_info.get("gps_strain_rate")
        nst = gps_info.get("gps_n_stations", 0) or 0

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

    score = min(1.0, round(score, 4))

    return {
        "score": score,
        "level": risk_level(score),
        "factors": factors,
        "base_segment_score": base,
        "slip_deficit_m": seg.get("slip_deficit_m"),
        "coupling_ratio": seg.get("coupling_ratio"),
    }


# ---------------------------------------------------------
# Birleşik operasyonel risk
# ---------------------------------------------------------
def combine_risks(long_term, short_term, cff_info=None, nlp_info=None):
    """
    Uzun ve kısa vadeyi ayırıp birleştirir.
    """

    lt = long_term["score"] if long_term else 0.0
    st = short_term["score"] if short_term else 0.0

    # CFF katkısı (kısa vadeye daha yakın)
    cff_boost = 0.0
    if cff_info:
        cff_total = cff_info.get("cff_total", 0) or 0
        if cff_total >= 5:
            cff_boost = 0.08
        elif cff_total >= 1:
            cff_boost = 0.05
        elif cff_total >= 0.1:
            cff_boost = 0.02

    # NLP tarihsel katkı (uzun vadeye hafif)
    nlp_boost = 0.0
    if nlp_info:
        n_hist = len(nlp_info.get("historical", []))
        if n_hist >= 2:
            nlp_boost = 0.03
        elif n_hist == 1:
            nlp_boost = 0.015

    # Dinamik ağırlıklar:
    # kısa vadeli sinyal çok güçlüyse kısa vade daha baskın
    if st >= 0.60:
        w_lt, w_st = 0.45, 0.55
    elif st >= 0.45:
        w_lt, w_st = 0.55, 0.45
    else:
        w_lt, w_st = 0.65, 0.35

    combined = w_lt * lt + w_st * st + cff_boost + nlp_boost
    combined = min(1.0, max(0.0, combined))
    combined = round(float(combined), 4)

    return {
        "score": combined,
        "level": risk_level(combined),
        "weights": {"long_term": w_lt, "short_term": w_st},
        "cff_boost": round(cff_boost, 4),
        "nlp_boost": round(nlp_boost, 4),
    }


# ---------------------------------------------------------
# Ufuk (horizon) skorları
# ---------------------------------------------------------
def compute_horizon_scores(zone, long_term, short_term, combined):
    """
    30g / 90g / 1y / 5y için operasyonel olasılık benzeri skorlar.
    Bu deterministik zaman tahmini değildir.
    """

    if zone is None:
        return {}

    rec = zone.get("recurrence_years", 250)
    last = zone.get("last_major")

    # Poisson temel oran
    lam = 1.0 / rec

    # short-term anomaly multiplier
    st = short_term["score"] if short_term else 0.0
    lt = long_term["score"] if long_term else 0.0
    comb = combined["score"] if combined else 0.0

    # anomaly multiplier 0.6 .. 2.4
    anomaly_mult = 0.6 + 1.8 * st

    # segment multiplier 0.7 .. 2.0
    segment_mult = 0.7 + 1.3 * lt

    # overall multiplier
    total_mult = min(3.0, anomaly_mult * segment_mult)

    def p_for_days(days):
        years = days / 365.25
        p = 1 - math.exp(-lam * years * total_mult)
        return round(p * 100, 2)

    scores = {
        "30d": p_for_days(30),
        "90d": p_for_days(90),
        "1y": p_for_days(365),
        "5y": p_for_days(365 * 5),
        "multiplier": round(total_mult, 3),
    }

    return scores


# ---------------------------------------------------------
# Bölge bulma
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Ana analiz
# ---------------------------------------------------------
def analyze_zone(zone_key=None, lat=None, lon=None, ref_date=None):
    """
    Tek bölge için dual risk analizi.
    """

    if zone_key:
        zone = get_zone_by_key(zone_key)
        if zone is None:
            raise ValueError(f"Bölge bulunamadi: {zone_key}")
        lat = zone["lat"]
        lon = zone["lon"]
    else:
        zone = get_zone_context(lat, lon)

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "zone": zone,
        "lat": lat,
        "lon": lon,
        "ref_date": ref_date,
    }

    # 1) Kisa vadeli sismik veri
    feats, meta, err = None, None, "Veri yok"
    if HAS_ISC:
        feats, meta, err = fetch_and_analyze(
            lat, lon, 300, 2.5, ref_date=ref_date, use_cache=True
        )

    # 2) Fay
    fault_info = None
    if HAS_FAULTS:
        try:
            fault_info = get_fault_info_for_app(lat, lon)
            result["faults"] = fault_info
        except Exception as e:
            result["faults"] = {"error": str(e)}

    # 3) GPS
    gps_info = None
    if HAS_GPS:
        try:
            gps_info = get_gps_info_for_app(lat, lon)
            result["gps"] = gps_info
        except Exception as e:
            result["gps"] = {"error": str(e)}

    # 4) CFF
    cff_info = None
    if HAS_CFF:
        try:
            cff_info = compute_cff_for_app(lat, lon, ref_date)
            result["cff"] = cff_info
        except Exception as e:
            result["cff"] = {"error": str(e)}

    # 5) NLP
    nlp_info = None
    if HAS_NLP:
        try:
            nlp_info = get_nlp_info_for_app(lat, lon)
            result["nlp"] = nlp_info
        except Exception as e:
            result["nlp"] = {"error": str(e)}

    # 6) Long-term
    lt = compute_long_term_risk(zone, fault_info=fault_info, gps_info=gps_info)
    result["long_term"] = lt

    # 7) Short-term (quality-adjusted)
    if feats is not None and HAS_QAR:
        st = compute_quality_adjusted_from_data(
            lat=lat,
            lon=lon,
            feats=feats,
            meta=meta,
            lt_score=lt["score"],
            ref_date=ref_date
        )
        st["features"] = feats
        st["meta"] = meta
        result["short_term"] = st
    elif feats is not None:
        # fallback
        st = predict_short_term_from_features(feats)
        result["short_term"] = st
    else:
        result["short_term"] = {"error": err}

    # 8) Combined
    st_for_combined = {"score": 0.0}
    if result.get("short_term") and not result["short_term"].get("error"):
        if "st_adjusted" in result["short_term"]:
            st_for_combined = {"score": result["short_term"]["st_adjusted"],
                               "confidence": result["short_term"].get("confidence")}
        else:
            st_for_combined = result["short_term"]

    combined = combine_risks(lt, st_for_combined, cff_info=cff_info, nlp_info=nlp_info)
    result["combined"] = combined

    # 9) Horizons
    if HAS_HAZARD:
        result["horizons"] = compute_hazard_scores(
            zone=zone,
            long_term=lt,
            short_term=result["short_term"] if not result["short_term"].get("error") else {"st_adjusted": 0.0, "confidence": "DUSUK"},
            cff_info=cff_info,
            nlp_info=nlp_info
        )
    else:
        result["horizons"] = compute_horizon_scores(
            zone, lt, st_for_combined, cff_info, nlp_info
        )

    # 10) Açıklama
    result["summary"] = build_summary(result)

    return clean_nans(result)


def build_summary(result):
    lines = []

    z = result.get("zone") or {}
    lt = result.get("long_term") or {}
    st = result.get("short_term") or {}
    cb = result.get("combined") or {}
    hz = result.get("horizons") or {}
    cff = result.get("cff") or {}
    gps = result.get("gps") or {}

    lines.append(f"BOLGE: {z.get('name', 'Bilinmiyor')}")
    lines.append("")

    lines.append(f"UZUN VADELI SEGMENT RISKI: {lt.get('level','?')} ({(lt.get('score',0)*100):.1f}%)")
    if lt.get("factors"):
        for f in lt["factors"][:6]:
            lines.append(f"  - {f}")
    lines.append("")

    if st and not st.get("error"):
        st_level = st.get("st_adjusted_level", st.get("level", "?"))
        st_score = st.get("st_adjusted", st.get("score", 0))
        lines.append(f"KISA VADELI SISMIK ANOMALI: {st_level} ({(st_score*100):.1f}%)")

        if "st_raw" in st:
            lines.append(f"  Ham skor      : {(st.get('st_raw',0)*100):.1f}%")
            lines.append(f"  Debiased skor : {(st.get('st_debiased',0)*100):.1f}%")
            lines.append(f"  Ayarlanmis    : {(st.get('st_adjusted',0)*100):.1f}%")
            lines.append(f"  Kalite cezasi : -{st.get('quality_penalty',0)*100:.1f}%")
            lines.append(f"  Tektonik ceza : -{st.get('tectonic_penalty',0)*100:.1f}%")
            lines.append(f"  Guven         : {st.get('confidence','?')}")

        if st.get("explanation"):
            lines.append(f"  {st['explanation']}")
        lines.append("")

    lines.append(f"Nihai OPERASYONEL RISK: {cb.get('level','?')} ({(cb.get('score',0)*100):.1f}%)")
    lines.append(f"  Agirliklar: UzunVade={cb.get('weights',{}).get('long_term','?')} | KisaVade={cb.get('weights',{}).get('short_term','?')}")
    lines.append(f"  CFF katkisi: +{cb.get('cff_boost',0):.3f}")
    lines.append(f"  NLP katkisi: +{cb.get('nlp_boost',0):.3f}")
    lines.append("")

    if cff and not cff.get("error"):
        lines.append(f"CFF: {cff.get('cff_total',0):.6f} bar | kaynak={cff.get('cff_n_sources',0)}")
    if gps and not gps.get("error"):
        lines.append(f"GPS: Vh={gps.get('gps_mean_vh_mm_yr','?')} mm/yil | strain={gps.get('gps_strain_rate','?')}")
    lines.append("")

    if z:
        lines.append(f"Beklenen buyukluk: Mw {z.get('expected_mw','?')}")
        lines.append(f"Son buyuk deprem: {z.get('last_major','?')} (Mw {z.get('last_major_mw','?')})")
        if z.get("population_risk"):
            lines.append(f"Nufus riski: {z['population_risk']}")
        lines.append("")

    if hz:
        lines.append("UFUK SKORLARI (deterministik zaman tahmini degil):")
        lines.append(f"  30 gun : %{hz.get('30d','?')}")
        lines.append(f"  90 gun : %{hz.get('90d','?')}")
        lines.append(f"  1 yil  : %{hz.get('1y','?')}")
        lines.append(f"  5 yil  : %{hz.get('5y','?')}")
        lines.append(f"  anomaly multiplier: {hz.get('multiplier','?')}")
        lines.append("")

    lines.append("NOT: Bu sistem deprem zamani tahmin etmez;")
    lines.append("uzun vadeli segment tehlikesi ile kisa vadeli sismik anomalileri ayirir.")

    return "\n".join(lines)


# ---------------------------------------------------------
# Tüm bölgeler
# ---------------------------------------------------------
def run_all_zones():
    results = []
    print("=" * 72)
    print("SEISMOPATTERN DUAL RISK FRAMEWORK")
    print(f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Bolge sayisi: {len(EXTENDED_ZONES)}")
    print("=" * 72)

    for key in EXTENDED_ZONES:
        print(f"\n[{key}]")
        try:
            res = analyze_zone(zone_key=key)
            lt = res["long_term"]["score"]
            st = res["short_term"]["score"] if not res["short_term"].get("error") else 0.0
            cb = res["combined"]["score"]
            print(f"  LT={lt:.3f} ({res['long_term']['level']}) | "
                  f"ST={st:.3f} ({res['short_term'].get('level','?')}) | "
                  f"COMB={cb:.3f} ({res['combined']['level']})")
            results.append(res)
        except Exception as e:
            print(f"  HATA: {e}")

    # özet tablo
    rows = []
    for r in results:
        z = r.get("zone") or {}
        rows.append({
            "key": z.get("key"),
            "name": z.get("name"),
            "region": z.get("region"),
            "long_term_score": r.get("long_term", {}).get("score"),
            "long_term_level": r.get("long_term", {}).get("level"),
            "short_term_score": r.get("short_term", {}).get("score") if not r.get("short_term", {}).get("error") else None,
            "short_term_level": r.get("short_term", {}).get("level") if not r.get("short_term", {}).get("error") else None,
            "combined_score": r.get("combined", {}).get("score"),
            "combined_level": r.get("combined", {}).get("level"),
            "h30": r.get("horizons", {}).get("30d"),
            "h90": r.get("horizons", {}).get("90d"),
            "h1y": r.get("horizons", {}).get("1y"),
            "h5y": r.get("horizons", {}).get("5y"),
        })

    df = pd.DataFrame(rows).sort_values("combined_score", ascending=False)

    print("\n" + "=" * 72)
    print("DUAL RISK RAPORU")
    print("=" * 72)
    print(f"\n{'Bolge':<30} {'LT':>8} {'ST':>8} {'COMB':>8} {'30d':>8} {'1y':>8}")
    print("-" * 75)

    for _, row in df.iterrows():
        print(f"  {row['name'][:28]:<28} "
              f"{(row['long_term_score'] or 0)*100:>7.1f}% "
              f"{(row['short_term_score'] or 0)*100:>7.1f}% "
              f"{(row['combined_score'] or 0)*100:>7.1f}% "
              f"{str(row['h30'])+'%':>8} "
              f"{str(row['h1y'])+'%':>8}")

    out_json = OUTPUT_DIR / "dual_risk_report.json"
    out_csv = OUTPUT_DIR / "dual_risk_table.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nKaydedildi:")
    print(f"  {out_json}")
    print(f"  {out_csv}")

    return results, df


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", type=str, default=None, help="Bolge anahtari")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--refdate", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        run_all_zones()
        return

    if args.zone:
        res = analyze_zone(zone_key=args.zone, ref_date=args.refdate)
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return

    if args.lat is not None and args.lon is not None:
        res = analyze_zone(lat=args.lat, lon=args.lon, ref_date=args.refdate)
        print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return

    print("Kullanim:")
    print("  python scripts/dual_risk_framework.py --zone marmara")
    print("  python scripts/dual_risk_framework.py --all")
    print("  python scripts/dual_risk_framework.py --lat 40.77 --lon 29.00")
    print("  python scripts/dual_risk_framework.py --zone tohoku --refdate 2011-02-11")


if __name__ == "__main__":
    main()