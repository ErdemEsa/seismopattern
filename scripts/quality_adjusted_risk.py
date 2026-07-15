#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Quality-Adjusted Short-Term Risk
=================================================
Debiased model + veri kalitesi cezasi + tektonik prior.

Adimlar:
1. Debiased model skoru (magnitude-free)
2. Veri kalitesi cezasi (dusuk veri = dusuk guven)
3. Tektonik prior (stabil kraton = daha dusuk prior)
4. Kalibrasyon (final olasiligin gercekci olmasi)

Cikti:
  {
    "st_raw": 0.52,
    "st_debiased": 0.52,
    "quality_penalty": 0.00,
    "tectonic_penalty": 0.00,
    "st_adjusted": 0.52,
    "confidence": "YUKSEK",
    "data_quality": "HIGH",
    "tectonic_class": "STRIKE_SLIP_BOUNDARY"
  }

Kullanim:
  python scripts/quality_adjusted_risk.py --lat 40.77 --lon 29.00
  python scripts/quality_adjusted_risk.py --lat 60.00 --lon 10.00
  python scripts/quality_adjusted_risk.py --test-all
"""

import json
import math
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from isc_fetch_v2 import fetch_and_analyze
    HAS_ISC = True
except Exception:
    HAS_ISC = False

OUTPUT_DIR = Path("output/quality_risk")
OUTPUT_DIR.mkdir(exist_ok=True)

DEB_MODEL_PATH = Path("output/counterfactual/debiased_model.joblib")
DEB_FEAT_PATH = Path("output/counterfactual/debiased_features.json")
MODEL_DIR = Path("output/models")


def load_debiased():
    if DEB_MODEL_PATH.exists() and DEB_FEAT_PATH.exists():
        model = joblib.load(DEB_MODEL_PATH)
        with open(DEB_FEAT_PATH) as f:
            feats = json.load(f)
        return model, feats
    return None, None


def load_standard():
    models = {}
    fl_path = MODEL_DIR / "feature_lists.json"
    if fl_path.exists():
        with open(fl_path) as f:
            fl = json.load(f)
        for pt in ["TIP_A", "TIP_B", "TIP_C"]:
            p = MODEL_DIR / f"model_{pt}.joblib"
            if p.exists():
                models[pt] = (joblib.load(p), fl.get(pt, []))
    return models


DEB_MODEL, DEB_FEATS = load_debiased()
STD_MODELS = load_standard()


# =========================================================
# TEKTONIK SINIFLANDIRMA
# =========================================================

def classify_tectonics(lat, lon):
    """
    Koordinattan tektonik rejimi tahmin et.
    Haversine mesafesi tabanlı kaba siniflandirma.
    """

    # Stabil kratonlar / intraplate
    stable_zones = [
        (60.0, 10.0, 800, "STABLE_CRATON"),    # Baltica
        (-25.0, 135.0, 1200, "STABLE_CRATON"), # Avustralya
        (55.0, -95.0, 1000, "STABLE_CRATON"),  # Kanada
        (-15.0, -47.0, 900, "STABLE_CRATON"),  # Brezilya
        (62.0, 26.0, 600, "STABLE_CRATON"),    # Fennoskandia
        (25.0, 5.0, 800, "STABLE_CRATON"),     # Sahara
        (60.0, 100.0, 900, "STABLE_CRATON"),   # Sibirya
        (-29.0, 24.0, 800, "STABLE_CRATON"),   # Guney Afrika
        (-20.0, 25.0, 700, "STABLE_CRATON"),   # Kalahari
        (45.0, 25.0, 500, "STABLE_CRATON"),    # Dogu Avrupa
    ]

    # Subduction zonlari
    subduction_zones = [
        (38.30, 142.37, 500, "SUBDUCTION"),    # Japonya
        (-35.0, -72.0, 800, "SUBDUCTION"),     # Sili
        (-12.0, -77.0, 600, "SUBDUCTION"),     # Peru
        (45.5, -125.0, 700, "SUBDUCTION"),     # Cascadia
        (1.0, 95.0, 600, "SUBDUCTION"),        # Sumatra
        (16.0, 119.5, 600, "SUBDUCTION"),      # Manila
        (33.0, 135.0, 400, "SUBDUCTION"),      # Nankai
        (28.0, 84.0, 500, "SUBDUCTION"),       # Himalaya
    ]

    # Dogrultu atimli sinirlar
    ss_zones = [
        (40.0, 29.0, 600, "STRIKE_SLIP"),      # KAF / Marmara
        (37.0, 37.0, 400, "STRIKE_SLIP"),      # DAF
        (34.0, -118.0, 500, "STRIKE_SLIP"),    # San Andreas
        (35.5, 35.5, 400, "STRIKE_SLIP"),      # Olu Deniz
    ]

    # Normal faylar / rift
    normal_zones = [
        (-2.0, 36.0, 800, "NORMAL_RIFT"),      # Dogu Afrika
        (35.0, 135.0, 400, "NORMAL_RIFT"),     # Japonya ic
    ]

    def hav_km(la1, lo1, la2, lo2):
        la1, lo1, la2, lo2 = map(math.radians, [la1, lo1, la2, lo2])
        dlat, dlon = la2 - la1, lo2 - lo1
        a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
        return 6371 * 2 * math.asin(math.sqrt(a))

    # En yakin zonu bul
    candidates = []
    for zones, tclass in [(stable_zones, None),
                          (subduction_zones, None),
                          (ss_zones, None),
                          (normal_zones, None)]:
        for z_lat, z_lon, z_radius, z_class in zones:
            d = hav_km(lat, lon, z_lat, z_lon)
            if d < z_radius:
                candidates.append((d, z_class))

    if not candidates:
        return "ACTIVE_BOUNDARY_GENERAL"

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# =========================================================
# VERİ KALİTESİ CEZASI
# =========================================================

def compute_quality_penalty(meta, features):
    """
    Veri kalitesine gore ceza hesapla.
    Dusuk kaliteli veri = dusuk guven = skoru asagi cek.

    Ceza aralik: 0.0 - 0.35
    """
    penalty = 0.0
    reasons = []

    b_quality = meta.get("b_quality", "LOW")
    n_total = meta.get("n_total", 0) or 0
    n_above_mc = meta.get("n_above_mc", 0) or 0
    b1_n = meta.get("b1_n", 0) or 0

    # b-kalitesi cezasi
    if b_quality == "LOW":
        penalty += 0.15
        reasons.append("b-degeri kalitesi DUSUK (+0.15 ceza)")
    elif b_quality == "MEDIUM":
        penalty += 0.07
        reasons.append("b-degeri kalitesi ORTA (+0.07 ceza)")

    # Toplam olay sayisi cezasi
    if n_total < 20:
        penalty += 0.12
        reasons.append(f"Cok az olay ({n_total}, +0.12 ceza)")
    elif n_total < 50:
        penalty += 0.08
        reasons.append(f"Az olay ({n_total}, +0.08 ceza)")
    elif n_total < 100:
        penalty += 0.04
        reasons.append(f"Orta olay ({n_total}, +0.04 ceza)")

    # Mc ustunde olay sayisi cezasi
    if n_above_mc < 15:
        penalty += 0.10
        reasons.append(f"Mc ustu az olay ({n_above_mc}, +0.10 ceza)")
    elif n_above_mc < 30:
        penalty += 0.05
        reasons.append(f"Mc ustu orta olay ({n_above_mc}, +0.05 ceza)")

    # Son 1 yil b-degeri cezasi
    if b1_n < 10:
        penalty += 0.06
        reasons.append(f"Son 1y b hesabi zayif (n={b1_n}, +0.06 ceza)")

    # w1_b_value yoksa
    if features.get("w1_b_value") is None:
        penalty += 0.05
        reasons.append("w1_b_value hesaplanamadi (+0.05 ceza)")

    # z-score'lar sifirsa
    z_all_zero = all(
        (features.get(k, 0) == 0 or features.get(k, 0) is None)
        for k in ["z_rate_1y", "z_b_value_1y", "z_max_mw_1y", "z_dist_1y"]
    )
    if z_all_zero:
        penalty += 0.04
        reasons.append("Z-score normalize eksik (+0.04 ceza)")

    # Toplam ceza siniri
    penalty = min(0.40, round(penalty, 4))

    # Kalite seviyesi
    if penalty <= 0.05:
        quality_level = "YUKSEK"
    elif penalty <= 0.15:
        quality_level = "ORTA"
    elif penalty <= 0.25:
        quality_level = "DUSUK"
    else:
        quality_level = "COK_DUSUK"

    return penalty, quality_level, reasons


# =========================================================
# TEKTONİK PRIOR CEZASI
# =========================================================

def compute_tectonic_penalty(tectonic_class, lt_score=None):
    """
    Tektonik rejime gore prior ceza.
    Stabil kraton = daha dusuk prior.

    Ceza aralik: 0.0 - 0.20
    """
    if tectonic_class == "STABLE_CRATON":
        penalty = 0.18
        reason = "Stabil kraton bolgesi (+0.18 ceza)"
    elif tectonic_class == "ACTIVE_BOUNDARY_GENERAL":
        penalty = 0.02
        reason = "Genel aktif sinir (+0.02 ceza)"
    elif tectonic_class == "NORMAL_RIFT":
        penalty = 0.05
        reason = "Normal fay/rift bolgesi (+0.05 ceza)"
    elif tectonic_class == "SUBDUCTION":
        penalty = 0.00
        reason = "Subduction zonu (ceza yok)"
    elif tectonic_class == "STRIKE_SLIP":
        penalty = 0.00
        reason = "Dogrultu atimli fay (ceza yok)"
    else:
        penalty = 0.03
        reason = f"Bilinmeyen tektonik ({tectonic_class}, +0.03 ceza)"

    return round(penalty, 4), reason


# =========================================================
# STANDARD MODEL SHORT-TERM SKORU
# =========================================================

def predict_standard_short_term(features):
    """Tip bazli standart model."""
    if not STD_MODELS:
        return None

    row = features.copy()
    c0 = float(row.get("count_0_1y", 0) or 0)
    c1 = float(row.get("count_1_2y", 0) or 0)
    c2 = float(row.get("count_2_3y", 0) or 0)
    row["count_linear_trend"] = c0 - c2
    row["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)

    try:
        row["b_drop_w3_w1"] = float(row.get("w3_b_value", 0) or 0) - \
                               float(row.get("w1_b_value", 0) or 0)
    except Exception:
        row["b_drop_w3_w1"] = np.nan

    try:
        row["spatial_focus_change"] = float(row.get("w3_mean_dist_km", 0) or 0) - \
                                       float(row.get("w1_mean_dist_km", 0) or 0)
    except Exception:
        row["spatial_focus_change"] = np.nan

    try:
        row["depth_change_km"] = float(row.get("w1_mean_depth_km", 0) or 0) - \
                                  float(row.get("w3_mean_depth_km", 0) or 0)
    except Exception:
        row["depth_change_km"] = np.nan

    qr = row.get("quiescence_ratio")
    acc = row.get("accel_90d")
    n3 = row.get("w3_n_events", 0) or 0

    if qr is None or pd.isna(qr) or n3 < 3:
        pt = "TIP_C"
    elif qr < 0.5:
        pt = "TIP_B"
    elif qr >= 1.0:
        pt = "TIP_A"
    elif 0.5 <= qr < 0.8:
        pt = "TIP_A" if (acc is not None and not pd.isna(acc) and acc >= 1.5) else "TIP_B"
    else:
        pt = "TIP_A"

    comp = {}
    for tip, (pipe, feats) in STD_MODELS.items():
        try:
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
            comp[tip] = float(pipe.predict_proba(x)[0, 1])
        except Exception:
            pass

    valid = {k: v for k, v in comp.items() if v is not None}
    if not valid:
        return None

    weights = {"TIP_A": 1.0, "TIP_B": 1.2, "TIP_C": 0.5}
    primary = valid.get(pt)
    ws = sum(weights.get(t, 1.0) * s for t, s in valid.items())
    wt = sum(weights.get(t, 1.0) for t in valid)
    ens = ws / wt if wt > 0 else 0.5

    return round(0.7 * primary + 0.3 * ens if primary else ens, 4)


# =========================================================
# DEBIASED MODEL SKORU
# =========================================================

def predict_debiased(features):
    """Magnitude-free debiased model."""
    if DEB_MODEL is None or DEB_FEATS is None:
        return None

    row = features.copy()
    c0 = float(row.get("count_0_1y", 0) or 0)
    c1 = float(row.get("count_1_2y", 0) or 0)
    c2 = float(row.get("count_2_3y", 0) or 0)
    row["count_linear_trend"] = c0 - c2
    row["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)

    try:
        row["b_drop_w3_w1"] = float(row.get("w3_b_value", 0) or 0) - \
                               float(row.get("w1_b_value", 0) or 0)
    except Exception:
        row["b_drop_w3_w1"] = np.nan

    try:
        row["spatial_focus_change"] = float(row.get("w3_mean_dist_km", 0) or 0) - \
                                       float(row.get("w1_mean_dist_km", 0) or 0)
    except Exception:
        row["spatial_focus_change"] = np.nan

    try:
        row["depth_change_km"] = float(row.get("w1_mean_depth_km", 0) or 0) - \
                                  float(row.get("w3_mean_depth_km", 0) or 0)
    except Exception:
        row["depth_change_km"] = np.nan

    x = pd.DataFrame([{f: row.get(f, np.nan) for f in DEB_FEATS}])
    return round(float(DEB_MODEL.predict_proba(x)[0, 1]), 4)

def _build_quality_adjusted_result(lat, lon, feats, meta, lt_score=None, ref_date=None):
    """
    Veri cekmeden, hazir feature + meta uzerinden quality-adjusted skor hesapla.
    """
    # 1. Standard model skoru
    st_raw = predict_standard_short_term(feats)

    # 2. Debiased model skoru
    st_debiased = predict_debiased(feats)
    if st_debiased is None:
        st_debiased = st_raw

    # 3. Veri kalitesi cezasi
    quality_penalty, quality_level, quality_reasons = compute_quality_penalty(
        meta, feats
    )

    # 4. Tektonik ceza
    tectonic_class = classify_tectonics(lat, lon)
    tectonic_penalty, tectonic_reason = compute_tectonic_penalty(
        tectonic_class, lt_score
    )

    reasons = list(quality_reasons) + [tectonic_reason]

    # 5. Ayarlanmis skor
    total_penalty = quality_penalty + tectonic_penalty
    st_adjusted = max(0.0, round((st_debiased or 0.0) - total_penalty, 4))

    # 6. Guven
    if quality_level == "YUKSEK" and tectonic_penalty < 0.05:
        confidence = "YUKSEK"
    elif quality_level in ("COK_DUSUK",) or tectonic_penalty >= 0.15:
        confidence = "DUSUK"
    elif quality_level == "DUSUK" or tectonic_penalty >= 0.08:
        confidence = "ORTA"
    else:
        confidence = "ORTA"

    def level(s):
        if s >= 0.75: return "KRITIK"
        if s >= 0.60: return "YUKSEK"
        if s >= 0.45: return "ORTA"
        if s >= 0.30: return "DIKKAT"
        return "DUSUK"

    return {
        "lat": lat,
        "lon": lon,
        "ref_date": ref_date,
        "st_raw": st_raw,
        "st_raw_level": level(st_raw) if st_raw is not None else None,
        "st_debiased": st_debiased,
        "st_debiased_level": level(st_debiased) if st_debiased is not None else None,
        "quality_penalty": quality_penalty,
        "tectonic_penalty": tectonic_penalty,
        "total_penalty": round(total_penalty, 4),
        "st_adjusted": st_adjusted,
        "st_adjusted_level": level(st_adjusted),
        "confidence": confidence,
        "data_quality": quality_level,
        "tectonic_class": tectonic_class,
        "reasons": reasons,
        "meta": {
            "n_total": meta.get("n_total"),
            "n_above_mc": meta.get("n_above_mc"),
            "mc": meta.get("mc"),
            "b_quality": meta.get("b_quality"),
            "b1_n": meta.get("b1_n"),
            "source": meta.get("source"),
        },
    }


def compute_quality_adjusted_from_data(lat, lon, feats, meta, lt_score=None, ref_date=None):
    """
    app / dual_risk tarafinda tekrar veri cekmeden
    hazir feature seti uzerinden quality-adjusted skor hesapla.
    """
    return _build_quality_adjusted_result(
        lat=lat,
        lon=lon,
        feats=feats,
        meta=meta,
        lt_score=lt_score,
        ref_date=ref_date
    )


# =========================================================
# ANA FONKSİYON
# =========================================================

def compute_quality_adjusted_risk(lat, lon, ref_date=None,
                                   lt_score=None, min_mag=2.5,
                                   radius_km=300):
    """
    Tam kalite-ayarli kisa vadeli risk hesapla.

    Returns:
        dict: {
            st_raw: float,
            st_debiased: float,
            quality_penalty: float,
            tectonic_penalty: float,
            st_adjusted: float,
            confidence: str,
            data_quality: str,
            tectonic_class: str,
            reasons: list,
        }
    """
    # 1. Veri cek
    if HAS_ISC:
        feats, meta, err = fetch_and_analyze(
            lat, lon, radius_km, min_mag,
            ref_date=ref_date, use_cache=True
        )
    else:
        feats, meta, err = None, None, "ISC yok"

    if feats is None:
        return {
            "error": err,
            "st_raw": None,
            "st_debiased": None,
            "st_adjusted": None,
        }

    # 2. Standard model skoru
    st_raw = predict_standard_short_term(feats)

    # 3. Debiased model skoru
    st_debiased = predict_debiased(feats)
    if st_debiased is None:
        st_debiased = st_raw

    # 4. Veri kalitesi cezasi
    quality_penalty, quality_level, quality_reasons = compute_quality_penalty(
        meta, feats
    )

    # 5. Tektonik ceza
    tectonic_class = classify_tectonics(lat, lon)
    tectonic_penalty, tectonic_reason = compute_tectonic_penalty(
        tectonic_class, lt_score
    )

    reasons = quality_reasons + [tectonic_reason]

    # 6. Ayarlanmis skor
    # Debiased modeli baz al, uzerine kalite ve tektonik ceza uygula
    total_penalty = quality_penalty + tectonic_penalty
    st_adjusted = max(0.0, round(st_debiased - total_penalty, 4))

    # 7. Guven hesabi
    if quality_level == "YUKSEK" and tectonic_penalty < 0.05:
        confidence = "YUKSEK"
    elif quality_level in ("COK_DUSUK",) or tectonic_penalty >= 0.15:
        confidence = "DUSUK"
    elif quality_level == "DUSUK" or tectonic_penalty >= 0.08:
        confidence = "ORTA"
    else:
        confidence = "ORTA"

    # 8. Seviye
    def level(s):
        if s >= 0.75: return "KRITIK"
        if s >= 0.60: return "YUKSEK"
        if s >= 0.45: return "ORTA"
        if s >= 0.30: return "DIKKAT"
        return "DUSUK"

    return _build_quality_adjusted_result(
        lat=lat,
        lon=lon,
        feats=feats,
        meta=meta,
        lt_score=lt_score,
        ref_date=ref_date
    )
    


# =========================================================
# TOPLU TEST
# =========================================================

def test_all():
    """Bilinen lokasyonlar uzerinde test et."""
    print("=" * 72)
    print("QUALITY-ADJUSTED RISK - TOPLU TEST")
    print("=" * 72)

    locations = [
        ("Marmara",         40.77,  29.00, None,         None),
        ("Cascadia",        45.50, -125.0, None,         None),
        ("Manila",          16.00, 119.50, None,         None),
        ("Norveç (stabil)", 60.00,  10.00, None,         None),
        ("Avustralya ic",  -25.00, 135.00, None,         None),
        ("Kanada kalkan",   55.00, -95.00, None,         None),
        ("Sibirya",         60.00, 100.00, None,         None),
        ("Izmit oncesi",    40.75,  29.86, "1999-07-17", None),
        ("Tohoku oncesi",   38.30, 142.37, "2011-02-11", None),
        ("K.Maras oncesi",  37.22,  37.02, "2023-01-06", None),
    ]

    results = []
    print(f"\n{'Bolge':<22} {'RAW':>7} {'DEB':>7} {'ADJ':>7} "
          f"{'Q_PEN':>7} {'T_PEN':>7} {'CONF':<8} {'TECT'}")
    print("-" * 92)

    for name, lat, lon, ref, lt in locations:
        try:
            r = compute_quality_adjusted_risk(lat, lon, ref, lt)
            if r.get("error"):
                print(f"  {name:<20} HATA: {r['error']}")
                continue

            raw = r.get("st_raw", 0) or 0
            deb = r.get("st_debiased", 0) or 0
            adj = r.get("st_adjusted", 0) or 0
            qp  = r.get("quality_penalty", 0)
            tp  = r.get("tectonic_penalty", 0)
            conf= r.get("confidence", "?")
            tect= r.get("tectonic_class", "?")[:20]

            print(f"  {name:<20} {raw*100:>6.1f}% {deb*100:>6.1f}% "
                  f"{adj*100:>6.1f}% {qp:>7.3f} {tp:>7.3f} "
                  f"{conf:<8} {tect}")

            results.append({"name": name, **r})

        except Exception as e:
            print(f"  {name}: HATA {e}")

    # Ayrım analizi
    print(f"\n{'='*72}")
    print("AYRIMI ANALİZİ")
    print(f"{'='*72}")

    aktif = [r for r in results if r.get("tectonic_class") not in
             ("STABLE_CRATON",) and r.get("st_adjusted") is not None]
    stabil = [r for r in results if r.get("tectonic_class") == "STABLE_CRATON"
              and r.get("st_adjusted") is not None]

    if aktif:
        adj_aktif = [r["st_adjusted"] for r in aktif]
        print(f"\nAktif bölge adjusted skorlar:")
        for r in aktif:
            print(f"  {r['name']:<22} {r['st_adjusted']*100:.1f}%")
        print(f"  Ortalama: {np.mean(adj_aktif)*100:.1f}%")

    if stabil:
        adj_stabil = [r["st_adjusted"] for r in stabil]
        print(f"\nStabil bölge adjusted skorlar:")
        for r in stabil:
            print(f"  {r['name']:<22} {r['st_adjusted']*100:.1f}%")
        print(f"  Ortalama: {np.mean(adj_stabil)*100:.1f}%")

    if aktif and stabil:
        ayrim = np.mean(adj_aktif) - np.mean(adj_stabil)
        print(f"\n  Ayrim (aktif - stabil): {ayrim*100:.1f}%")
        if ayrim > 0.15:
            print(f"  BASARILI: Model aktif bölgeleri stabil bölgelerden ayirt ediyor")
        else:
            print(f"  YETERSIZ: Ayrim hala zayif")

    # Kaydet
    report_path = OUTPUT_DIR / "quality_adjusted_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRapor: {report_path}")

    return results


# =========================================================
# API YARDIMCıSI (app.py'den cagirilacak)
# =========================================================

def get_adjusted_risk_for_app(lat, lon, ref_date=None, lt_score=None):
    """app.py'den cagirilacak ana fonksiyon."""
    return compute_quality_adjusted_risk(lat, lon, ref_date, lt_score)


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--refdate", type=str, default=None)
    ap.add_argument("--lt-score", type=float, default=None)
    ap.add_argument("--test-all", action="store_true")
    args = ap.parse_args()

    if args.test_all:
        test_all()
    elif args.lat is not None and args.lon is not None:
        r = compute_quality_adjusted_risk(
            args.lat, args.lon, args.refdate, args.lt_score
        )
        if r.get("error"):
            print(f"Hata: {r['error']}")
        else:
            print(f"\nKonum: {args.lat}, {args.lon}")
            print(f"Tektonik: {r['tectonic_class']}")
            print(f"Veri kalitesi: {r['data_quality']}")
            print(f"\nSkorlar:")
            print(f"  ST Raw      : {(r['st_raw'] or 0)*100:.1f}%")
            print(f"  ST Debiased : {(r['st_debiased'] or 0)*100:.1f}%")
            print(f"  Kalite ceza : -{r['quality_penalty']*100:.1f}%")
            print(f"  Tektonik ceza: -{r['tectonic_penalty']*100:.1f}%")
            print(f"  ST Adjusted : {r['st_adjusted']*100:.1f}% "
                  f"[{r['st_adjusted_level']}]")
            print(f"  Guven       : {r['confidence']}")
            print(f"\nGerekce:")
            for reason in r["reasons"]:
                print(f"  - {reason}")
    else:
        print("Kullanim:")
        print("  python scripts/quality_adjusted_risk.py --test-all")
        print("  python scripts/quality_adjusted_risk.py --lat 40.77 --lon 29.00")
        print("  python scripts/quality_adjusted_risk.py --lat 60.0 --lon 10.0")


if __name__ == "__main__":
    main()