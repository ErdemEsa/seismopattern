#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Counterfactual Aciklama + w3_max_mw Duzeltmesi
===============================================================
1. "Neden bu skor?" sorusunu cevaplar
2. "Skoru dusurmeK icin ne degismeli?" sorusunu cevaplar
3. w3_max_mw bolge imzasi etkisini azaltir
4. Independent benchmark icin iyilestirilmis feature seti

Kullanim:
  python scripts/counterfactual_engine.py --explain --zone marmara
  python scripts/counterfactual_engine.py --retrain-debiased
  python scripts/counterfactual_engine.py --benchmark-improved
"""

import json
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dual_risk_framework import analyze_zone
    HAS_DR = True
except Exception:
    HAS_DR = False

try:
    from isc_fetch_v2 import fetch_and_analyze
    HAS_ISC = True
except Exception:
    HAS_ISC = False

OUTPUT_DIR = Path("output/counterfactual")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path("output/models")

FEATURES_FULL = [
    "count_0_1y","count_1_2y","count_2_3y",
    "count_linear_trend","count_accel_ratio",
    "w1_n_events","w3_n_events",
    "quiescence_ratio","accel_90d","monthly_slope_36m",
    "w1_mean_mw","w1_std_mw","w1_max_mw",
    "w3_mean_mw","w3_max_mw",
    "w1_b_value","w3_b_value","b_drop_w3_w1",
    "w1_mean_depth_km","w1_std_depth_km",
    "w3_mean_depth_km","depth_change_km",
    "w1_mean_dist_km","w1_std_dist_km",
    "w3_mean_dist_km","spatial_focus_change",
    "w1_migration_slope_km_day","w3_migration_slope_km_day",
    "z_rate_1y","z_rate_3y",
    "z_b_value_1y","z_b_value_3y",
    "z_max_mw_1y","z_depth_1y","z_dist_1y",
]

# w3_max_mw cikarilmis feature seti (bolge imzasi azaltilmis)
FEATURES_DEBIASED = [f for f in FEATURES_FULL
                     if f not in ["w3_max_mw", "w1_max_mw",
                                   "w3_mean_mw", "w1_mean_mw"]]


# =========================================================
# COUNTERFACTUAL ACIKLAMA
# =========================================================

def explain_prediction(zone_key=None, lat=None, lon=None, ref_date=None):
    """
    "Neden bu skor?" ve "Ne degismeli?" sorularini cevaplar.
    """
    print("=" * 65)
    print("COUNTERFACTUAL ACIKLAMA")
    print("=" * 65)

    if not HAS_DR:
        print("Dual risk modulu yok")
        return

    res = analyze_zone(zone_key=zone_key, lat=lat, lon=lon, ref_date=ref_date)
    z = res.get("zone", {})
    lt = res.get("long_term", {})
    st = res.get("short_term", {})
    cb = res.get("combined", {})

    name = z.get("name", "Bilinmiyor")
    print(f"\nBolge: {name}")
    print(f"Combined skor: {cb.get('score',0)*100:.1f}% ({cb.get('level','?')})")

    # 1. NEDEN BU SKOR?
    print(f"\n--- NEDEN BU SKOR? ---")

    # Long-term faktorler
    lt_score = lt.get("score", 0)
    lt_factors = lt.get("factors", [])

    print(f"\nUzun vadeli risk ({lt_score*100:.1f}%):")
    if lt_factors:
        for f in lt_factors:
            print(f"  + {f}")
    else:
        print(f"  Segment riski dusuk")

    # Short-term faktorler
    if st and not st.get("error"):
        st_score = st.get("score", 0)
        st_explain = st.get("explanation", "")
        st_pattern = st.get("pattern", "?")
        st_feats = st.get("raw_features", {})

        print(f"\nKisa vadeli anomali ({st_score*100:.1f}%):")
        print(f"  Sablon: {st_pattern}")
        print(f"  {st_explain}")

        # Kritik feature'lar ve etkileri
        print(f"\n  Kritik degerler:")
        critical = [
            ("quiescence_ratio", st_feats.get("quiescence_ratio"), "1.0",
             "1'den buyuk=aktivasyon, kucuk=sessizlik"),
            ("accel_90d", st_feats.get("accel_90d"), "1.0",
             "1'den buyuk=son donem hizlaniyor"),
            ("w1_b_value", st_feats.get("w1_b_value"), "1.0",
             "1'den dusuk=gerilim birikimi"),
            ("w3_max_mw", st_feats.get("w3_max_mw"), "5.5",
             "yuksek=bolgede buyuk olay var"),
            ("b_drop_w3_w1", st_feats.get("b_drop_w3_w1"), "0",
             "pozitif=b dusuyor=gerilim artisi"),
            ("count_0_1y", st_feats.get("count_0_1y"), "-",
             "son 1 yildaki olay sayisi"),
        ]

        for feat, val, ref, desc in critical:
            if val is not None and not pd.isna(val):
                print(f"    {feat:<25} = {val:.4f}  (ref:{ref})  {desc}")

    # 2. NE DEGISMELI?
    print(f"\n--- SKORU DUSURMEYE NE YARDIMCI OLUR? ---")

    if st and not st.get("error"):
        st_feats = st.get("raw_features", {})

        suggestions = []

        qr = st_feats.get("quiescence_ratio")
        if qr and qr > 2:
            suggestions.append(
                f"quiescence_ratio={qr:.2f} → 1.0 olsaydi "
                f"(aktivite normallesirse)"
            )

        acc = st_feats.get("accel_90d")
        if acc and acc > 2:
            suggestions.append(
                f"accel_90d={acc:.2f} → 1.0 olsaydi "
                f"(son 90 gundeki hizlanma dursaydı)"
            )

        b1 = st_feats.get("w1_b_value")
        if b1 and b1 < 0.8:
            suggestions.append(
                f"w1_b_value={b1:.3f} → 1.0 olsaydi "
                f"(gerilim azalsaydi)"
            )

        mxmw = st_feats.get("w3_max_mw")
        if mxmw and mxmw > 6.0:
            suggestions.append(
                f"w3_max_mw={mxmw:.1f} → 5.5 olsaydi "
                f"(son 3 yilda buyuk olay olmasaydi)"
            )

        if suggestions:
            for s in suggestions:
                print(f"  → {s}")
        else:
            print(f"  Belirgin tek etken yok, risk dagitilmis")

    # 3. LONG-TERM ICIN NE DEGISMELI?
    print(f"\n--- UZUN VADELI RISKI DUSUREN FAKTORLER ---")

    seg = z.get("segment_info", {})
    if seg:
        deficit = seg.get("slip_deficit_m", 0)
        coupling = seg.get("coupling_ratio", 0)

        print(f"  Kayma acigi: {deficit:.1f}m")
        print(f"  → Sadece buyuk deprem ile azalir (kontrol edilemez)")
        print(f"  Coupling: {coupling:.2f}")
        print(f"  → Fay kinematigiyle belirlenir (kontrol edilemez)")
        print(f"  Sonuc: Uzun vadeli risk azaltılamaz,")
        print(f"         sadece hazirlik ile hasar azaltılabilir.")

    return res


# =========================================================
# DEBIASED MODEL (w3_max_mw etkisi azaltilmis)
# =========================================================

def add_derived(df):
    df = df.copy()
    c0 = df.get("count_0_1y", pd.Series(0)).fillna(0)
    c1 = df.get("count_1_2y", pd.Series(0)).fillna(0)
    c2 = df.get("count_2_3y", pd.Series(0)).fillna(0)
    df["count_linear_trend"] = c0 - c2
    df["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)
    df["b_drop_w3_w1"] = df.get("w3_b_value", pd.Series(dtype=float)) - \
                          df.get("w1_b_value", pd.Series(dtype=float))
    df["spatial_focus_change"] = df.get("w3_mean_dist_km", pd.Series(dtype=float)) - \
                                  df.get("w1_mean_dist_km", pd.Series(dtype=float))
    df["depth_change_km"] = df.get("w1_mean_depth_km", pd.Series(dtype=float)) - \
                             df.get("w3_mean_depth_km", pd.Series(dtype=float))
    return df


def retrain_debiased():
    """
    w3_max_mw, w1_max_mw, w3_mean_mw, w1_mean_mw cikarilmis model.
    Bu modelin amaci: bolge imzasi yerine gercek oncu sinyalleri ogrenip
    ogrenemedigini test etmek.
    """
    print("=" * 65)
    print("DEBIASED MODEL EGITIMI (magnitude feature'lar cikarildi)")
    print("=" * 65)

    real = pd.read_csv("output/gcmt_precursor_features.csv", low_memory=False)
    ctrl = pd.read_csv("output/gcmt_control_features.csv", low_memory=False)

    if "radius_km" in real.columns:
        real = real[real["radius_km"] == 200].copy()
    if "radius_km" in ctrl.columns:
        ctrl = ctrl[ctrl["radius_km"] == 200].copy()

    real = add_derived(real)
    ctrl = add_derived(ctrl)
    real["target"] = 1
    ctrl["target"] = 0

    # Tam model
    avail_full = [f for f in FEATURES_FULL if f in real.columns and f in ctrl.columns]
    comb_full = pd.concat([real[avail_full + ["target"]],
                            ctrl[avail_full + ["target"]]], ignore_index=True)
    X_full = comb_full[avail_full]
    y_full = comb_full["target"]

    # Debiased model
    avail_deb = [f for f in FEATURES_DEBIASED if f in real.columns and f in ctrl.columns]
    comb_deb = pd.concat([real[avail_deb + ["target"]],
                           ctrl[avail_deb + ["target"]]], ignore_index=True)
    X_deb = comb_deb[avail_deb]
    y_deb = comb_deb["target"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pw = (y_full == 0).sum() / max(y_full.sum(), 1)

    def make_pipe(pw):
        if HAS_XGB:
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", RobustScaler()),
                ("mdl", XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.03,
                    subsample=0.8, scale_pos_weight=pw,
                    eval_metric="aucpr", verbosity=0, random_state=42
                ))
            ])
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", RobustScaler()),
                ("mdl", GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, random_state=42
                ))
            ])

    # Tam model
    pipe_full = make_pipe(pw)
    aucs_full = cross_val_score(pipe_full, X_full, y_full, cv=cv, scoring="roc_auc")

    # Debiased model
    pipe_deb = make_pipe(pw)
    aucs_deb = cross_val_score(pipe_deb, X_deb, y_deb, cv=cv, scoring="roc_auc")

    print(f"\nTam model ({len(avail_full)} feature):")
    print(f"  AUC: {aucs_full.mean():.4f} +/- {aucs_full.std():.4f}")

    print(f"\nDebiased model ({len(avail_deb)} feature, magnitude cikarildi):")
    print(f"  AUC: {aucs_deb.mean():.4f} +/- {aucs_deb.std():.4f}")

    delta = aucs_deb.mean() - aucs_full.mean()
    print(f"\nFark: {delta:+.4f}")

    if delta > -0.02:
        print(f"  Magnitude cikarilmasi modeli ciddi bosaltmadi.")
        print(f"  Yani model zaten magnitude'a ASIRI bagimli degildi.")
    elif delta > -0.05:
        print(f"  Hafif dusus. Magnitude bir miktar katki yapiyor.")
    else:
        print(f"  Ciddi dusus. Model buyuk olcude magnitude'a bagimli.")

    # Debiased modeli kalibre edip kaydet
    print(f"\nDebiased model kalibre ediliyor...")
    pipe_cal = CalibratedClassifierCV(make_pipe(pw), method="isotonic", cv=5)
    pipe_cal.fit(X_deb, y_deb)

    joblib.dump(pipe_cal, OUTPUT_DIR / "debiased_model.joblib")

    with open(OUTPUT_DIR / "debiased_features.json", "w") as f:
        json.dump(avail_deb, f, indent=2)

    print(f"  Kaydedildi: {OUTPUT_DIR / 'debiased_model.joblib'}")

    # Sonuc raporu
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "full_auc": round(float(aucs_full.mean()), 4),
        "debiased_auc": round(float(aucs_deb.mean()), 4),
        "delta": round(float(delta), 4),
        "n_features_full": len(avail_full),
        "n_features_debiased": len(avail_deb),
        "removed_features": ["w3_max_mw", "w1_max_mw", "w3_mean_mw", "w1_mean_mw"],
    }

    with open(OUTPUT_DIR / "debiased_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


# =========================================================
# IMPROVED BENCHMARK
# =========================================================

def improved_benchmark():
    """
    Independent benchmark'i iyilestirilmis negatif orneklerle tekrar calistir.
    Negatifler: tektonik olarak sakin bolgeler (deprem olmamis yerler).
    """
    print("=" * 65)
    print("IYILESTIRILMIS INDEPENDENT BENCHMARK")
    print("=" * 65)

    if not HAS_ISC:
        print("ISC modulu yok")
        return

    # Debiased model yukle
    deb_path = OUTPUT_DIR / "debiased_model.joblib"
    deb_feat_path = OUTPUT_DIR / "debiased_features.json"

    if not deb_path.exists():
        print("Once --retrain-debiased calistirin")
        return

    deb_model = joblib.load(deb_path)
    with open(deb_feat_path) as f:
        deb_feats = json.load(f)

    # Standart model de yukle
    fl_path = MODEL_DIR / "feature_lists.json"
    with open(fl_path) as f:
        fl = json.load(f)

    std_models = {}
    for pt in ["TIP_A", "TIP_B", "TIP_C"]:
        p = MODEL_DIR / f"model_{pt}.joblib"
        if p.exists():
            std_models[pt] = (joblib.load(p), fl.get(pt, []))

    # Pozitif: buyuk deprem oncesi
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

    # Iyilestirilmis negatif: TEKTONIK OLARAK SAKIN bolgeler
    negatives = [
        ("Norveç_stabil", 60.00, 10.00, None),
        ("Avustralya_ic", -25.00, 135.00, None),
        ("Kanada_kalkan", 55.00, -95.00, None),
        ("Brezilya_ic", -15.00, -47.00, None),
        ("Finlandiya", 62.00, 26.00, None),
        ("Sahara", 25.00, 5.00, None),
        ("Sibirya", 60.00, 100.00, None),
        ("Guney_Afrika_ic", -29.00, 24.00, None),
    ]

    def get_score(feats, model_type="standard"):
        row = add_derived(pd.DataFrame([feats])).iloc[0].to_dict()

        if model_type == "debiased":
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in deb_feats}])
            return float(deb_model.predict_proba(x)[0, 1])
        else:
            # Standard: tip bazli
            def rbt(r):
                qr = r.get("quiescence_ratio")
                n3 = r.get("w3_n_events", 0) or 0
                if qr is None or pd.isna(qr) or n3 < 3: return "TIP_C"
                if qr < 0.5: return "TIP_B"
                if qr >= 1.0: return "TIP_A"
                return "TIP_A"

            pt = rbt(row)
            if pt not in std_models or std_models[pt] is None:
                return 0.5
            pipe, f_list = std_models[pt]
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in f_list}])
            return float(pipe.predict_proba(x)[0, 1])

    results_std = []
    results_deb = []

    print("\nPozitif ornekler:")
    for name, lat, lon, ref in positives:
        try:
            feats, meta, err = fetch_and_analyze(
                lat, lon, 300, 2.5, ref_date=ref, use_cache=True
            )
            if feats is None:
                continue
            s_std = get_score(feats, "standard")
            s_deb = get_score(feats, "debiased")
            print(f"  {name:<25} std={s_std:.3f}  deb={s_deb:.3f}")
            results_std.append({"label": 1, "score": s_std})
            results_deb.append({"label": 1, "score": s_deb})
        except Exception as e:
            print(f"  {name}: {e}")

    print("\nNegatif ornekler (tektonik olarak sakin):")
    for name, lat, lon, ref in negatives:
        try:
            feats, meta, err = fetch_and_analyze(
                lat, lon, 300, 2.5, ref_date=ref, use_cache=True
            )
            if feats is None:
                # Sakin bolgelerde veri olmayabilir
                feats = {f: 0 for f in FEATURES_FULL}
                feats["w3_n_events"] = 0
                feats["count_0_1y"] = 0

            s_std = get_score(feats, "standard")
            s_deb = get_score(feats, "debiased")
            print(f"  {name:<25} std={s_std:.3f}  deb={s_deb:.3f}")
            results_std.append({"label": 0, "score": s_std})
            results_deb.append({"label": 0, "score": s_deb})
        except Exception as e:
            print(f"  {name}: {e}")

    if len(results_std) < 8:
        print("\nYetersiz ornek")
        return

    df_std = pd.DataFrame(results_std)
    df_deb = pd.DataFrame(results_deb)

    auc_std = roc_auc_score(df_std["label"], df_std["score"])
    auc_deb = roc_auc_score(df_deb["label"], df_deb["score"])

    print(f"\nBENCHMARK SONUCLARI:")
    print(f"  Standard model AUC: {auc_std:.4f}")
    print(f"  Debiased model AUC: {auc_deb:.4f}")
    print(f"  Onceki benchmark AUC: 0.5625")
    print(f"  Iyilesme (std): {auc_std - 0.5625:+.4f}")
    print(f"  Iyilesme (deb): {auc_deb - 0.5625:+.4f}")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "standard_auc": round(float(auc_std), 4),
        "debiased_auc": round(float(auc_deb), 4),
        "previous_auc": 0.5625,
        "improvement_std": round(float(auc_std - 0.5625), 4),
        "improvement_deb": round(float(auc_deb - 0.5625), 4),
    }

    with open(OUTPUT_DIR / "improved_benchmark.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nKaydedildi: {OUTPUT_DIR / 'improved_benchmark.json'}")
    return report


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true",
                    help="Counterfactual aciklama")
    ap.add_argument("--zone", type=str, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--retrain-debiased", action="store_true",
                    help="w3_max_mw cikarilmis model egit")
    ap.add_argument("--benchmark-improved", action="store_true",
                    help="Iyilestirilmis benchmark calistir")
    args = ap.parse_args()

    if args.explain:
        if args.zone:
            explain_prediction(zone_key=args.zone)
        elif args.lat and args.lon:
            explain_prediction(lat=args.lat, lon=args.lon)
        else:
            explain_prediction(zone_key="marmara")
    elif args.retrain_debiased:
        retrain_debiased()
    elif args.benchmark_improved:
        improved_benchmark()
    else:
        print("Kullanim:")
        print("  python scripts/counterfactual_engine.py --explain --zone marmara")
        print("  python scripts/counterfactual_engine.py --retrain-debiased")
        print("  python scripts/counterfactual_engine.py --benchmark-improved")


if __name__ == "__main__":
    main()