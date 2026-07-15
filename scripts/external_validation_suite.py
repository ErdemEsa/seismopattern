#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - External Validation Suite
=========================================
Harici / bagimsiz dogrulama paketi.

TESTLER:
1. Temporal external validation
2. Leave-one-region-out validation
3. Independent historical benchmark (ISC+USGS)

Kullanim:
  python scripts/external_validation_suite.py --all
  python scripts/external_validation_suite.py --temporal
  python scripts/external_validation_suite.py --regions
  python scripts/external_validation_suite.py --benchmark
"""

import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

# ISC+USGS özellik üretici
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from isc_fetch_v2 import fetch_and_analyze
    HAS_ISC = True
except Exception:
    HAS_ISC = False

OUTPUT_DIR = Path("output/external_validation")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
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


# =========================================================
# ORTAK YARDIMCI FONKSIYONLAR
# =========================================================

def add_derived(df):
    df = df.copy()
    c0 = df.get("count_0_1y", pd.Series(0, index=df.index)).fillna(0)
    c1 = df.get("count_1_2y", pd.Series(0, index=df.index)).fillna(0)
    c2 = df.get("count_2_3y", pd.Series(0, index=df.index)).fillna(0)

    df["count_linear_trend"] = c0 - c2
    df["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)
    df["b_drop_w3_w1"] = df.get("w3_b_value", pd.Series(dtype=float)) - \
                         df.get("w1_b_value", pd.Series(dtype=float))
    df["spatial_focus_change"] = df.get("w3_mean_dist_km", pd.Series(dtype=float)) - \
                                 df.get("w1_mean_dist_km", pd.Series(dtype=float))
    df["depth_change_km"] = df.get("w1_mean_depth_km", pd.Series(dtype=float)) - \
                            df.get("w3_mean_depth_km", pd.Series(dtype=float))
    return df


def rule_based_type(row):
    qr = row.get("quiescence_ratio")
    acc = row.get("accel_90d")
    n3 = row.get("w3_n_events", 0) or 0

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


def make_pipe(pos_weight=1.0):
    if HAS_XGB:
        mdl = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            verbosity=0,
            random_state=42
        )
    else:
        mdl = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            random_state=42
        )

    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", mdl),
    ])


def load_feature_data(radius=200):
    real = pd.read_csv("output/gcmt_precursor_features.csv", low_memory=False)
    ctrl = pd.read_csv("output/gcmt_control_features.csv", low_memory=False)

    if "radius_km" in real.columns:
        real = real[real["radius_km"] == radius].copy()
    if "radius_km" in ctrl.columns:
        ctrl = ctrl[ctrl["radius_km"] == radius].copy()

    real = add_derived(real)
    ctrl = add_derived(ctrl)

    real["target"] = 1
    ctrl["target"] = 0
    real["pattern_type"] = real.apply(rule_based_type, axis=1)

    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]
    return real, ctrl, avail


def train_two_stage_models(real_train, ctrl_train, avail):
    """
    Her tip icin ayri model egit.
    """
    models = {}

    for tip in ["TIP_A", "TIP_B", "TIP_C"]:
        r_tip = real_train[real_train["pattern_type"] == tip].copy()
        if len(r_tip) < 10:
            models[tip] = None
            continue

        train_df = pd.concat([
            r_tip[avail + ["target"]],
            ctrl_train[avail + ["target"]]
        ], ignore_index=True)

        X = train_df[avail]
        y = train_df["target"]

        pw = (y == 0).sum() / max(y.sum(), 1)
        pipe = make_pipe(pw)
        pipe.fit(X, y)
        models[tip] = (pipe, avail)

    return models


def predict_two_stage(df, models, avail):
    """
    Iki asamali tahmin.
    """
    scores = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        pt = rule_based_type(row_dict)

        component_scores = {}
        for tip, model_info in models.items():
            if model_info is None:
                continue
            pipe, feats = model_info
            try:
                x = pd.DataFrame([{f: row_dict.get(f, np.nan) for f in feats}])
                p = float(pipe.predict_proba(x)[0, 1])
                component_scores[tip] = p
            except Exception:
                pass

        if not component_scores:
            scores.append(0.5)
            continue

        weights = {"TIP_A": 1.0, "TIP_B": 1.2, "TIP_C": 0.5}
        primary = component_scores.get(pt)

        ws = sum(weights.get(t, 1.0) * s for t, s in component_scores.items())
        wt = sum(weights.get(t, 1.0) for t in component_scores)
        ensemble = ws / wt if wt > 0 else 0.5

        final = 0.7 * primary + 0.3 * ensemble if primary is not None else ensemble
        scores.append(final)

    return np.array(scores)


# =========================================================
# TEST 1: TEMPORAL EXTERNAL VALIDATION
# =========================================================

def run_temporal_external():
    print("=" * 70)
    print("TEST 1 — TEMPORAL EXTERNAL VALIDATION")
    print("=" * 70)

    real, ctrl, avail = load_feature_data()

    time_col_r = next((c for c in ["main_datetime_utc", "datetime_utc"]
                       if c in real.columns), None)
    time_col_c = next((c for c in ["ref_datetime_utc", "datetime_utc"]
                       if c in ctrl.columns), None)

    if not time_col_r or not time_col_c:
        print("Zaman sutunu bulunamadi.")
        return []

    real[time_col_r] = pd.to_datetime(real[time_col_r], errors="coerce")
    ctrl[time_col_c] = pd.to_datetime(ctrl[time_col_c], errors="coerce")

    real["_year"] = real[time_col_r].dt.year
    ctrl["_year"] = ctrl[time_col_c].dt.year

    splits = [
        ("1976-2005 -> 2006-2015", (1976, 2005), (2006, 2015)),
        ("1976-2010 -> 2011-2020", (1976, 2010), (2011, 2020)),
        ("1976-2015 -> 2016-2025", (1976, 2015), (2016, 2025)),
    ]

    results = []

    for label, train_range, test_range in splits:
        tr_s, tr_e = train_range
        te_s, te_e = test_range

        real_train = real[(real["_year"] >= tr_s) & (real["_year"] <= tr_e)].copy()
        real_test  = real[(real["_year"] >= te_s) & (real["_year"] <= te_e)].copy()

        ctrl_train = ctrl[(ctrl["_year"] >= tr_s) & (ctrl["_year"] <= tr_e)].copy()
        ctrl_test  = ctrl[(ctrl["_year"] >= te_s) & (ctrl["_year"] <= te_e)].copy()

        if len(real_train) < 30 or len(real_test) < 10:
            print(f"{label}: yetersiz veri, atlandi")
            continue

        print(f"\n{label}")
        print(f"  Train real={len(real_train)}, ctrl={len(ctrl_train)}")
        print(f"  Test  real={len(real_test)}, ctrl={len(ctrl_test)}")

        models = train_two_stage_models(real_train, ctrl_train, avail)

        test_df = pd.concat([real_test, ctrl_test], ignore_index=True)
        y_true = test_df["target"].values
        y_prob = predict_two_stage(test_df, models, avail)

        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        print(f"  AUC={auc:.4f}, AP={ap:.4f}")

        results.append({
            "split": label,
            "auc": round(float(auc), 4),
            "ap": round(float(ap), 4),
            "n_train_real": len(real_train),
            "n_train_ctrl": len(ctrl_train),
            "n_test_real": len(real_test),
            "n_test_ctrl": len(ctrl_test),
        })

    return results


# =========================================================
# TEST 2: LEAVE-ONE-REGION-OUT
# =========================================================

def run_region_external():
    print("\n" + "=" * 70)
    print("TEST 2 — LEAVE-ONE-REGION-OUT")
    print("=" * 70)

    real, ctrl, avail = load_feature_data()

    lat_col = next((c for c in ["main_lat", "lat"] if c in real.columns), None)
    lon_col = next((c for c in ["main_lon", "lon"] if c in real.columns), None)

    if not lat_col or not lon_col:
        print("Koordinat sutunlari bulunamadi.")
        return []

    regions = {
        "Turkiye": {"lat": (35, 43), "lon": (25, 45)},
        "Japonya": {"lat": (30, 46), "lon": (128, 148)},
        "GuneyAmerika": {"lat": (-45, 5), "lon": (-85, -65)},
        "KuzeyAmerikaBatı": {"lat": (30, 50), "lon": (-130, -110)},
        "GuneydoguAsya": {"lat": (-15, 20), "lon": (90, 145)},
    }

    results = []

    for region_name, b in regions.items():
        mask = (
            (real[lat_col] >= b["lat"][0]) & (real[lat_col] <= b["lat"][1]) &
            (real[lon_col] >= b["lon"][0]) & (real[lon_col] <= b["lon"][1])
        )

        real_test = real[mask].copy()
        real_train = real[~mask].copy()

        if len(real_test) < 10 or len(real_train) < 50:
            continue

        print(f"\n{region_name}")
        print(f"  Train real={len(real_train)} | Test real={len(real_test)}")

        models = train_two_stage_models(real_train, ctrl, avail)

        # Test için dengelemek amacıyla kontrol örnekle
        ctrl_sample = ctrl.sample(min(len(ctrl), len(real_test) * 3),
                                  random_state=42).copy()

        test_df = pd.concat([real_test, ctrl_sample], ignore_index=True)
        y_true = test_df["target"].values
        y_prob = predict_two_stage(test_df, models, avail)

        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        print(f"  AUC={auc:.4f}, AP={ap:.4f}")

        results.append({
            "region": region_name,
            "auc": round(float(auc), 4),
            "ap": round(float(ap), 4),
            "n_test_real": len(real_test),
            "n_train_real": len(real_train),
        })

    return results


# =========================================================
# TEST 3: BAĞIMSIZ TARİHSEL BENCHMARK
# =========================================================

def run_independent_benchmark():
    print("\n" + "=" * 70)
    print("TEST 3 — INDEPENDENT HISTORICAL BENCHMARK")
    print("=" * 70)

    if not HAS_ISC:
        print("ISC modulu yok, benchmark atlandi.")
        return []

    # Pozitif örnekler: büyük depremden kısa süre önce
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

    # Negatif örnekler: aynı bölge, ama büyük depremden çok önce / sonra
    negatives = [
        ("Kahramanmaras_2018", 37.22, 37.02, "2018-01-06"),
        ("Izmit_1993", 40.75, 29.86, "1993-07-17"),
        ("Tohoku_2006", 38.30, 142.37, "2006-02-11"),
        ("Maule_2004", -35.85, -72.72, "2004-01-27"),
        ("Landers_1988", 34.20, -116.44, "1988-05-28"),
        ("Ridgecrest_2013", 35.77, -117.60, "2013-06-06"),
        ("Nepal_2009", 28.23, 84.73, "2009-03-25"),
        ("Sumatra_1998", 3.30, 95.98, "1998-11-26"),
    ]

    # Modeli tüm ana veriyle eğit
    real, ctrl, avail = load_feature_data()
    models = train_two_stage_models(real, ctrl, avail)

    rows = []

    print("\nPozitif benchmark örnekleri:")
    for name, lat, lon, ref in positives:
        try:
            feats, meta, err = fetch_and_analyze(
                lat, lon, 300, 2.5, ref_date=ref, use_cache=True
            )
            if feats is None:
                print(f"  {name}: HATA")
                continue

            row = add_derived(pd.DataFrame([feats])).iloc[0]
            score = predict_two_stage(pd.DataFrame([row]), models, avail)[0]

            print(f"  {name:<20} score={score:.4f} "
                  f"n={meta.get('n_total', '?')} "
                  f"Mc={meta.get('mc', '?')}")

            rows.append({
                "case": name, "label": 1, "score": round(float(score), 4),
                "n_total": meta.get("n_total"),
                "mc": meta.get("mc"),
            })
        except Exception as e:
            print(f"  {name}: {e}")

    print("\nNegatif benchmark örnekleri:")
    for name, lat, lon, ref in negatives:
        try:
            feats, meta, err = fetch_and_analyze(
                lat, lon, 300, 2.5, ref_date=ref, use_cache=True
            )
            if feats is None:
                print(f"  {name}: HATA")
                continue

            row = add_derived(pd.DataFrame([feats])).iloc[0]
            score = predict_two_stage(pd.DataFrame([row]), models, avail)[0]

            print(f"  {name:<20} score={score:.4f} "
                  f"n={meta.get('n_total', '?')} "
                  f"Mc={meta.get('mc', '?')}")

            rows.append({
                "case": name, "label": 0, "score": round(float(score), 4),
                "n_total": meta.get("n_total"),
                "mc": meta.get("mc"),
            })
        except Exception as e:
            print(f"  {name}: {e}")

    if len(rows) < 6:
        return []

    bench = pd.DataFrame(rows)
    auc = roc_auc_score(bench["label"], bench["score"])
    ap = average_precision_score(bench["label"], bench["score"])

    print(f"\nIndependent benchmark sonucu:")
    print(f"  AUC={auc:.4f}, AP={ap:.4f}")
    print(f"  Pozitif ort score: {bench[bench['label']==1]['score'].mean():.4f}")
    print(f"  Negatif ort score: {bench[bench['label']==0]['score'].mean():.4f}")

    bench.to_csv(OUTPUT_DIR / "historical_benchmark.csv",
                 index=False, encoding="utf-8-sig")

    return {
        "auc": round(float(auc), 4),
        "ap": round(float(ap), 4),
        "positive_mean": round(float(
            bench[bench["label"] == 1]["score"].mean()), 4),
        "negative_mean": round(float(
            bench[bench["label"] == 0]["score"].mean()), 4),
        "rows": rows,
    }


# =========================================================
# RAPOR
# =========================================================

def run_all():
    print("=" * 70)
    print("SEISMOPATTERN EXTERNAL VALIDATION SUITE")
    print(f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    print(f"ISC modulu: {'OK' if HAS_ISC else 'YOK'}")
    print(f"Model: {'XGBoost' if HAS_XGB else 'GradientBoosting'}")

    temporal = run_temporal_external()
    region = run_region_external()
    benchmark = run_independent_benchmark()

    print("\n" + "=" * 70)
    print("GENEL EXTERNAL VALIDATION OZETI")
    print("=" * 70)

    if temporal:
        t_aucs = [x["auc"] for x in temporal]
        print(f"Temporal average AUC     : {np.mean(t_aucs):.4f}")
        print(f"Temporal min AUC         : {np.min(t_aucs):.4f}")

    if region:
        r_aucs = [x["auc"] for x in region]
        print(f"Region leave-out avg AUC : {np.mean(r_aucs):.4f}")
        print(f"Region leave-out min AUC : {np.min(r_aucs):.4f}")

    if benchmark:
        print(f"Independent benchmark    : AUC={benchmark['auc']:.4f}, "
              f"AP={benchmark['ap']:.4f}")

    # Nihai yorum
    issues = []

    if temporal and np.mean([x["auc"] for x in temporal]) < 0.65:
        issues.append("Temporal generalization zayif")

    if region and np.min([x["auc"] for x in region]) < 0.60:
        issues.append("Bazi bolgelerde genelleme zayif")

    if benchmark and benchmark["auc"] < 0.65:
        issues.append("Bagimsiz benchmark zayif")

    if not issues:
        verdict = "MODEL EXTERNAL VALIDATION'DAN GECTI"
    elif len(issues) == 1:
        verdict = "MODEL BUYUK OLCEKTE GECTI, KISMI ZAYIFLIK VAR"
    else:
        verdict = "MODEL EXTERNAL VALIDATION'DA ZAYIF"

    print(f"\nNIHAI HUKUM: {verdict}")
    if issues:
        for x in issues:
            print(f"  - {x}")

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "temporal": temporal,
        "leave_region_out": region,
        "independent_benchmark": benchmark,
        "verdict": verdict,
        "issues": issues,
    }

    out = OUTPUT_DIR / "external_validation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nRapor kaydedildi: {out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--regions", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    if args.all:
        run_all()
    elif args.temporal:
        run_temporal_external()
    elif args.regions:
        run_region_external()
    elif args.benchmark:
        run_independent_benchmark()
    else:
        print("Kullanim:")
        print("  python scripts/external_validation_suite.py --all")
        print("  python scripts/external_validation_suite.py --temporal")
        print("  python scripts/external_validation_suite.py --regions")
        print("  python scripts/external_validation_suite.py --benchmark")


if __name__ == "__main__":
    main()