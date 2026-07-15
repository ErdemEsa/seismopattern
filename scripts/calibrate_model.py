#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Model Kalibrasyon Duzeltmesi
=============================================
Platt Scaling ve Isotonic Regression ile
overconfident modeli kalibre eder.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

OUTPUT_DIR = Path("output/calibrated_models")
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


def load_data():
    real = pd.read_csv("output/gcmt_precursor_features.csv", low_memory=False)
    ctrl = pd.read_csv("output/gcmt_control_features.csv", low_memory=False)

    if "radius_km" in real.columns:
        real = real[real["radius_km"] == 200].copy()
    if "radius_km" in ctrl.columns:
        ctrl = ctrl[ctrl["radius_km"] == 200].copy()

    for df in [real, ctrl]:
        c0 = df.get("count_0_1y", pd.Series(0)).fillna(0)
        c1 = df.get("count_1_2y", pd.Series(0)).fillna(0)
        c2 = df.get("count_2_3y", pd.Series(0)).fillna(0)
        df["count_linear_trend"] = c0 - c2
        df["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)
        df["b_drop_w3_w1"] = df.get("w3_b_value",
                                     pd.Series(dtype=float)) - \
                              df.get("w1_b_value", pd.Series(dtype=float))
        df["spatial_focus_change"] = df.get("w3_mean_dist_km",
                                             pd.Series(dtype=float)) - \
                                      df.get("w1_mean_dist_km",
                                             pd.Series(dtype=float))
        df["depth_change_km"] = df.get("w1_mean_depth_km",
                                        pd.Series(dtype=float)) - \
                                 df.get("w3_mean_depth_km",
                                        pd.Series(dtype=float))

    real["target"] = 1
    ctrl["target"] = 0
    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]

    combined = pd.concat([real[avail + ["target"]],
                           ctrl[avail + ["target"]]], ignore_index=True)
    X = combined[avail]
    y = combined["target"]
    return X, y, avail


def make_base_pipe(pos_weight=1.0):
    if HAS_XGB:
        mdl = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr", verbosity=0, random_state=42
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        mdl = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42
        )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", mdl),
    ])


def ece_score(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece


def evaluate_calibration(y_true, y_prob, label=""):
    """Kalibrasyon metriklerini hesapla ve yazdir."""
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = ece_score(np.array(y_true), np.array(y_prob))
    return {"label": label, "auc": round(auc, 4),
            "brier": round(brier, 4), "ece": round(ece, 4)}


def run_calibration():
    print("=" * 65)
    print("SEISMOPATTERN - KALIBRASYON DUZELTMESI")
    print("=" * 65)

    X, y, avail = load_data()
    print(f"Veri: {len(X)} ornek, real={int(y.sum())}")

    pw = (y == 0).sum() / max(y.sum(), 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Ham model OOS tahminleri
    print("\n[1/4] Ham model OOS tahminleri...")
    y_prob_raw = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        pipe = make_base_pipe(pw)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob_raw[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]

    raw_cal = evaluate_calibration(y.values, y_prob_raw, "Ham Model")
    print(f"  AUC={raw_cal['auc']:.4f}  "
          f"Brier={raw_cal['brier']:.4f}  ECE={raw_cal['ece']:.4f}")

    # 2. Platt Scaling (sigmoid)
    print("\n[2/4] Platt Scaling kalibrasyonu...")
    y_prob_platt = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        base = make_base_pipe(pw)
        cal_pipe = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob_platt[test_idx] = cal_pipe.predict_proba(
            X.iloc[test_idx])[:, 1]

    platt_cal = evaluate_calibration(y.values, y_prob_platt, "Platt Scaling")
    print(f"  AUC={platt_cal['auc']:.4f}  "
          f"Brier={platt_cal['brier']:.4f}  ECE={platt_cal['ece']:.4f}")

    # 3. Isotonic Regression
    print("\n[3/4] Isotonic Regression kalibrasyonu...")
    y_prob_iso = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        base = make_base_pipe(pw)
        cal_pipe = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob_iso[test_idx] = cal_pipe.predict_proba(
            X.iloc[test_idx])[:, 1]

    iso_cal = evaluate_calibration(y.values, y_prob_iso, "Isotonic Reg.")
    print(f"  AUC={iso_cal['auc']:.4f}  "
          f"Brier={iso_cal['brier']:.4f}  ECE={iso_cal['ece']:.4f}")

    # 4. Karsilastirma
    print(f"\n[4/4] Karsilastirma")
    print(f"\n  {'Model':<20} {'AUC':>8} {'Brier':>8} {'ECE':>8} "
          f"{'Brier Degisim':>15} {'ECE Degisim':>12}")
    print(f"  {'-'*75}")

    for res in [raw_cal, platt_cal, iso_cal]:
        b_delta = res["brier"] - raw_cal["brier"]
        e_delta = res["ece"] - raw_cal["ece"]
        print(f"  {res['label']:<20} {res['auc']:>8.4f} "
              f"{res['brier']:>8.4f} {res['ece']:>8.4f} "
              f"{b_delta:>+15.4f} {e_delta:>+12.4f}")

    # En iyi metodu sec
    best = min([raw_cal, platt_cal, iso_cal], key=lambda x: x["ece"])
    print(f"\n  En iyi kalibrasyon: {best['label']} (ECE={best['ece']:.4f})")

    # Reliability diagram karsilastirmasi
    print(f"\n  Reliability Diagram Karsilastirmasi (10 bin):")
    print(f"  {'Bin':>6} {'Ham':>8} {'Platt':>8} {'Iso':>8} {'Gercek':>8}")
    print(f"  {'-'*45}")

    n_bins = 10
    prob_true_raw, prob_pred_raw = calibration_curve(
        y.values, y_prob_raw, n_bins=n_bins, strategy="uniform"
    )
    prob_true_platt, prob_pred_platt = calibration_curve(
        y.values, y_prob_platt, n_bins=n_bins, strategy="uniform"
    )
    prob_true_iso, prob_pred_iso = calibration_curve(
        y.values, y_prob_iso, n_bins=n_bins, strategy="uniform"
    )

    n_pts = min(len(prob_pred_raw), len(prob_pred_platt), len(prob_pred_iso))
    for i in range(n_pts):
        print(f"  {prob_pred_raw[i]:>6.2f} "
              f"{prob_pred_raw[i]:>8.3f} "
              f"{prob_pred_platt[i]:>8.3f} "
              f"{prob_pred_iso[i]:>8.3f} "
              f"{prob_true_raw[i]:>8.3f}")

    # En iyi modeli tam veriyle egit ve kaydet
    print(f"\n  En iyi model ({best['label']}) tam veriyle egitiliyor...")

    best_method = "sigmoid" if "Platt" in best["label"] else \
                  "isotonic" if "Iso" in best["label"] else None

    if best_method:
        base = make_base_pipe(pw)
        final_model = CalibratedClassifierCV(base, method=best_method, cv=5)
        final_model.fit(X, y)
        joblib.dump(final_model, OUTPUT_DIR / "calibrated_model.joblib")
        print(f"  Kaydedildi: {OUTPUT_DIR / 'calibrated_model.joblib'}")
    else:
        base = make_base_pipe(pw)
        base.fit(X, y)
        joblib.dump(base, OUTPUT_DIR / "calibrated_model.joblib")
        print(f"  Ham model kaydedildi (kalibrasyon gerekmedi)")

    # Kalibre modelin feature listesi
    with open(OUTPUT_DIR / "calibrated_features.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    # Ozet rapor
    report = {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "raw": raw_cal,
        "platt": platt_cal,
        "isotonic": iso_cal,
        "best_method": best["label"],
        "best_ece": best["ece"],
        "best_brier": best["brier"],
        "improvement": {
            "brier": round(raw_cal["brier"] - best["brier"], 4),
            "ece": round(raw_cal["ece"] - best["ece"], 4),
        }
    }

    with open(OUTPUT_DIR / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Rapor: {OUTPUT_DIR / 'calibration_report.json'}")

    # Nihai ozet
    print(f"\n{'='*65}")
    print("KALIBRASYON OZETI")
    print(f"{'='*65}")
    print(f"  Ham model:    Brier={raw_cal['brier']:.4f}, ECE={raw_cal['ece']:.4f}")
    print(f"  Platt:        Brier={platt_cal['brier']:.4f}, ECE={platt_cal['ece']:.4f}")
    print(f"  Isotonic:     Brier={iso_cal['brier']:.4f}, ECE={iso_cal['ece']:.4f}")
    print(f"  Secilen:      {best['label']}")
    print(f"  ECE iyilesme: {raw_cal['ece'] - best['ece']:+.4f}")
    print(f"  Brier iyilesme: {raw_cal['brier'] - best['brier']:+.4f}")

    return report


if __name__ == "__main__":
    run_calibration()