#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Istatistiksel Dogrulama Paketi
================================================
1. DeLong Significance Test (model vs baseline AUC farki anlamli mi?)
2. Bootstrap Confidence Interval (AUC icin 95% CI)
3. Bootstrap AUC Dagilimi (1000 tekrar)
4. Calibration Analysis (Brier Score, Reliability Diagram verileri)

Kullanim:
  python scripts/statistical_validation.py --all
"""

import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve,
                              precision_recall_curve)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

OUTPUT_DIR = Path("output/stats_validation")
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
# VERI HAZIRLAMA
# =========================================================

def load_and_prepare():
    real = pd.read_csv("output/gcmt_precursor_features.csv", low_memory=False)
    ctrl = pd.read_csv("output/gcmt_control_features.csv", low_memory=False)

    if "radius_km" in real.columns:
        real = real[real["radius_km"] == 200].copy()
    if "radius_km" in ctrl.columns:
        ctrl = ctrl[ctrl["radius_km"] == 200].copy()

    # Turetilmis feature'lar
    for df in [real, ctrl]:
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

    real["target"] = 1
    ctrl["target"] = 0

    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]

    combined = pd.concat([real[avail + ["target"]],
                           ctrl[avail + ["target"]]], ignore_index=True)

    X = combined[avail]
    y = combined["target"]

    return X, y, avail, real, ctrl


def make_pipe(pos_weight=1.0):
    if HAS_XGB:
        mdl = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr", verbosity=0, random_state=42
        )
    else:
        mdl = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42
        )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", mdl),
    ])


def get_cv_predictions(X, y, n_splits=5):
    """CV ile her ornegin OOS tahminini al."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all = np.zeros(len(y))
    y_prob_all = np.zeros(len(y))

    pw = (y == 0).sum() / max(y.sum(), 1)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = make_pipe(pw)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        y_true_all[test_idx] = y_test.values
        y_prob_all[test_idx] = probs

    return y_true_all, y_prob_all


def get_baseline_predictions(X, y, feature="w3_max_mw"):
    """Tek feature baseline tahmini."""
    if feature not in X.columns:
        return None, None

    vals = X[feature].fillna(X[feature].median()).values

    # Min-max normalize (0-1 arasi)
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        probs = (vals - vmin) / (vmax - vmin)
    else:
        probs = np.full(len(vals), 0.5)

    return y.values, probs


# =========================================================
# 1. DeLong TEST
# =========================================================

def delong_roc_variance(ground_truth, predictions):
    """
    DeLong AUC varyans hesabi.
    Sun & Xu (2014) yontemi.
    """
    order = np.argsort(-predictions)
    label_ordered = ground_truth[order]

    positive_examples = np.sum(label_ordered == 1)
    negative_examples = np.sum(label_ordered == 0)

    if positive_examples == 0 or negative_examples == 0:
        return 0.5, 0.0

    # Placement values
    k = len(label_ordered)
    tx = np.zeros(k)
    ty = np.zeros(k)

    # Siralamaya gore placement
    sorted_preds = predictions[order]

    # Ties icin ortalama rank
    ranks = np.zeros(k)
    i = 0
    while i < k:
        j = i
        while j < k and sorted_preds[j] == sorted_preds[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for l in range(i, j):
            ranks[l] = avg_rank
        i = j

    # Pozitif ve negatif ornekler icin placement
    pos_ranks = ranks[label_ordered == 1]
    neg_ranks = ranks[label_ordered == 0]

    auc = (np.sum(pos_ranks) - positive_examples *
           (positive_examples + 1) / 2) / (positive_examples * negative_examples)

    # Varyans (Hanley & McNeil yaklasimi)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)

    var = (auc * (1 - auc) +
           (positive_examples - 1) * (q1 - auc**2) +
           (negative_examples - 1) * (q2 - auc**2)) / \
          (positive_examples * negative_examples)

    return auc, var


def delong_test(y_true, y_pred_model, y_pred_baseline):
    """
    DeLong testi: iki AUC arasinda anlamli fark var mi?

    H0: AUC_model = AUC_baseline
    H1: AUC_model != AUC_baseline
    """
    auc1, var1 = delong_roc_variance(y_true, y_pred_model)
    auc2, var2 = delong_roc_variance(y_true, y_pred_baseline)

    # Kovaryans tahmini (basitlestirilmis)
    # Tam DeLong kovaryans icin daha karmasik hesap gerekir
    # Burada bagimsiz varsayimi yapiyoruz (muhafazakar)
    z = (auc1 - auc2) / np.sqrt(var1 + var2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "auc_model": round(float(auc1), 4),
        "auc_baseline": round(float(auc2), 4),
        "auc_difference": round(float(auc1 - auc2), 4),
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p_value), 6),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "var_model": round(float(var1), 8),
        "var_baseline": round(float(var2), 8),
    }


def run_delong_tests(y_true, y_prob_model, X, y):
    """Model vs coklu baseline DeLong testleri."""
    print("\n" + "=" * 65)
    print("1. DeLong SIGNIFICANCE TEST")
    print("=" * 65)
    print("  H0: Model AUC = Baseline AUC")
    print("  H1: Model AUC != Baseline AUC")
    print("  Anlamlilik esigi: p < 0.05")

    baselines = {
        "w3_max_mw": "En buyuk Mw (3y)",
        "w1_max_mw": "En buyuk Mw (1y)",
        "count_0_1y": "Olay sayisi (1y)",
        "quiescence_ratio": "Quiescence orani",
        "accel_90d": "90 gun hizlanma",
    }

    results = []

    print(f"\n  {'Baseline':<30} {'BL AUC':>8} {'Model':>8} "
          f"{'Fark':>8} {'z':>8} {'p':>10} {'Anlamli'}")
    print(f"  {'-'*85}")

    for feat, label in baselines.items():
        if feat not in X.columns:
            continue

        _, y_pred_bl = get_baseline_predictions(X, y, feat)
        if y_pred_bl is None:
            continue

        dl = delong_test(y_true, y_prob_model, y_pred_bl)
        sig = "***" if dl["significant_001"] else ("*" if dl["significant_005"] else "")

        print(f"  {label:<30} {dl['auc_baseline']:>8.4f} "
              f"{dl['auc_model']:>8.4f} {dl['auc_difference']:>+8.4f} "
              f"{dl['z_statistic']:>8.3f} {dl['p_value']:>10.6f} {sig}")

        dl["baseline_name"] = label
        dl["baseline_feature"] = feat
        results.append(dl)

    # Top-3 feature baseline
    top3_feats = ["w3_max_mw", "w1_max_mw", "w1_std_mw"]
    top3_avail = [f for f in top3_feats if f in X.columns]
    if len(top3_avail) >= 2:
        pw = (y == 0).sum() / max(y.sum(), 1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_top3 = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X, y):
            pipe = make_pipe(pw)
            pipe.fit(X.iloc[train_idx][top3_avail], y.iloc[train_idx])
            y_pred_top3[test_idx] = pipe.predict_proba(
                X.iloc[test_idx][top3_avail])[:, 1]

        dl_top3 = delong_test(y_true, y_prob_model, y_pred_top3)
        sig = "***" if dl_top3["significant_001"] else (
            "*" if dl_top3["significant_005"] else "")

        print(f"  {'Top-3 Feature Model':<30} {dl_top3['auc_baseline']:>8.4f} "
              f"{dl_top3['auc_model']:>8.4f} {dl_top3['auc_difference']:>+8.4f} "
              f"{dl_top3['z_statistic']:>8.3f} {dl_top3['p_value']:>10.6f} {sig}")

        dl_top3["baseline_name"] = "Top-3 Feature Model"
        results.append(dl_top3)

    n_sig = sum(1 for r in results if r["significant_005"])
    print(f"\n  Anlamli fark (p<0.05): {n_sig}/{len(results)} baseline")

    return results


# =========================================================
# 2. BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================

def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000,
                      ci_level=0.95, random_state=42):
    """
    Bootstrap ile AUC icin guven araligi.
    """
    print("\n" + "=" * 65)
    print("2. BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 65)
    print(f"  Bootstrap tekrar: {n_bootstrap}")
    print(f"  Guven duzeyi: {ci_level*100:.0f}%")

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    boot_aucs = []
    boot_aps = []
    boot_briers = []

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]

        # Her iki sinif da olmali
        if len(np.unique(y_t)) < 2:
            continue

        try:
            boot_aucs.append(roc_auc_score(y_t, y_p))
            boot_aps.append(average_precision_score(y_t, y_p))
            boot_briers.append(brier_score_loss(y_t, y_p))
        except:
            continue

    boot_aucs = np.array(boot_aucs)
    boot_aps = np.array(boot_aps)
    boot_briers = np.array(boot_briers)

    alpha = 1 - ci_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    # AUC CI
    auc_mean = np.mean(boot_aucs)
    auc_std = np.std(boot_aucs)
    auc_lower = np.percentile(boot_aucs, lower_pct)
    auc_upper = np.percentile(boot_aucs, upper_pct)
    auc_median = np.median(boot_aucs)

    # AP CI
    ap_mean = np.mean(boot_aps)
    ap_lower = np.percentile(boot_aps, lower_pct)
    ap_upper = np.percentile(boot_aps, upper_pct)

    # Brier CI
    brier_mean = np.mean(boot_briers)
    brier_lower = np.percentile(boot_briers, lower_pct)
    brier_upper = np.percentile(boot_briers, upper_pct)

    print(f"\n  AUC-ROC:")
    print(f"    Nokta tahmin : {auc_mean:.4f}")
    print(f"    Median       : {auc_median:.4f}")
    print(f"    Std          : {auc_std:.4f}")
    print(f"    95% CI       : [{auc_lower:.4f}, {auc_upper:.4f}]")

    print(f"\n  Average Precision:")
    print(f"    Nokta tahmin : {ap_mean:.4f}")
    print(f"    95% CI       : [{ap_lower:.4f}, {ap_upper:.4f}]")

    print(f"\n  Brier Score (dusuk=iyi):")
    print(f"    Nokta tahmin : {brier_mean:.4f}")
    print(f"    95% CI       : [{brier_lower:.4f}, {brier_upper:.4f}]")

    # AUC dagilim ozeti
    print(f"\n  AUC Dagilim Yuzdelikleri:")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(boot_aucs, pct)
        print(f"    {pct:>3}. yuzdelik: {val:.4f}")

    return {
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
        "auc": {
            "mean": round(float(auc_mean), 4),
            "median": round(float(auc_median), 4),
            "std": round(float(auc_std), 4),
            "ci_lower": round(float(auc_lower), 4),
            "ci_upper": round(float(auc_upper), 4),
        },
        "ap": {
            "mean": round(float(ap_mean), 4),
            "ci_lower": round(float(ap_lower), 4),
            "ci_upper": round(float(ap_upper), 4),
        },
        "brier": {
            "mean": round(float(brier_mean), 4),
            "ci_lower": round(float(brier_lower), 4),
            "ci_upper": round(float(brier_upper), 4),
        },
        "auc_distribution": {
            "p5": round(float(np.percentile(boot_aucs, 5)), 4),
            "p25": round(float(np.percentile(boot_aucs, 25)), 4),
            "p50": round(float(np.percentile(boot_aucs, 50)), 4),
            "p75": round(float(np.percentile(boot_aucs, 75)), 4),
            "p95": round(float(np.percentile(boot_aucs, 95)), 4),
        },
        "boot_aucs": boot_aucs.tolist(),
    }


# =========================================================
# 3. CALIBRATION ANALYSIS
# =========================================================

def calibration_analysis(y_true, y_prob):
    """
    Kalibrasyon analizi:
    - Calibration curve (reliability diagram)
    - Brier score
    - Expected Calibration Error (ECE)
    """
    print("\n" + "=" * 65)
    print("3. CALIBRATION ANALYSIS")
    print("=" * 65)

    # Brier Score
    brier = brier_score_loss(y_true, y_prob)
    print(f"\n  Brier Score: {brier:.4f}")
    print(f"  (0=mukemmel, 0.25=rastgele, dusuk=iyi)")

    if brier < 0.15:
        brier_verdict = "IYI kalibrasyon"
    elif brier < 0.20:
        brier_verdict = "KABUL EDILEBILIR kalibrasyon"
    elif brier < 0.25:
        brier_verdict = "ZAYIF kalibrasyon"
    else:
        brier_verdict = "KOTU kalibrasyon (rastgeleden kotu)"
    print(f"  Degerlendirme: {brier_verdict}")

    # Calibration Curve
    n_bins = 10
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                              strategy="uniform")

    print(f"\n  Reliability Diagram ({n_bins} bin):")
    print(f"  {'Tahmin Ort':>12} {'Gercek Ort':>12} {'Fark':>8} {'n':>6}")
    print(f"  {'-'*42}")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = []
    bin_details = []

    for i in range(len(prob_true)):
        pred_mean = prob_pred[i]
        true_mean = prob_true[i]
        diff = pred_mean - true_mean

        # Bu bin'deki ornek sayisi
        if i < len(bin_edges) - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            n_in_bin = mask.sum()
        else:
            n_in_bin = 0

        bin_counts.append(int(n_in_bin))

        print(f"  {pred_mean:>12.4f} {true_mean:>12.4f} "
              f"{diff:>+8.4f} {n_in_bin:>6}")

        bin_details.append({
            "pred_mean": round(float(pred_mean), 4),
            "true_mean": round(float(true_mean), 4),
            "diff": round(float(diff), 4),
            "n": int(n_in_bin),
        })

    # Expected Calibration Error (ECE)
    total = len(y_true)
    ece = 0
    for i in range(len(prob_true)):
        n_bin = bin_counts[i] if i < len(bin_counts) else 0
        if n_bin > 0:
            ece += (n_bin / total) * abs(prob_pred[i] - prob_true[i])

    print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  (0=mukemmel kalibre, dusuk=iyi)")

    if ece < 0.05:
        ece_verdict = "MUKEMMEL kalibrasyon"
    elif ece < 0.10:
        ece_verdict = "IYI kalibrasyon"
    elif ece < 0.15:
        ece_verdict = "KABUL EDILEBILIR"
    else:
        ece_verdict = "KOTU kalibrasyon"
    print(f"  Degerlendirme: {ece_verdict}")

    # Over/under-confidence analizi
    overconfident = sum(1 for d in bin_details if d["diff"] > 0.05)
    underconfident = sum(1 for d in bin_details if d["diff"] < -0.05)

    if overconfident > underconfident:
        conf_verdict = "Model ASIRI GUVENLI (overconfident)"
    elif underconfident > overconfident:
        conf_verdict = "Model AZ GUVENLI (underconfident)"
    else:
        conf_verdict = "Model DENGELI"

    print(f"\n  Guven analizi:")
    print(f"    Asiri guvenli bin: {overconfident}/{len(bin_details)}")
    print(f"    Az guvenli bin:    {underconfident}/{len(bin_details)}")
    print(f"    Degerlendirme: {conf_verdict}")

    # Skor dagilimi
    print(f"\n  Tahmin Dagilimi:")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(y_prob, pct)
        print(f"    {pct:>3}. yuzdelik: {val:.4f}")

    print(f"\n  Sinif bazli tahmin ortalamasi:")
    print(f"    Gercek pozitif (real):    {y_prob[y_true == 1].mean():.4f}")
    print(f"    Gercek negatif (ctrl):    {y_prob[y_true == 0].mean():.4f}")
    print(f"    Fark:                     {y_prob[y_true == 1].mean() - y_prob[y_true == 0].mean():.4f}")

    return {
        "brier_score": round(float(brier), 4),
        "brier_verdict": brier_verdict,
        "ece": round(float(ece), 4),
        "ece_verdict": ece_verdict,
        "confidence_verdict": conf_verdict,
        "calibration_bins": bin_details,
        "positive_mean_prob": round(float(y_prob[y_true == 1].mean()), 4),
        "negative_mean_prob": round(float(y_prob[y_true == 0].mean()), 4),
        "prob_separation": round(float(
            y_prob[y_true == 1].mean() - y_prob[y_true == 0].mean()
        ), 4),
    }


# =========================================================
# TAM RAPOR
# =========================================================

def run_all():
    print("=" * 65)
    print("SEISMOPATTERN ISTATISTIKSEL DOGRULAMA")
    print(f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)

    # Veri yukle
    X, y, avail, real, ctrl = load_and_prepare()
    print(f"\nVeri: {len(X)} ornek, {len(avail)} feature")
    print(f"  real: {int(y.sum())}, ctrl: {int((y == 0).sum())}")

    # OOS tahminleri al
    print("\nOOS tahminleri hesaplaniyor (5-Fold CV)...")
    y_true, y_prob = get_cv_predictions(X, y)

    point_auc = roc_auc_score(y_true, y_prob)
    point_ap = average_precision_score(y_true, y_prob)
    print(f"  Nokta AUC: {point_auc:.4f}")
    print(f"  Nokta AP:  {point_ap:.4f}")

    all_results = {}

    # 1. DeLong
    all_results["delong"] = run_delong_tests(y_true, y_prob, X, y)

    # 2. Bootstrap CI
    boot_results = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000)
    # boot_aucs listesini kaydetme (cok buyuk)
    boot_save = {k: v for k, v in boot_results.items() if k != "boot_aucs"}
    all_results["bootstrap"] = boot_save

    # 3. Calibration
    all_results["calibration"] = calibration_analysis(y_true, y_prob)

    # =========================================================
    # GENEL OZET
    # =========================================================

    print(f"\n{'=' * 65}")
    print("GENEL ISTATISTIKSEL OZET")
    print(f"{'=' * 65}")

    auc_ci = boot_results["auc"]
    cal = all_results["calibration"]
    delong_results = all_results["delong"]

    n_sig_delong = sum(1 for r in delong_results if r.get("significant_005"))

    print(f"""
  MODEL PERFORMANSI:
    AUC-ROC        : {auc_ci['mean']:.4f}
    95% CI         : [{auc_ci['ci_lower']:.4f}, {auc_ci['ci_upper']:.4f}]
    Std            : {auc_ci['std']:.4f}

  KALIBRASYON:
    Brier Score    : {cal['brier_score']:.4f} ({cal['brier_verdict']})
    ECE            : {cal['ece']:.4f} ({cal['ece_verdict']})
    Guven durumu   : {cal['confidence_verdict']}
    Prob ayirimi   : {cal['prob_separation']:.4f}

  DeLong TESTLERI:
    Anlamli fark   : {n_sig_delong}/{len(delong_results)} baseline
    Model baseline'dan anlamli olarak ustun mu?
""")

    if n_sig_delong > len(delong_results) / 2:
        delong_verdict = "EVET — Cogu baseline'dan istatistiksel olarak ustun"
    elif n_sig_delong > 0:
        delong_verdict = "KISMI — Bazi baseline'lardan ustun"
    else:
        delong_verdict = "HAYIR — Hicbir baseline'dan anlamli fark yok"

    print(f"    {delong_verdict}")

    # Nihai hukum
    print(f"\n  {'=' * 50}")
    print(f"  NIHAI ISTATISTIKSEL HUKUM:")

    issues = []
    if auc_ci["ci_lower"] < 0.65:
        issues.append("AUC alt siniri dusuk")
    if cal["brier_score"] > 0.25:
        issues.append("Brier score rastgeleden kotu")
    if cal["ece"] > 0.15:
        issues.append("Kalibrasyon zayif")
    if n_sig_delong == 0:
        issues.append("Baseline'lardan anlamli fark yok")

    if not issues:
        final = "MODEL ISTATISTIKSEL OLARAK GUVENILIR"
    elif len(issues) <= 1:
        final = "MODEL KABUL EDILEBILIR (kucuk sorunlar var)"
    elif len(issues) <= 2:
        final = "MODEL DIKKATLE KULLANILMALI"
    else:
        final = "MODEL ISTATISTIKSEL OLARAK ZAYIF"

    print(f"  {final}")
    if issues:
        for issue in issues:
            print(f"    - {issue}")
    print(f"  {'=' * 50}")

    # Kaydet
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "point_auc": round(float(point_auc), 4),
        "point_ap": round(float(point_ap), 4),
        "auc_95ci": [auc_ci["ci_lower"], auc_ci["ci_upper"]],
        "tests": all_results,
        "final_verdict": final,
        "issues": issues,
    }

    report_path = OUTPUT_DIR / "statistical_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Rapor: {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--delong", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    args = ap.parse_args()

    if args.all:
        run_all()
    elif args.delong or args.bootstrap or args.calibration:
        X, y, avail, real, ctrl = load_and_prepare()
        y_true, y_prob = get_cv_predictions(X, y)

        if args.delong:
            run_delong_tests(y_true, y_prob, X, y)
        if args.bootstrap:
            bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000)
        if args.calibration:
            calibration_analysis(y_true, y_prob)
    else:
        print("python scripts/statistical_validation.py --all")


if __name__ == "__main__":
    main()