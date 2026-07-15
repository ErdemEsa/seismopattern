#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Her büyük deprem için tek bir öncü şablon skoru hesapla.

Skor metodolojisi:
  İstatistiksel olarak anlamlı bulunan 6 metriği normalize edip
  ağırlıklı toplamını alıyoruz.

  Kullanılan metrikler (200km yarıçap):
    1. w1_max_mw          (ağırlık: 0.25) → en güçlü etki
    2. w3_b_value_inv     (ağırlık: 0.25) → b düşüklüğü (ters)
    3. w1_n_events        (ağırlık: 0.20)
    4. w1_cum_moment_nm   (ağırlık: 0.15)
    5. w1_mean_dist_km_inv(ağırlık: 0.10) → uzaklık (ters)
    6. accel_90d          (ağırlık: 0.05)
"""
import pandas as pd
import numpy as np

def robust_normalize(series):
    """Medyan ve IQR ile normalize (outlier'a dayanıklı)"""
    med = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)
    if iqr == 0:
        iqr = series.std()
    if iqr == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - med) / iqr

def compute_precursor_score(df, radius=200):
    df = df[df["radius_km"] == radius].copy()
    df = df.reset_index(drop=True)

    # Ters metrikler (düşük = iyi işaret)
    if "w3_b_value" in df.columns:
        df["b_value_inv"] = -df["w3_b_value"]
    else:
        df["b_value_inv"] = np.nan

    if "w1_mean_dist_km" in df.columns:
        df["dist_inv"] = -df["w1_mean_dist_km"]
    else:
        df["dist_inv"] = np.nan

    # Normalize et
    metrics_weights = {
        "w1_max_mw": 0.25,
        "b_value_inv": 0.25,
        "w1_n_events": 0.20,
        "w1_cum_moment_nm": 0.15,
        "dist_inv": 0.10,
        "accel_90d": 0.05,
    }

    score = pd.Series(np.zeros(len(df)), index=df.index)
    weight_used = pd.Series(np.zeros(len(df)), index=df.index)

    for metric, weight in metrics_weights.items():
        if metric not in df.columns:
            continue
        col = df[metric].copy()
        valid_mask = col.notna()
        if valid_mask.sum() < 5:
            continue
        normalized = robust_normalize(col)
        score += normalized.fillna(0) * weight
        weight_used += valid_mask.astype(float) * weight

    # Ağırlık kullanımına göre düzelt
    weight_used = weight_used.replace(0, np.nan)
    score = score / weight_used

    df["precursor_score"] = score
    df["weight_coverage"] = weight_used

    return df

def main():
    real = pd.read_csv("output/gcmt_precursor_features.csv")
    ctrl = pd.read_csv("output/gcmt_control_features.csv")

    # Gerçek öncü skorlar
    real_scored = compute_precursor_score(real, radius=200)
    real_scored["label"] = "REAL"

    # Kontrol skorlar
    if "main_fault_type" in ctrl.columns:
        ctrl = ctrl.rename(columns={"main_fault_type": "parent_fault_type"})
    ctrl_work = ctrl.copy()
    ctrl_work["main_fault_type"] = ctrl_work.get("parent_fault_type", "UNKNOWN")
    ctrl_work["main_mw"] = ctrl_work.get("parent_mw", np.nan)
    ctrl_work["main_event_id"] = ctrl_work.get("parent_event_id", "CTRL")
    ctrl_work["main_datetime_utc"] = ctrl_work.get("ref_datetime_utc", pd.NaT)
    ctrl_work["main_region"] = "CONTROL"

    ctrl_scored = compute_precursor_score(ctrl_work, radius=200)
    ctrl_scored["label"] = "CONTROL"

    print("=" * 70)
    print("ÖNCÜ ŞABLON SKORU ANALİZİ")
    print("=" * 70)

    print(f"\nGerçek öncü - skor özeti:")
    print(real_scored["precursor_score"].describe())

    print(f"\nKontrol - skor özeti:")
    print(ctrl_scored["precursor_score"].describe())

    # İki grup karşılaştırması
    from scipy import stats
    r_vals = real_scored["precursor_score"].dropna()
    c_vals = ctrl_scored["precursor_score"].dropna()
    _, p = stats.mannwhitneyu(r_vals, c_vals, alternative="two-sided")
    pooled = np.sqrt((r_vals.std()**2 + c_vals.std()**2) / 2)
    d = (r_vals.mean() - c_vals.mean()) / pooled if pooled > 0 else np.nan

    print(f"\nKarşılaştırma:")
    print(f"  Gerçek median skor : {r_vals.median():.4f}")
    print(f"  Kontrol median skor: {c_vals.median():.4f}")
    print(f"  Mann-Whitney p     : {p:.6f}")
    print(f"  Cohen's d          : {d:+.4f}")

    # Skor eşiği analizi
    print(f"\nSkor eşiği - Doğruluk analizi:")
    thresholds = np.arange(-2.0, 3.0, 0.25)
    best_threshold = None
    best_accuracy = 0
    results_thresh = []

    for th in thresholds:
        real_positive = (real_scored["precursor_score"] >= th).sum()
        real_negative = (real_scored["precursor_score"] < th).sum()
        ctrl_positive = (ctrl_scored["precursor_score"] >= th).sum()
        ctrl_negative = (ctrl_scored["precursor_score"] < th).sum()

        tp = real_positive
        fn = real_negative
        fp = ctrl_positive
        tn = ctrl_negative

        total = tp + fn + fp + tn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results_thresh.append({
            "threshold": th,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn
        })

        if f1 > best_accuracy:
            best_accuracy = f1
            best_threshold = th

    thresh_df = pd.DataFrame(results_thresh)
    thresh_df.to_csv("output/gcmt_score_thresholds.csv", index=False, encoding="utf-8-sig")

    print(f"\n  En iyi F1 eşiği: {best_threshold:.2f}")
    best_row = thresh_df[thresh_df["threshold"] == best_threshold].iloc[0]
    print(f"  Accuracy : {best_row['accuracy']:.3f}")
    print(f"  Precision: {best_row['precision']:.3f}")
    print(f"  Recall   : {best_row['recall']:.3f}")
    print(f"  F1       : {best_row['f1']:.3f}")
    print(f"  TP={best_row['tp']:.0f}  FN={best_row['fn']:.0f}  "
          f"FP={best_row['fp']:.0f}  TN={best_row['tn']:.0f}")

    # Fay tipine göre skor
    print(f"\nFay tipine göre ortalama skor (REAL):")
    if "main_fault_type" in real_scored.columns:
        print(real_scored.groupby("main_fault_type")["precursor_score"].agg(
            ["median", "mean", "std", "count"]
        ).to_string())

    # En yüksek ve en düşük skorlu büyük depremler
    print(f"\nEn yüksek öncü skorlu 10 büyük deprem:")
    top = real_scored.nlargest(10, "precursor_score")
    show_cols = ["main_event_id", "main_datetime_utc", "main_mw",
                 "main_region", "main_fault_type", "precursor_score",
                 "w1_max_mw", "w3_b_value", "w1_n_events"]
    print(top[[c for c in show_cols if c in top.columns]].to_string(index=False))

    print(f"\nEn düşük öncü skorlu 10 büyük deprem:")
    bot = real_scored.nsmallest(10, "precursor_score")
    print(bot[[c for c in show_cols if c in bot.columns]].to_string(index=False))

    # Kaydet
    real_scored.to_csv("output/gcmt_real_scored.csv", index=False, encoding="utf-8-sig")
    ctrl_scored.to_csv("output/gcmt_ctrl_scored.csv", index=False, encoding="utf-8-sig")

    print(f"\nKaydedildi:")
    print(f"  output/gcmt_real_scored.csv")
    print(f"  output/gcmt_ctrl_scored.csv")
    print(f"  output/gcmt_score_thresholds.csv")

if __name__ == "__main__":
    main()