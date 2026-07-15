#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 2 Karşılaştırma
=====================================
Gerçek öncü pencereleri ile kontrol pencerelerini karşılaştırır.
İstatistiksel fark testi uygular.
"""

import pandas as pd
import numpy as np
from scipy import stats


def compare_groups(real_df, ctrl_df, metric, radius):
    """İki grup arasında istatistiksel karşılaştırma"""
    r_vals = real_df[real_df["radius_km"] == radius][metric].dropna()
    c_vals = ctrl_df[ctrl_df["radius_km"] == radius][metric].dropna()

    if len(r_vals) < 10 or len(c_vals) < 10:
        return None

    # Mann-Whitney U testi (non-parametric)
    try:
        u_stat, u_pval = stats.mannwhitneyu(r_vals, c_vals, alternative="two-sided")
    except:
        u_stat, u_pval = np.nan, np.nan

    # Kolmogorov-Smirnov testi
    try:
        ks_stat, ks_pval = stats.ks_2samp(r_vals, c_vals)
    except:
        ks_stat, ks_pval = np.nan, np.nan

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((r_vals.std()**2 + c_vals.std()**2) / 2)
    cohens_d = (r_vals.mean() - c_vals.mean()) / pooled_std if pooled_std > 0 else np.nan

    return {
        "metric": metric,
        "radius_km": radius,
        "real_n": len(r_vals),
        "real_median": r_vals.median(),
        "real_mean": r_vals.mean(),
        "real_std": r_vals.std(),
        "ctrl_n": len(c_vals),
        "ctrl_median": c_vals.median(),
        "ctrl_mean": c_vals.mean(),
        "ctrl_std": c_vals.std(),
        "mann_whitney_U": u_stat,
        "mann_whitney_p": u_pval,
        "ks_stat": ks_stat,
        "ks_p": ks_pval,
        "cohens_d": cohens_d,
        "significant_005": u_pval < 0.05 if pd.notna(u_pval) else False,
        "significant_001": u_pval < 0.01 if pd.notna(u_pval) else False,
    }


def main():
    print("=" * 70)
    print("SEISMOPattern - GERÇEK vs KONTROL KARŞILAŞTIRMASI")
    print("=" * 70)

    # Gerçek öncü featurelar
    real = pd.read_csv("output/gcmt_precursor_features.csv")
    real["label"] = "REAL"

    # Kontrol featurelar
    ctrl = pd.read_csv("output/gcmt_control_features.csv")

    print(f"\nGerçek öncü pencere satırı : {len(real):,}")
    print(f"Kontrol pencere satırı    : {len(ctrl):,}")

    # Karşılaştırılacak metrikler
    metrics = [
        "quiescence_ratio",
        "accel_90d",
        "count_0_1y",
        "monthly_slope_36m",
        "w1_n_events",
        "w1_max_mw",
        "w1_mean_mw",
        "w1_b_value",
        "w1_cum_moment_nm",
        "w1_mean_depth_km",
        "w1_std_depth_km",
        "w1_mean_dist_km",
        "w1_migration_slope_km_day",
        "w3_n_events",
        "w3_b_value",
        "w3_mean_mw",
    ]

    results = []
    for radius in [100, 200, 300]:
        for metric in metrics:
            if metric in real.columns and metric in ctrl.columns:
                result = compare_groups(real, ctrl, metric, radius)
                if result:
                    results.append(result)

    res_df = pd.DataFrame(results)

    if len(res_df) == 0:
        print("\nKarşılaştırma yapılamadı - yeterli veri yok.")
        return

    # Sonuçları yazdır
    print("\n" + "=" * 70)
    print("KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 70)

    for radius in [100, 200, 300]:
        r_df = res_df[res_df["radius_km"] == radius].copy()
        if len(r_df) == 0:
            continue

        print(f"\n{'─'*60}")
        print(f"YARIÇAP: {radius} km")
        print(f"{'─'*60}")

        for _, row in r_df.iterrows():
            sig = "***" if row["significant_001"] else ("*  " if row["significant_005"] else "   ")
            direction = ""
            if pd.notna(row["real_median"]) and pd.notna(row["ctrl_median"]):
                if row["real_median"] > row["ctrl_median"]:
                    direction = "REAL > CTRL"
                elif row["real_median"] < row["ctrl_median"]:
                    direction = "REAL < CTRL"
                else:
                    direction = "REAL = CTRL"

            print(f"  {sig} {row['metric']:<35s} "
                  f"real_med={row['real_median']:>10.4f}  "
                  f"ctrl_med={row['ctrl_median']:>10.4f}  "
                  f"p={row['mann_whitney_p']:.4f}  "
                  f"d={row['cohens_d']:>+.3f}  "
                  f"{direction}")

    # Anlamlı sonuçlar özeti
    sig_results = res_df[res_df["significant_005"]].copy()
    print(f"\n{'='*70}")
    print(f"ANLAMLI SONUÇLAR (p < 0.05): {len(sig_results)} adet")
    print(f"{'='*70}")

    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            direction = "↑" if row["real_median"] > row["ctrl_median"] else "↓"
            print(f"  {direction} {row['metric']} @ {row['radius_km']}km  "
                  f"(p={row['mann_whitney_p']:.4f}, d={row['cohens_d']:+.3f})")

    # CSV kaydet
    res_df.to_csv("output/gcmt_real_vs_control.csv", index=False, encoding="utf-8-sig")
    print(f"\nDetaylı sonuçlar: output/gcmt_real_vs_control.csv")

    # Çok anlamlı olanlar (p < 0.01)
    very_sig = res_df[res_df["significant_001"]].copy()
    print(f"\nÇOK ANLAMLI SONUÇLAR (p < 0.01): {len(very_sig)} adet")
    if len(very_sig) > 0:
        for _, row in very_sig.iterrows():
            direction = "↑" if row["real_median"] > row["ctrl_median"] else "↓"
            print(f"  {direction} {row['metric']} @ {row['radius_km']}km  "
                  f"(p={row['mann_whitney_p']:.6f}, d={row['cohens_d']:+.3f})")


if __name__ == "__main__":
    main()