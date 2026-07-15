#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fay tipine göre ayrı karşılaştırma
"""
import pandas as pd
import numpy as np
from scipy import stats

def compare_fault_type(real, ctrl, metric, radius, fault_type):
    r = real[(real["radius_km"] == radius) & (real["main_fault_type"] == fault_type)][metric].dropna()
    c = ctrl[(ctrl["radius_km"] == radius) & (ctrl["parent_fault_type"] == fault_type)][metric].dropna()
    if len(r) < 5 or len(c) < 5:
        return None
    try:
        _, p = stats.mannwhitneyu(r, c, alternative="two-sided")
    except:
        p = np.nan
    pooled = np.sqrt((r.std()**2 + c.std()**2) / 2)
    d = (r.mean() - c.mean()) / pooled if pooled > 0 else np.nan
    return {
        "fault_type": fault_type,
        "radius_km": radius,
        "metric": metric,
        "real_n": len(r),
        "real_median": r.median(),
        "ctrl_n": len(c),
        "ctrl_median": c.median(),
        "p_value": p,
        "cohens_d": d,
        "significant": p < 0.05 if pd.notna(p) else False,
        "direction": "REAL>CTRL" if r.median() > c.median() else "REAL<CTRL"
    }

def main():
    real = pd.read_csv("output/gcmt_precursor_features.csv")
    ctrl = pd.read_csv("output/gcmt_control_features.csv")

    metrics = [
        "quiescence_ratio", "accel_90d", "w1_n_events",
        "w1_max_mw", "w1_b_value", "w1_cum_moment_nm",
        "w1_mean_depth_km", "w1_mean_dist_km",
        "w3_n_events", "w3_b_value", "monthly_slope_36m"
    ]

    fault_types = ["REVERSE", "STRIKE_SLIP", "NORMAL"]
    radii = [100, 200, 300]

    rows = []
    for ft in fault_types:
        for r in radii:
            for m in metrics:
                if m in real.columns and m in ctrl.columns:
                    res = compare_fault_type(real, ctrl, m, r, ft)
                    if res:
                        rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv("output/gcmt_by_faulttype.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("FAY TİPİNE GÖRE KARŞILAŞTIRMA")
    print("=" * 80)

    for ft in fault_types:
        print(f"\n{'─'*70}")
        print(f"FAY TİPİ: {ft}")
        print(f"{'─'*70}")
        fdf = df[(df["fault_type"] == ft) & (df["radius_km"] == 200)].copy()
        for _, row in fdf.iterrows():
            sig = "***" if row["p_value"] < 0.01 else ("*  " if row["significant"] else "   ")
            print(f"  {sig} {row['metric']:<35s} "
                  f"real={row['real_median']:>10.4f}  "
                  f"ctrl={row['ctrl_median']:>10.4f}  "
                  f"p={row['p_value']:.4f}  "
                  f"d={row['cohens_d']:>+.3f}  "
                  f"{row['direction']}")

    # En güçlü fay bazlı sinyaller
    print(f"\n{'='*80}")
    print("EN GÜÇLÜ FAY BAZLI SİNYALLER (p<0.01, |d|>0.3)")
    print(f"{'='*80}")
    strong = df[
        (df["p_value"] < 0.01) & (df["cohens_d"].abs() > 0.3)
    ].sort_values("cohens_d", key=abs, ascending=False)
    for _, row in strong.iterrows():
        direction = "↑" if row["direction"] == "REAL>CTRL" else "↓"
        print(f"  {direction} [{row['fault_type']}] {row['metric']} "
              f"@ {row['radius_km']}km  "
              f"(p={row['p_value']:.4f}, d={row['cohens_d']:+.3f})")

if __name__ == "__main__":
    main()