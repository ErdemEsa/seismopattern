#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 2 İstatistik Özeti
Öncü feature dosyasının detaylı istatistiksel analizi
"""

import pandas as pd
import numpy as np
import sys

def main():
    f = pd.read_csv("output/gcmt_precursor_features.csv")

    print("=" * 70)
    print("SEISMOPattern - FAZ 2 İSTATİSTİK ÖZETİ")
    print("=" * 70)

    # ──────────────────────────────────────────
    print("\n--- 1) GENEL SAYILAR ---")
    print(f"Satır sayısı       : {len(f):,}")
    print(f"Ana deprem sayısı  : {f['main_event_id'].nunique():,}")
    print(f"Yarıçaplar         : {sorted(f['radius_km'].unique())}")

    # ──────────────────────────────────────────
    print("\n--- 2) QUIESCENCE RATIO - describe ---")
    print(f.groupby("radius_km")["quiescence_ratio"].describe().to_string())

    # ──────────────────────────────────────────
    print("\n--- 3) ACCEL_90D - describe ---")
    print(f.groupby("radius_km")["accel_90d"].describe().to_string())

    # ──────────────────────────────────────────
    print("\n--- 4) FAY TİPİNE GÖRE MEDIAN QUIESCENCE ---")
    tbl = f.groupby(["radius_km", "main_fault_type"])["quiescence_ratio"].agg(
        ["median", "mean", "std", "count"]
    )
    print(tbl.to_string())

    # ──────────────────────────────────────────
    print("\n--- 5) FAY TİPİNE GÖRE MEDIAN ACCEL_90D ---")
    tbl2 = f.groupby(["radius_km", "main_fault_type"])["accel_90d"].agg(
        ["median", "mean", "std", "count"]
    )
    print(tbl2.to_string())

    # ──────────────────────────────────────────
    print("\n--- 6) YETERLİ VERİ OLAN PENCERELER (w3_n_events >= 10) ---")
    f10 = f[f["w3_n_events"] >= 10].copy()
    print(f"Toplam yeterli pencere: {len(f10):,} / {len(f):,} ({len(f10)/len(f)*100:.1f}%)")
    if len(f10) > 0:
        print(f10.groupby("radius_km")[
            ["w1_n_events", "w2_n_events", "w3_n_events",
             "quiescence_ratio", "accel_90d"]
        ].median().to_string())

    # ──────────────────────────────────────────
    print("\n--- 7) SESSİZLİK ORANI (quiescence_ratio < 0.8) ---")
    for r in sorted(f["radius_km"].unique()):
        fr = f[f["radius_km"] == r]
        valid = fr["quiescence_ratio"].notna()
        n_valid = valid.sum()
        if n_valid > 0:
            n_q = (fr.loc[valid, "quiescence_ratio"] < 0.8).sum()
            pct = n_q / n_valid * 100
            print(f"  radius {r:>3d} km: {n_q:>4d} / {n_valid:>4d} = %{pct:.1f}")

    # ──────────────────────────────────────────
    print("\n--- 8) AKTİVASYON ORANI (quiescence_ratio > 1.2) ---")
    for r in sorted(f["radius_km"].unique()):
        fr = f[f["radius_km"] == r]
        valid = fr["quiescence_ratio"].notna()
        n_valid = valid.sum()
        if n_valid > 0:
            n_a = (fr.loc[valid, "quiescence_ratio"] > 1.2).sum()
            pct = n_a / n_valid * 100
            print(f"  radius {r:>3d} km: {n_a:>4d} / {n_valid:>4d} = %{pct:.1f}")

    # ──────────────────────────────────────────
    print("\n--- 9) SON 90 GÜN HIZLANMA (accel_90d > 1.5) ---")
    for r in sorted(f["radius_km"].unique()):
        fr = f[f["radius_km"] == r]
        valid = fr["accel_90d"].notna()
        n_valid = valid.sum()
        if n_valid > 0:
            n_h = (fr.loc[valid, "accel_90d"] > 1.5).sum()
            pct = n_h / n_valid * 100
            print(f"  radius {r:>3d} km: {n_h:>4d} / {n_valid:>4d} = %{pct:.1f}")

    # ──────────────────────────────────────────
    print("\n--- 10) b-DEĞERİ MEVCUTLUK ---")
    for r in sorted(f["radius_km"].unique()):
        fr = f[f["radius_km"] == r]
        for w in ["w1", "w2", "w3"]:
            col = f"{w}_b_value"
            if col in fr.columns:
                n_ok = fr[col].notna().sum()
                print(f"  radius {r:>3d} km | {w}: {n_ok:>4d} / {len(fr):>4d} dolu ({n_ok/len(fr)*100:.1f}%)")

    # ──────────────────────────────────────────
    print("\n--- 11) b-DEĞERİ İSTATİSTİĞİ (mevcut olanlar) ---")
    for r in sorted(f["radius_km"].unique()):
        fr = f[f["radius_km"] == r]
        for w in ["w1", "w2", "w3"]:
            col = f"{w}_b_value"
            vals = fr[col].dropna()
            if len(vals) > 0:
                print(f"  radius {r:>3d} km | {w}: "
                      f"median={vals.median():.3f}  "
                      f"mean={vals.mean():.3f}  "
                      f"std={vals.std():.3f}  "
                      f"min={vals.min():.3f}  "
                      f"max={vals.max():.3f}")

    # ──────────────────────────────────────────
    print("\n--- 12) DERİNLİK SINIFINA GÖRE QUIESCENCE ---")
    f["depth_class"] = pd.cut(
        f["main_depth_km"],
        bins=[0, 35, 70, 300, 700],
        labels=["very_shallow", "shallow", "intermediate", "deep"]
    )
    tbl3 = f.groupby(["radius_km", "depth_class"])["quiescence_ratio"].agg(
        ["median", "mean", "count"]
    )
    print(tbl3.to_string())

    # ──────────────────────────────────────────
    print("\n--- 13) BÜYÜKLÜK SINIFINA GÖRE QUIESCENCE ---")
    f["mw_class"] = pd.cut(
        f["main_mw"],
        bins=[7.0, 7.5, 8.0, 8.5, 10.0],
        labels=["Mw7.0-7.5", "Mw7.5-8.0", "Mw8.0-8.5", "Mw8.5+"],
        right=False
    )
    tbl4 = f.groupby(["radius_km", "mw_class"])["quiescence_ratio"].agg(
        ["median", "mean", "count"]
    )
    print(tbl4.to_string())

    # ──────────────────────────────────────────
    print("\n--- 14) GÖÇ TRENDİ (migration_slope) MEVCUTLUK ---")
    for r in sorted(f["radius_km"].unique()):
        fr = f[f["radius_km"] == r]
        for w in ["w1", "w2", "w3"]:
            col = f"{w}_migration_slope_km_day"
            if col in fr.columns:
                n_ok = fr[col].notna().sum()
                vals = fr[col].dropna()
                if len(vals) > 0:
                    print(f"  radius {r:>3d} km | {w}: dolu={n_ok:>4d}  "
                          f"median={vals.median():.6f}  "
                          f"mean={vals.mean():.6f}")

    # ──────────────────────────────────────────
    print("\n--- 15) EN BÜYÜK 10 DEPREM ÖNCESİ ÖZETİ (300km, w3) ---")
    f300 = f[f["radius_km"] == 300].copy()
    top10 = f300.nlargest(10, "main_mw")
    cols_show = [
        "main_event_id", "main_datetime_utc", "main_mw", "main_region",
        "main_fault_type", "w3_n_events", "quiescence_ratio", "accel_90d",
        "w3_b_value", "w3_migration_slope_km_day"
    ]
    print(top10[[c for c in cols_show if c in top10.columns]].to_string(index=False))

    print("\n" + "=" * 70)
    print("İSTATİSTİK ÖZETİ TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    main()