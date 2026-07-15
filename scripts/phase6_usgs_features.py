#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 6: USGS Mikro-Sismik Feature Çıkarımı
===========================================================
USGS bölgesel verilerinden (Mw 2.5+) detaylı feature'lar çıkar,
GCMT feature'larıyla birleştir, hibrit model eğit.

Aşamalar:
1. Her USGS olayı için completeness magnitude (Mc) tahmini
2. Mc üzerindeki verilerle b-değeri zaman serisi
3. Aktivite hızlanma / sessizlik metrikleri
4. Mekânsal göç ve odaklanma metrikleri
5. GCMT + USGS hibrit feature tablosu
6. Hibrit model eğitimi ve değerlendirmesi
"""

import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      cross_validate)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


EARTH_RADIUS_KM = 6371.0

PRIORITY_EVENTS = [
    ("tohoku_2011",        "Tohoku Japonya 2011",     "2011-03-11", 38.297, 142.373, 9.1, "REVERSE"),
    ("sumatra_2004",       "Sumatra 2004",             "2004-12-26",  3.295,  95.982, 9.1, "REVERSE"),
    ("chile_2010",         "Maule Şili 2010",          "2010-02-27",-35.846, -72.719, 8.8, "REVERSE"),
    ("chile_2015",         "Illapel Şili 2015",        "2015-09-16",-31.573, -71.674, 8.3, "REVERSE"),
    ("sumatra_2012",       "N.Sumatra 2012",           "2012-04-11",  2.327,  93.063, 8.6, "STRIKE_SLIP"),
    ("kahramanmaras_2023", "Kahramanmaraş 2023",       "2023-02-06", 37.220,  37.019, 7.8, "STRIKE_SLIP"),
    ("izmit_1999",         "İzmit 1999",               "1999-08-17", 40.748,  29.864, 7.6, "STRIKE_SLIP"),
    ("landers_1992",       "Landers ABD 1992",         "1992-06-28", 34.200,-116.437, 7.3, "STRIKE_SLIP"),
    ("ridgecrest_2019",    "Ridgecrest ABD 2019",      "2019-07-06", 35.770,-117.599, 7.1, "STRIKE_SLIP"),
    ("nias_2005",          "Nias Endonezya 2005",      "2005-03-28",  1.665,  97.004, 8.6, "REVERSE"),
    ("kuril_2006",         "Kuril 2006",               "2006-11-15", 46.607, 153.266, 8.3, "REVERSE"),
    ("samoa_2009",         "Samoa 2009",               "2009-09-29",-15.489,-172.095, 8.1, "NORMAL"),
    ("nepal_2015",         "Nepal 2015",               "2015-04-25", 28.231,  84.731, 7.8, "REVERSE"),
    ("new_zealand_2016",   "Kaikoura YZ 2016",         "2016-11-13",-42.757, 173.054, 7.8, "STRIKE_SLIP"),
    ("mexico_2017",        "Puebla Meksika 2017",      "2017-09-19", 18.584, -98.399, 7.1, "NORMAL"),
    ("palu_2018",          "Palu Endonezya 2018",      "2018-09-28", -0.256, 119.846, 7.5, "STRIKE_SLIP"),
    ("bam_2003",           "Bam İran 2003",            "2003-12-26", 29.010,  58.310, 6.6, "STRIKE_SLIP"),
    ("haiti_2010",         "Haiti 2010",               "2010-01-12", 18.443, -72.571, 7.0, "STRIKE_SLIP"),
]


# ═══════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians,
                                   [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (np.sin(dlat/2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def estimate_mc(mags, dm=0.1):
    """Maximum Curvature yöntemi ile Mc"""
    mags = np.array(mags)
    mags = mags[np.isfinite(mags)]
    if len(mags) < 10:
        return np.nanmin(mags) if len(mags) > 0 else 3.0
    bins = np.arange(
        np.floor(mags.min() * 10) / 10,
        np.ceil(mags.max() * 10) / 10 + dm, dm
    )
    counts, edges = np.histogram(mags, bins=bins)
    if len(counts) == 0:
        return mags.min()
    mc_idx = np.argmax(counts)
    return round(edges[mc_idx] + dm / 2, 1)


def b_value_mle(mags, mc, dm=0.05):
    """MLE ile b-değeri"""
    mags = np.array(mags)
    above = mags[np.isfinite(mags) & (mags >= mc)]
    n = len(above)
    if n < 15:
        return np.nan, np.nan
    mean_m = np.mean(above)
    denom = mean_m - (mc - dm / 2.0)
    if denom <= 0:
        return np.nan, np.nan
    b = np.log10(np.e) / denom
    b_err = 2.3 * b**2 * np.std(above, ddof=1) / np.sqrt(n * (n - 1))
    return round(b, 4), round(b_err, 4)


def z_score_seismicity(n1, n2, t1_days, t2_days):
    """
    Seismicity Rate Change Z-score
    n1: son dönem olay sayısı, t1_days: süre
    n2: önceki dönem olay sayısı, t2_days: süre
    """
    if t1_days <= 0 or t2_days <= 0:
        return np.nan
    r1 = n1 / t1_days
    r2 = n2 / t2_days
    if r1 + r2 == 0:
        return 0.0
    # Pooled rate
    r_pool = (n1 + n2) / (t1_days + t2_days)
    se = np.sqrt(r_pool * (1/t1_days + 1/t2_days))
    if se == 0:
        return np.nan
    return (r1 - r2) / se


# ═══════════════════════════════════════════════════════════
# 1. USGS FEATURE ÇIKARIMI
# ═══════════════════════════════════════════════════════════

def extract_usgs_features(event_id, name, date_str, lat, lon,
                           mw, fault_type, data_dir="output/usgs_regional"):
    """Tek bir olay için USGS feature'ları çıkar"""
    filepath = Path(data_dir) / f"{event_id}_precursor.csv"
    if not filepath.exists():
        return None

    df = pd.read_csv(filepath)
    if df.empty or len(df) < 5:
        return None

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.dropna(subset=["datetime_utc", "magnitude"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    main_time = pd.Timestamp(date_str)
    df["days_before"] = (main_time - df["datetime_utc"]).dt.total_seconds() / 86400.0
    df = df[df["days_before"] >= 0].copy()

    n_total = len(df)
    if n_total < 5:
        return None

    # Completeness magnitude
    mc = estimate_mc(df["magnitude"].values)
    df_mc = df[df["magnitude"] >= mc].copy()
    n_mc = len(df_mc)

    feats = {
        "event_id": event_id,
        "name": name,
        "date": date_str,
        "mw": mw,
        "fault_type": fault_type,
        "lat": lat,
        "lon": lon,
        "usgs_n_total": n_total,
        "usgs_mc": mc,
        "usgs_n_above_mc": n_mc,
    }

    if n_mc < 5:
        return feats

    # ── b-değeri (3 pencere) ──────────────────────────────
    for wname, d_min, d_max in [("w1", 0, 365),
                                  ("w2", 365, 730),
                                  ("w3", 730, 1095)]:
        sub = df_mc[
            (df_mc["days_before"] >= d_min) &
            (df_mc["days_before"] < d_max)
        ]
        b, b_err = b_value_mle(sub["magnitude"].values, mc)
        feats[f"usgs_b_{wname}"] = b
        feats[f"usgs_b_{wname}_err"] = b_err
        feats[f"usgs_n_{wname}"] = len(sub)

    # b-değeri düşüş trendi (w3→w1)
    b1 = feats.get("usgs_b_w1", np.nan)
    b3 = feats.get("usgs_b_w3", np.nan)
    if pd.notna(b1) and pd.notna(b3):
        feats["usgs_b_drop"] = b3 - b1   # pozitif = b azaldı

    # Genel b-değeri (tüm 3 yıl)
    b_all, _ = b_value_mle(df_mc["magnitude"].values, mc)
    feats["usgs_b_3y"] = b_all

    # ── Aktivite trendi ──────────────────────────────────
    n_w1 = feats.get("usgs_n_w1", 0)
    n_w2 = feats.get("usgs_n_w2", 0)
    n_w3 = feats.get("usgs_n_w3", 0)

    # Quiescence ratio: son 1y / ortalama önceki 2y
    prev_avg = (n_w2 + n_w3) / 2.0
    feats["usgs_quiescence_ratio"] = (
        n_w1 / prev_avg if prev_avg > 0 else np.nan
    )

    # Son 90 gün vs önceki 275 gün hızlanma
    n_90d = len(df_mc[df_mc["days_before"] <= 90])
    n_275d = len(df_mc[
        (df_mc["days_before"] > 90) & (df_mc["days_before"] <= 365)
    ])
    rate_90 = n_90d / 90.0
    rate_275 = n_275d / 275.0
    feats["usgs_accel_90d"] = (
        rate_90 / rate_275 if rate_275 > 0 else np.nan
    )

    # Z-score: son 365 vs önceki 365
    z_1y = z_score_seismicity(n_w1, n_w2, 365, 365)
    feats["usgs_z_score_1y"] = z_1y

    # Z-score: son 90 vs önceki 90
    n_90_prev = len(df_mc[
        (df_mc["days_before"] > 90) & (df_mc["days_before"] <= 180)
    ])
    z_90 = z_score_seismicity(n_90d, n_90_prev, 90, 90)
    feats["usgs_z_score_90d"] = z_90

    # Aylık trend eğimi (son 36 ay)
    monthly = []
    for i in range(36, 0, -1):
        end = main_time - pd.Timedelta(days=(i-1)*30)
        start = main_time - pd.Timedelta(days=i*30)
        n = ((df_mc["datetime_utc"] >= start) &
             (df_mc["datetime_utc"] < end)).sum()
        monthly.append(int(n))
    monthly = np.array(monthly)
    feats["usgs_monthly_slope"] = np.polyfit(
        np.arange(36), monthly, 1
    )[0] if len(monthly) == 36 else np.nan
    feats["usgs_monthly_cv"] = (
        np.std(monthly) / np.mean(monthly)
        if np.mean(monthly) > 0 else np.nan
    )

    # ── Derinlik analizi ─────────────────────────────────
    if "depth_km" in df_mc.columns:
        dep = df_mc["depth_km"].dropna()
        dep_w1 = df_mc[df_mc["days_before"] <= 365]["depth_km"].dropna()
        dep_w3 = df_mc[df_mc["days_before"] > 730]["depth_km"].dropna()

        feats["usgs_mean_depth_3y"] = dep.mean() if len(dep) > 0 else np.nan
        feats["usgs_std_depth_3y"] = dep.std() if len(dep) > 0 else np.nan

        if len(dep_w1) >= 3 and len(dep_w3) >= 3:
            feats["usgs_depth_change"] = dep_w1.mean() - dep_w3.mean()
            _, p_dep = stats.mannwhitneyu(
                dep_w1, dep_w3, alternative="two-sided"
            )
            feats["usgs_depth_change_p"] = p_dep
        else:
            feats["usgs_depth_change"] = np.nan
            feats["usgs_depth_change_p"] = np.nan

    # ── Mekânsal analiz ──────────────────────────────────
    if "lat" in df_mc.columns and "lon" in df_mc.columns:
        df_mc = df_mc.copy()
        df_mc["dist_km"] = haversine_km(
            lat, lon,
            df_mc["lat"].values,
            df_mc["lon"].values
        )
        dist_w1 = df_mc[df_mc["days_before"] <= 365]["dist_km"]
        dist_w3 = df_mc[df_mc["days_before"] > 730]["dist_km"]

        feats["usgs_mean_dist_3y"] = df_mc["dist_km"].mean()

        if len(dist_w1) >= 3 and len(dist_w3) >= 3:
            feats["usgs_dist_change"] = dist_w1.mean() - dist_w3.mean()
            # Negatif = odaklanma (yaklaşma)
        else:
            feats["usgs_dist_change"] = np.nan

        # PCA ile yayılım eksen oranı
        if len(df_mc) >= 10:
            lat0r = np.radians(lat)
            x = (df_mc["lon"].values - lon) * 111.32 * np.cos(lat0r)
            y_coord = (df_mc["lat"].values - lat) * 110.57
            pts = np.column_stack([x, y_coord])
            try:
                pts_c = pts - pts.mean(axis=0)
                cov = np.cov(pts_c.T)
                vals, _ = np.linalg.eigh(cov)
                vals = np.sort(vals)[::-1]
                feats["usgs_pca_var_ratio"] = (
                    vals[0] / vals.sum() if vals.sum() > 0 else np.nan
                )
            except Exception:
                feats["usgs_pca_var_ratio"] = np.nan

    # ── Büyüklük istatistikleri ──────────────────────────
    feats["usgs_mean_mw_3y"] = df_mc["magnitude"].mean()
    feats["usgs_std_mw_3y"] = df_mc["magnitude"].std()
    feats["usgs_max_mw_3y"] = df_mc["magnitude"].max()
    feats["usgs_max_mw_1y"] = (
        df_mc[df_mc["days_before"] <= 365]["magnitude"].max()
        if len(df_mc[df_mc["days_before"] <= 365]) > 0 else np.nan
    )
    feats["usgs_max_mw_norm"] = feats["usgs_max_mw_3y"] - mw

    return feats


# ═══════════════════════════════════════════════════════════
# 2. HİBRİT FEATURE TABLOSU
# ═══════════════════════════════════════════════════════════

def build_hybrid_table(usgs_features_list, gcmt_real_df,
                        gcmt_ctrl_df, radius_km=200):
    """
    USGS feature'larını GCMT feature'larıyla eşleştir.
    Sadece USGS'te bulunan olayları kullan.
    """
    usgs_df = pd.DataFrame(usgs_features_list).dropna(subset=["event_id"])
    print(f"\nUSGS feature tablosu: {len(usgs_df)} olay")

    # GCMT real verisiyle eşleştir
    gcmt_r = gcmt_real_df[gcmt_real_df["radius_km"] == radius_km].copy()

    # event_id eşleştirmesi için olay adı/tarih üzerinden join
    # USGS event_id ile GCMT event_id farklı → tarih+konum eşleştirmesi
    usgs_df["_date"] = pd.to_datetime(usgs_df["date"]).dt.date

    # GCMT'de tarih bazlı en yakın eşleşme
    gcmt_r["_date"] = pd.to_datetime(
        gcmt_r["main_datetime_utc"]
    ).dt.date

    merged_rows = []
    for _, usgs_row in usgs_df.iterrows():
        # Tarih ve fay tipi eşleştirmesi
        candidates = gcmt_r[
            (gcmt_r["_date"] == usgs_row["_date"]) |
            (gcmt_r["main_fault_type"] == usgs_row["fault_type"])
        ]

        # Koordinat bazlı en yakın eşleşme
        if len(candidates) > 0:
            dists = haversine_km(
                usgs_row["lat"], usgs_row["lon"],
                candidates["main_lat"].fillna(0).values,
                candidates["main_lon"].fillna(0).values
            )
            nearest_idx = dists.argmin()
            if dists[nearest_idx] < 150:  # 150km tolerans
                gcmt_row = candidates.iloc[nearest_idx]
                combined = {}
                # USGS feature'ları
                for k, v in usgs_row.items():
                    combined[f"usgs_{k}"] = v if k.startswith("usgs_") else v
                combined.update({
                    k: v for k, v in usgs_row.items()
                })
                # GCMT feature'ları
                for col in gcmt_r.columns:
                    if col not in combined:
                        combined[f"gcmt_{col}"] = gcmt_row.get(col)
                merged_rows.append(combined)

    hybrid_df = pd.DataFrame(merged_rows)
    print(f"Eşleşen hibrit kayıt: {len(hybrid_df)}")
    return hybrid_df


# ═══════════════════════════════════════════════════════════
# 3. USGS-ONLY MODEL
# ═══════════════════════════════════════════════════════════

USGS_FEATURES = [
    "usgs_b_3y",
    "usgs_b_w1",
    "usgs_b_w2",
    "usgs_b_w3",
    "usgs_b_drop",
    "usgs_n_w1",
    "usgs_n_w2",
    "usgs_n_w3",
    "usgs_quiescence_ratio",
    "usgs_accel_90d",
    "usgs_z_score_1y",
    "usgs_z_score_90d",
    "usgs_monthly_slope",
    "usgs_monthly_cv",
    "usgs_mean_depth_3y",
    "usgs_depth_change",
    "usgs_depth_change_p",
    "usgs_mean_dist_3y",
    "usgs_dist_change",
    "usgs_pca_var_ratio",
    "usgs_mean_mw_3y",
    "usgs_std_mw_3y",
    "usgs_max_mw_3y",
    "usgs_max_mw_1y",
    "usgs_max_mw_norm",
]


def evaluate_usgs_model(usgs_features_list):
    """USGS feature'larıyla basit model değerlendirmesi"""
    usgs_df = pd.DataFrame(usgs_features_list)
    usgs_df = usgs_df.dropna(subset=["event_id"])

    if len(usgs_df) < 10:
        print("Yetersiz USGS verisi.")
        return

    print(f"\n{'='*70}")
    print("USGS FEATURE ANALİZİ")
    print(f"{'='*70}")
    print(f"Analiz edilen olay sayısı: {len(usgs_df)}")

    # Mevcut feature'ları göster
    avail = [f for f in USGS_FEATURES if f in usgs_df.columns]
    print(f"Mevcut USGS feature: {len(avail)}")

    # Feature istatistikleri
    print("\nUSGS Feature Özeti:")
    summary_cols = [
        "usgs_b_3y", "usgs_b_drop",
        "usgs_quiescence_ratio", "usgs_accel_90d",
        "usgs_z_score_1y", "usgs_monthly_slope",
        "usgs_depth_change", "usgs_dist_change",
        "usgs_max_mw_norm", "usgs_pca_var_ratio"
    ]
    show = [c for c in summary_cols if c in usgs_df.columns]
    print(usgs_df[show].describe().round(3).to_string())

    # Fay tipine göre
    if "fault_type" in usgs_df.columns:
        print("\nFay tipine göre medyanlar:")
        numeric_cols = usgs_df[show].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if numeric_cols:
            print(usgs_df.groupby("fault_type")[numeric_cols].median()
                  .round(3).to_string())

    # b-değeri düşüş analizi
    print(f"\nb-Değeri Düşüş Analizi:")
    if "usgs_b_drop" in usgs_df.columns:
        b_drop = usgs_df["usgs_b_drop"].dropna()
        n_decrease = (b_drop > 0).sum()
        n_increase = (b_drop <= 0).sum()
        print(f"  b azalan (b3 > b1): {n_decrease}/{len(b_drop)} "
              f"({n_decrease/len(b_drop)*100:.0f}%)")
        print(f"  b artan  (b3 < b1): {n_increase}/{len(b_drop)}")
        for _, row in usgs_df[usgs_df["usgs_b_drop"].notna()].iterrows():
            bd = row["usgs_b_drop"]
            arrow = "↓" if bd > 0 else "↑"
            print(f"    {arrow} {row['name']:<35s} "
                  f"b3→b1: "
                  f"{row.get('usgs_b_w3', np.nan):.3f} → "
                  f"{row.get('usgs_b_w1', np.nan):.3f}  "
                  f"(drop={bd:.3f})")

    # Mekânsal odaklanma
    print(f"\nMekânsal Odaklanma (usgs_dist_change < 0):")
    if "usgs_dist_change" in usgs_df.columns:
        dc = usgs_df["usgs_dist_change"].dropna()
        n_focus = (dc < 0).sum()
        print(f"  Odaklanma gösteren: {n_focus}/{len(dc)} "
              f"({n_focus/len(dc)*100:.0f}%)")

    # Hızlanma
    print(f"\nSon 90 Gün Hızlanma (accel > 1.5):")
    if "usgs_accel_90d" in usgs_df.columns:
        ac = usgs_df["usgs_accel_90d"].dropna()
        n_accel = (ac > 1.5).sum()
        print(f"  Hızlanma gösteren: {n_accel}/{len(ac)} "
              f"({n_accel/len(ac)*100:.0f}%)")

    return usgs_df


# ═══════════════════════════════════════════════════════════
# 4. GELİŞMİŞ HİBRİT MODEL
# ═══════════════════════════════════════════════════════════

HYBRID_FEATURES = [
    # GCMT katmanı (tüm olaylar)
    "quiescence_ratio",
    "accel_90d",
    "monthly_slope_36m",
    "count_0_1y",
    "count_linear_trend",
    "count_accel_ratio",
    "w1_n_events",
    "w1_mean_mw",
    "w1_std_mw",
    "w1_b_value",
    "w3_b_value",
    "b_drop_w3_w1",
    "w1_mean_depth_km",
    "depth_change_km",
    "w1_mean_dist_km",
    "spatial_focus_change",
    "w3_n_events",
    "w1_migration_slope_km_day",
    # Z-score katmanı
    "z_rate_1y",
    "z_b_value_1y",
    "z_max_mw_1y",
    "z_depth_1y",
    "z_dist_1y",
]


def run_hybrid_model(real_norm_path, ctrl_norm_path):
    """
    Normalize edilmiş GCMT verileriyle hibrit model.
    (USGS verisi sınırlı olayda mevcut olduğundan
     şimdilik GCMT normalize katmanıyla çalışır)
    """
    print(f"\n{'='*70}")
    print("HİBRİT MODEL (GCMT Normalize + Z-score)")
    print(f"{'='*70}")

    real = pd.read_csv(real_norm_path)
    ctrl = pd.read_csv(ctrl_norm_path)

    real["target"] = 1
    ctrl["target"] = 0

    # Türetilmiş feature'lar ekle
    for df in [real, ctrl]:
        c0 = df.get("count_0_1y", pd.Series(0, index=df.index))
        c1 = df.get("count_1_2y", pd.Series(0, index=df.index))
        c2 = df.get("count_2_3y", pd.Series(0, index=df.index))
        df["count_linear_trend"] = c0.fillna(0) - c2.fillna(0)
        avg_prev = (c1.fillna(0) + c2.fillna(0)) / 2.0 + 1e-6
        df["count_accel_ratio"] = c0.fillna(0) / avg_prev

        b1 = df.get("w1_b_value", pd.Series(dtype=float))
        b3 = df.get("w3_b_value", pd.Series(dtype=float))
        df["b_drop_w3_w1"] = b3.fillna(np.nan) - b1.fillna(np.nan)

        d1 = df.get("w1_mean_dist_km", pd.Series(dtype=float))
        d3 = df.get("w3_mean_dist_km", pd.Series(dtype=float))
        df["spatial_focus_change"] = d3.fillna(np.nan) - d1.fillna(np.nan)

        dep1 = df.get("w1_mean_depth_km", pd.Series(dtype=float))
        dep3 = df.get("w3_mean_depth_km", pd.Series(dtype=float))
        df["depth_change_km"] = dep1.fillna(np.nan) - dep3.fillna(np.nan)

    avail_r = [f for f in HYBRID_FEATURES if f in real.columns]
    avail_c = [f for f in HYBRID_FEATURES if f in ctrl.columns]
    common = list(set(avail_r) & set(avail_c))

    ft_col = "fault_type" if "fault_type" in real.columns else None

    combined = pd.concat([
        real[common + ["target"] + ([ft_col] if ft_col else [])],
        ctrl[common + ["target"] + ([ft_col] if ft_col else [])],
    ], ignore_index=True)

    X = combined[common]
    y = combined["target"]

    print(f"Veri: {len(X)} örnek, {len(common)} feature")
    print(f"  {int(y.sum())} real, {int((y==0).sum())} ctrl")

    # Modeller
    models = {
        "LR": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", LogisticRegression(
                max_iter=2000, C=0.1,
                class_weight="balanced", random_state=42
            ))
        ]),
        "RF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", RandomForestClassifier(
                n_estimators=300, max_depth=6,
                min_samples_leaf=8, class_weight="balanced",
                max_features="sqrt", random_state=42
            ))
        ]),
        "GB": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", GradientBoostingClassifier(
                n_estimators=300, max_depth=3,
                learning_rate=0.03, min_samples_leaf=15,
                subsample=0.8, random_state=42
            ))
        ]),
    }

    if HAS_XGB:
        from xgboost import XGBClassifier
        models["XGB"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", XGBClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y==0).sum() / max(y.sum(), 1),
                eval_metric="logloss", verbosity=0, random_state=42
            ))
        ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "average_precision": "average_precision",
    }

    print(f"\n  {'Model':<6} {'AUC':>8} {'F1':>8} "
          f"{'Prec':>8} {'Recall':>8} {'AP':>8}")
    print(f"  {'─'*50}")

    best_auc = 0
    best_name = ""
    all_res = {}

    for name, pipe in models.items():
        try:
            res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
            auc = res["test_roc_auc"].mean()
            f1 = res["test_f1"].mean()
            prec = res["test_precision"].mean()
            rec = res["test_recall"].mean()
            ap = res["test_average_precision"].mean()
            auc_std = res["test_roc_auc"].std()

            print(f"  {name:<6} "
                  f"{auc:>8.4f} "
                  f"{f1:>8.4f} "
                  f"{prec:>8.4f} "
                  f"{rec:>8.4f} "
                  f"{ap:>8.4f}  "
                  f"±{auc_std:.4f}")

            all_res[name] = {
                "auc": auc, "f1": f1,
                "prec": prec, "rec": rec
            }
            if auc > best_auc:
                best_auc = auc
                best_name = name
        except Exception as e:
            print(f"  {name}: HATA: {e}")

    # Fay tipine göre AUC
    if ft_col and ft_col in combined.columns:
        fault_col = combined[ft_col]
        print(f"\nFay tipine göre AUC ({best_name}):")
        for ft in ["REVERSE", "STRIKE_SLIP", "NORMAL"]:
            mask = (fault_col == ft)
            X_ft = X[mask]
            y_ft = y[mask]
            if len(X_ft) < 20 or y_ft.sum() < 5:
                continue
            try:
                cv_ft = StratifiedKFold(
                    n_splits=min(5, y_ft.sum()),
                    shuffle=True, random_state=42
                )
                auc_ft = cross_val_score(
                    models[best_name], X_ft, y_ft,
                    cv=cv_ft, scoring="roc_auc"
                )
                print(f"  {ft:<15}: {auc_ft.mean():.4f} ± {auc_ft.std():.4f}")
            except Exception as e:
                print(f"  {ft:<15}: {e}")

    # Feature importance
    print(f"\nFeature Importance ({best_name}):")
    try:
        best_pipe = models[best_name]
        best_pipe.fit(X, y)
        imp = best_pipe.named_steps["mdl"].feature_importances_
        imp_df = pd.DataFrame({
            "feature": common,
            "importance": imp
        }).sort_values("importance", ascending=False)

        for _, row in imp_df.head(15).iterrows():
            is_z = "★" if str(row["feature"]).startswith("z_") else " "
            bar = "█" * int(row["importance"] * 150)
            print(f"  {is_z} {row['feature']:<35s} "
                  f"{row['importance']:.4f} {bar}")
    except Exception as e:
        print(f"  {e}")

    return all_res, best_auc


# ═══════════════════════════════════════════════════════════
# 5. ANA FONKSİYON
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SeismoPattern - FAZ 6: USGS FEATURE ÇIKARIMI")
    print("=" * 70)
    print(f"XGBoost: {'✅' if HAS_XGB else '❌'}")

    # ─── 1. USGS feature çıkar ────────────────────────────
    print("\n" + "─" * 50)
    print("USGS FEATURE ÇIKARIMI")
    print("─" * 50)

    usgs_features_list = []
    for event_info in PRIORITY_EVENTS:
        (event_id, name, date_str,
         lat, lon, mw, fault_type) = event_info
        feats = extract_usgs_features(
            event_id, name, date_str,
            lat, lon, mw, fault_type,
            data_dir="output/usgs_regional"
        )
        if feats:
            usgs_features_list.append(feats)
            print(f"  ✅ {name}: "
                  f"n={feats.get('usgs_n_above_mc', 0)}, "
                  f"Mc={feats.get('usgs_mc', 'N/A')}, "
                  f"b_3y={feats.get('usgs_b_3y', 'N/A'):.3f}"
                  if pd.notna(feats.get("usgs_b_3y")) else
                  f"  ⚠️  {name}: b_3y hesaplanamadı")
        else:
            print(f"  ❌ {name}: veri yok")

    # ─── 2. USGS analizi ──────────────────────────────────
    usgs_df = evaluate_usgs_model(usgs_features_list)

    # Kaydet
    if usgs_df is not None:
        usgs_df.to_csv("output/usgs_features_detail.csv",
                        index=False, encoding="utf-8-sig")
        print(f"\nUSGS feature tablosu kaydedildi.")

    # ─── 3. Hibrit model ──────────────────────────────────
    print("\n" + "─" * 50)
    print("HİBRİT MODEL")
    print("─" * 50)

    hybrid_results, best_auc = run_hybrid_model(
        "output/real_normalized.csv",
        "output/ctrl_normalized.csv"
    )

    # ─── 4. Özet ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FAZ 6 ÖZET")
    print(f"{'='*70}")
    print(f"\nUSGS analiz edilen olay: {len(usgs_features_list)}")
    print(f"Hibrit model en iyi AUC : {best_auc:.4f}")

    # Aşama karşılaştırması
    print(f"\nAşama karşılaştırması:")
    print(f"  Faz 3 (GCMT basic)          : AUC = 0.7217")
    print(f"  Faz 4 (Feature engineering) : AUC = 0.7217")
    print(f"  Faz 5 (Regional normalize)  : AUC = 0.7131")
    print(f"  Faz 6 (Z-score + hibrit)    : AUC = {best_auc:.4f}")

    if best_auc >= 0.75:
        print(f"\n  ✅ 0.75 hedefi aşıldı!")
        print(f"  → Uygulama katmanına geçmeye hazır")
    else:
        delta_needed = 0.75 - best_auc
        print(f"\n  ⚠️  Hedefe {delta_needed:.4f} eksik")
        print(f"  → Daha fazla USGS bölgesel veri gerekiyor")
        print(f"  → Önerilen: ISC kataloğu entegrasyonu")


if __name__ == "__main__":
    main()