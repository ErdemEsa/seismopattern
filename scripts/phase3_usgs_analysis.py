#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 3 USGS Bölgesel Katalog Analizi
=====================================================
Her öncelikli deprem için:
1. Completeness magnitude (Mc) tahmini
2. b-değeri hesaplama (Mc üzerindeki verilerle)
3. Zaman serisi analizi (aylık/çeyreklik)
4. Mekânsal analiz (göç, kümelenme)
5. Fay tipine göre karşılaştırma
6. GCMT + USGS birleşik profil
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit


# ═══════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def estimate_mc_maxc(magnitudes, dm=0.1):
    """
    Maximum Curvature yöntemi ile Mc tahmini.
    Gutenberg-Richter dağılımının maksimum eğim noktası = Mc
    """
    mags = np.array(magnitudes)
    mags = mags[np.isfinite(mags)]
    if len(mags) < 20:
        return np.nanmin(mags) if len(mags) > 0 else np.nan

    bins = np.arange(np.floor(mags.min() * 10) / 10,
                     np.ceil(mags.max() * 10) / 10 + dm, dm)
    counts, edges = np.histogram(mags, bins=bins)
    if len(counts) == 0:
        return np.nan

    mc_idx = np.argmax(counts)
    mc = edges[mc_idx] + dm / 2
    return round(mc, 1)


def b_value_mle(magnitudes, mc, dm=0.05):
    """
    Maximum Likelihood Estimate ile b-değeri.
    Aki (1965) formülü: b = log10(e) / (mean_M - Mmin + dm/2)
    """
    mags = np.array(magnitudes)
    mags = mags[np.isfinite(mags) & (mags >= mc)]
    n = len(mags)
    if n < 20:
        return np.nan, np.nan

    mean_m = np.mean(mags)
    mmin = mc
    denom = mean_m - (mmin - dm / 2.0)
    if denom <= 0:
        return np.nan, np.nan

    b = np.log10(np.e) / denom
    # Shi & Bolt (1982) standart hata
    b_err = 2.3 * b**2 * np.std(mags, ddof=1) / np.sqrt(n * (n - 1))
    return round(b, 4), round(b_err, 4)


def a_value(magnitudes, mc, b):
    """Gutenberg-Richter a-değeri: N = 10^(a - b*M)"""
    mags = np.array(magnitudes)
    n = np.sum(mags >= mc)
    if n == 0 or b is None or np.isnan(b):
        return np.nan
    return np.log10(n) + b * mc


def gutenberg_richter_residual(mags, mc, b):
    """G-R yasasından sapma (fit kalitesi)"""
    mags = np.array(mags)
    above = mags[mags >= mc]
    if len(above) < 10:
        return np.nan
    bins = np.arange(mc, above.max() + 0.2, 0.1)
    counts, edges = np.histogram(above, bins=bins)
    log_counts = np.where(counts > 0, np.log10(counts), np.nan)
    M_centers = (edges[:-1] + edges[1:]) / 2
    valid = np.isfinite(log_counts)
    if valid.sum() < 3:
        return np.nan
    a_est = a_value(mags, mc, b)
    predicted = a_est - b * M_centers[valid]
    residuals = log_counts[valid] - predicted
    return np.sqrt(np.nanmean(residuals**2))


def z_test_seismicity(counts_recent, counts_past, window_days=365):
    """
    Z-testi ile sismisitedeki değişimi ölç.
    Seismicity Rate Change: Z = (R1 - R2) / sqrt(R1/n1 + R2/n2)
    """
    n1 = len(counts_recent)
    n2 = len(counts_past)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    r1 = n1 / window_days  # oran (olay/gün)
    r2 = n2 / window_days

    if r1 + r2 == 0:
        return 0.0, 1.0

    pooled = (n1 + n2) / (2 * window_days)
    se = np.sqrt(pooled / window_days + pooled / window_days)
    if se == 0:
        return np.nan, np.nan

    z = (r1 - r2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(z, 4), round(p, 6)


def time_series_analysis(df, main_time, mc, label=""):
    """
    Zaman serisi özellikleri:
    - Aylık sayım trendi
    - Son 6 ay vs önceki 6 ay oranı
    - Son 3 ay vs önceki 9 ay oranı
    - Hızlanma (accelerating seismicity)
    """
    df = df[df["magnitude"] >= mc].copy()
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    result = {}

    if len(df) == 0:
        return result

    # Aylık sayımlar (son 36 ay)
    monthly = []
    for i in range(36, 0, -1):
        end = main_time - pd.Timedelta(days=(i-1)*30)
        start = main_time - pd.Timedelta(days=i*30)
        n = ((df["datetime_utc"] >= start) & (df["datetime_utc"] < end)).sum()
        monthly.append(int(n))
    result["monthly_counts"] = monthly

    # Aylık trend (son 36 ay)
    x = np.arange(36)
    y = np.array(monthly)
    if y.sum() > 0:
        slope = np.polyfit(x, y, 1)[0]
        result["monthly_trend_slope"] = round(slope, 6)
    else:
        result["monthly_trend_slope"] = 0.0

    # Dönem karşılaştırmaları
    now = main_time

    # Son 365 gün vs 365-730 gün
    n_last1y = int(((df["datetime_utc"] >= now - pd.Timedelta(days=365)) &
                    (df["datetime_utc"] < now)).sum())
    n_prev1y = int(((df["datetime_utc"] >= now - pd.Timedelta(days=730)) &
                    (df["datetime_utc"] < now - pd.Timedelta(days=365))).sum())
    result["n_last_1y"] = n_last1y
    result["n_prev_1y"] = n_prev1y
    result["ratio_1y_vs_prev"] = n_last1y / n_prev1y if n_prev1y > 0 else np.nan

    # Son 180 gün vs 180-540 gün
    n_last6m = int(((df["datetime_utc"] >= now - pd.Timedelta(days=180)) &
                    (df["datetime_utc"] < now)).sum())
    n_prev6m = int(((df["datetime_utc"] >= now - pd.Timedelta(days=540)) &
                    (df["datetime_utc"] < now - pd.Timedelta(days=180))).sum())
    result["n_last_6m"] = n_last6m
    result["n_prev_6m"] = n_prev6m
    result["ratio_6m_vs_prev"] = n_last6m / n_prev6m if n_prev6m > 0 else np.nan

    # Son 90 gün vs 90-365 gün
    n_last3m = int(((df["datetime_utc"] >= now - pd.Timedelta(days=90)) &
                    (df["datetime_utc"] < now)).sum())
    n_prev3m_base = int(((df["datetime_utc"] >= now - pd.Timedelta(days=365)) &
                          (df["datetime_utc"] < now - pd.Timedelta(days=90))).sum())
    result["n_last_3m"] = n_last3m
    result["n_prev_3m_base"] = n_prev3m_base

    # Oranı günlük orana normalize et
    rate_last3m = n_last3m / 90.0
    rate_prev3m = n_prev3m_base / 275.0
    result["accel_ratio_3m"] = (rate_last3m / rate_prev3m
                                 if rate_prev3m > 0 else np.nan)

    # Z-testi: son 365 gün vs önceki 365 gün
    recent_idx = (df["datetime_utc"] >= now - pd.Timedelta(days=365))
    past_idx = ((df["datetime_utc"] >= now - pd.Timedelta(days=730)) &
                (df["datetime_utc"] < now - pd.Timedelta(days=365)))
    z, p = z_test_seismicity(df[recent_idx], df[past_idx])
    result["z_score"] = z
    result["z_pvalue"] = p

    # Quiescence tespiti: 3y içinde en sakin dönem
    min_count_month = min(monthly) if monthly else 0
    max_count_month = max(monthly) if monthly else 0
    result["min_monthly"] = min_count_month
    result["max_monthly"] = max_count_month
    result["monthly_cv"] = (np.std(monthly) / np.mean(monthly)
                             if np.mean(monthly) > 0 else np.nan)

    return result


def spatial_analysis(df, lat0, lon0, mc, main_time):
    """
    Mekânsal analiz:
    - Centroid hareketi (zaman içinde)
    - Uzaklık dağılımı
    - Mekânsal yoğunlaşma (son 1y vs 2-3y)
    - PCA ile ana aktivite ekseni
    """
    df = df[df["magnitude"] >= mc].copy()
    result = {}

    if len(df) < 10:
        return result

    df["dist_km"] = haversine_km(lat0, lon0, df["lat"].values, df["lon"].values)

    # Genel uzaklık istatistikleri
    result["mean_dist_km"] = round(df["dist_km"].mean(), 2)
    result["std_dist_km"] = round(df["dist_km"].std(), 2)
    result["median_dist_km"] = round(df["dist_km"].median(), 2)

    # Son 1y vs 2-3y mekânsal karşılaştırma
    now = main_time
    last1y = df[df["days_before"] <= 365]
    prev2y = df[(df["days_before"] > 365) & (df["days_before"] <= 1095)]

    if len(last1y) >= 5 and len(prev2y) >= 5:
        result["mean_dist_last1y"] = round(last1y["dist_km"].mean(), 2)
        result["mean_dist_prev2y"] = round(prev2y["dist_km"].mean(), 2)
        result["dist_change_km"] = round(
            last1y["dist_km"].mean() - prev2y["dist_km"].mean(), 2
        )
        # Yaklaşma: negatif = merkezleşme
        result["spatial_focusing"] = result["dist_change_km"] < 0

    # PCA ile ana aktivite ekseni
    if len(df) >= 10:
        lat0r = np.radians(lat0)
        x = (df["lon"].values - lon0) * 111.32 * np.cos(lat0r)
        y = (df["lat"].values - lat0) * 110.57
        pts = np.column_stack([x, y])
        try:
            pts_c = pts - pts.mean(axis=0)
            cov = np.cov(pts_c.T)
            vals, vecs = np.linalg.eigh(cov)
            idx = np.argmax(vals)
            result["pca_var_ratio"] = round(vals[idx] / vals.sum(), 4)
            result["pca_azimuth_deg"] = round(
                np.degrees(np.arctan2(vecs[0, idx], vecs[1, idx])) % 180, 1
            )
        except Exception:
            pass

    # Göç trendi (zaman ile uzaklık değişimi)
    if len(df) >= 10:
        days = df["days_before"].values
        dists = df["dist_km"].values
        try:
            slope, intercept, r, p, se = stats.linregress(-days, dists)
            result["migration_slope_km_day"] = round(slope, 6)
            result["migration_r2"] = round(r**2, 4)
            result["migration_pvalue"] = round(p, 6)
        except Exception:
            pass

    return result


def depth_analysis(df, mc, main_time):
    """Derinlik değişimi analizi"""
    df = df[df["magnitude"] >= mc].copy()
    result = {}

    if len(df) < 10:
        return result

    result["mean_depth"] = round(df["depth_km"].mean(), 2)
    result["std_depth"] = round(df["depth_km"].std(), 2)
    result["median_depth"] = round(df["depth_km"].median(), 2)

    # Son 1y vs 2-3y derinlik karşılaştırması
    last1y = df[df["days_before"] <= 365]
    prev2y = df[(df["days_before"] > 365) & (df["days_before"] <= 1095)]

    if len(last1y) >= 5 and len(prev2y) >= 5:
        result["mean_depth_last1y"] = round(last1y["depth_km"].mean(), 2)
        result["mean_depth_prev2y"] = round(prev2y["depth_km"].mean(), 2)
        result["depth_change_km"] = round(
            last1y["depth_km"].mean() - prev2y["depth_km"].mean(), 2
        )
        try:
            _, p_depth = stats.mannwhitneyu(
                last1y["depth_km"], prev2y["depth_km"],
                alternative="two-sided"
            )
            result["depth_change_pvalue"] = round(p_depth, 6)
        except Exception:
            pass

    # Zaman ile derinlik trendi
    if len(df) >= 10:
        days = df["days_before"].values
        depths = df["depth_km"].values
        try:
            slope, _, r, p, _ = stats.linregress(-days, depths)
            result["depth_trend_km_day"] = round(slope, 6)
            result["depth_trend_r2"] = round(r**2, 4)
        except Exception:
            pass

    return result


def b_value_time_windows(df, mc, main_time):
    """
    b-değerinin zaman içindeki değişimi.
    3 pencere: 0-1y, 1-2y, 2-3y öncesi
    """
    result = {}
    windows = {
        "w1": (0, 365),
        "w2": (365, 730),
        "w3": (730, 1095),
    }
    for wname, (d_min, d_max) in windows.items():
        subset = df[
            (df["days_before"] >= d_min) &
            (df["days_before"] < d_max) &
            (df["magnitude"] >= mc)
        ]
        b, b_err = b_value_mle(subset["magnitude"].values, mc)
        result[f"b_{wname}"] = b
        result[f"b_{wname}_err"] = b_err
        result[f"n_{wname}"] = len(subset)

    # b-değeri trendi (w3 → w2 → w1)
    b_vals = [result.get("b_w3"), result.get("b_w2"), result.get("b_w1")]
    b_vals_clean = [v for v in b_vals if v is not None and not np.isnan(v)]
    if len(b_vals_clean) >= 2:
        slope = np.polyfit(range(len(b_vals_clean)), b_vals_clean, 1)[0]
        result["b_trend_w3_to_w1"] = round(slope, 6)
        # Negatif trend: b azalıyor = gerilim artıyor
        result["b_decreasing"] = slope < 0
    else:
        result["b_trend_w3_to_w1"] = np.nan
        result["b_decreasing"] = None

    return result


def analyze_event(filepath, event_info):
    """Tek bir olay için tüm analizleri çalıştır"""
    event_id, name, date_str, lat, lon, mw, fault_type = event_info

    print(f"\n{'─'*60}")
    print(f"Analiz: {name} ({mw} Mw, {date_str})")

    if not os.path.exists(filepath):
        print(f"  Dosya bulunamadı: {filepath}")
        return None

    df = pd.read_csv(filepath)
    if df.empty:
        print(f"  Boş dosya.")
        return None

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df = df.dropna(subset=["datetime_utc", "magnitude", "lat", "lon"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    main_time = pd.Timestamp(date_str)
    df["days_before"] = (main_time - df["datetime_utc"]).dt.total_seconds() / 86400.0
    df = df[df["days_before"] >= 0].copy()

    n_total = len(df)
    print(f"  Toplam kayıt: {n_total}")

    if n_total < 20:
        print(f"  ⚠️  Yetersiz veri ({n_total} < 20). Analiz sınırlı.")

    # Completeness magnitude
    mc = estimate_mc_maxc(df["magnitude"].values)
    n_above_mc = int((df["magnitude"] >= mc).sum())
    print(f"  Mc = {mc}, n(Mc+) = {n_above_mc}")

    # b-değeri
    b_all, b_err = b_value_mle(df["magnitude"].values, mc)
    print(f"  b-değeri (3y) = {b_all} ± {b_err}")

    # GR sapması
    gr_residual = gutenberg_richter_residual(df["magnitude"].values, mc, b_all)

    # Zaman serisi
    ts = time_series_analysis(df, main_time, mc)

    # Mekânsal
    sp = spatial_analysis(df, lat, lon, mc, main_time)

    # Derinlik
    dp = depth_analysis(df, mc, main_time)

    # b-değeri zaman pencereleri
    bw = b_value_time_windows(df, mc, main_time)

    # Sonuç birleştir
    result = {
        "event_id": event_id,
        "name": name,
        "date": date_str,
        "mw": mw,
        "fault_type": fault_type,
        "lat": lat,
        "lon": lon,
        "n_total": n_total,
        "mc": mc,
        "n_above_mc": n_above_mc,
        "b_value_3y": b_all,
        "b_value_err": b_err,
        "gr_residual": gr_residual,
    }
    result.update({f"ts_{k}": v for k, v in ts.items()
                   if not isinstance(v, list)})
    result["monthly_counts_json"] = json.dumps(
        ts.get("monthly_counts", [])
    )
    result.update({f"sp_{k}": v for k, v in sp.items()})
    result.update({f"dp_{k}": v for k, v in dp.items()})
    result.update(bw)

    # Yazdır
    print(f"  Aylık trend eğimi   : {ts.get('monthly_trend_slope', 'N/A')}")
    print(f"  Son1y / Önceki1y   : {ts.get('ratio_1y_vs_prev', 'N/A')}")
    print(f"  Hızlanma (3m)      : {ts.get('accel_ratio_3m', 'N/A')}")
    print(f"  Z-skoru            : {ts.get('z_score', 'N/A')}")
    print(f"  b(w3→w1) trend     : {bw.get('b_trend_w3_to_w1', 'N/A')}")
    print(f"  b azalıyor mu?     : {bw.get('b_decreasing', 'N/A')}")
    print(f"  Mekânsal odaklanma : {sp.get('spatial_focusing', 'N/A')}")
    print(f"  Derinlik değişimi  : {dp.get('depth_change_km', 'N/A')} km")

    return result


PRIORITY_EVENTS = [
    ("tohoku_2011",        "Tohoku Japonya 2011",     "2011-03-11", 38.297, 142.373, 9.1, "REVERSE"),
    ("sumatra_2004",       "Sumatra 2004",             "2004-12-26",  3.295,  95.982, 9.1, "REVERSE"),
    ("chile_2010",         "Maule Şili 2010",          "2010-02-27",-35.846, -72.719, 8.8, "REVERSE"),
    ("alaska_1964",        "Alaska 1964",              "1964-03-28", 61.05, -147.07,  9.2, "REVERSE"),
    ("chile_2015",         "Illapel Şili 2015",        "2015-09-16",-31.573, -71.674, 8.3, "REVERSE"),
    ("sumatra_2012",       "N.Sumatra 2012",           "2012-04-11",  2.327,  93.063, 8.6, "STRIKE_SLIP"),
    ("kahramanmaras_2023", "Kahramanmaraş 2023",       "2023-02-06", 37.220,  37.019, 7.8, "STRIKE_SLIP"),
    ("izmit_1999",         "İzmit 1999",               "1999-08-17", 40.748,  29.864, 7.6, "STRIKE_SLIP"),
    ("landers_1992",       "Landers ABD 1992",         "1992-06-28", 34.200,-116.437, 7.3, "STRIKE_SLIP"),
    ("ridgecrest_2019",    "Ridgecrest ABD 2019",      "2019-07-06", 35.770,-117.599, 7.1, "STRIKE_SLIP"),
    ("north_iran_1990",    "Kuzey İran 1990",          "1990-06-20", 36.957,  49.409, 7.4, "REVERSE"),
    ("bam_2003",           "Bam İran 2003",            "2003-12-26", 29.010,  58.310, 6.6, "STRIKE_SLIP"),
    ("nias_2005",          "Nias Endonezya 2005",      "2005-03-28",  1.665,  97.004, 8.6, "REVERSE"),
    ("kuril_2006",         "Kuril 2006",               "2006-11-15", 46.607, 153.266, 8.3, "REVERSE"),
    ("samoa_2009",         "Samoa 2009",               "2009-09-29",-15.489,-172.095, 8.1, "NORMAL"),
    ("haiti_2010",         "Haiti 2010",               "2010-01-12", 18.443, -72.571, 7.0, "STRIKE_SLIP"),
    ("nepal_2015",         "Nepal 2015",               "2015-04-25", 28.231,  84.731, 7.8, "REVERSE"),
    ("new_zealand_2016",   "Kaikoura YZ 2016",         "2016-11-13",-42.757, 173.054, 7.8, "STRIKE_SLIP"),
    ("mexico_2017",        "Puebla Meksika 2017",      "2017-09-19", 18.584, -98.399, 7.1, "NORMAL"),
    ("palu_2018",          "Palu Endonezya 2018",      "2018-09-28", -0.256, 119.846, 7.5, "STRIKE_SLIP"),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="output/usgs_regional")
    ap.add_argument("--output", default="output/usgs_analysis.csv")
    args = ap.parse_args()

    print("=" * 70)
    print("SEISMOPattern - FAZ 3 USGS BÖLGESEL ANALİZ")
    print("=" * 70)

    results = []
    for event_info in PRIORITY_EVENTS:
        event_id = event_info[0]
        filepath = os.path.join(args.input_dir, f"{event_id}_precursor.csv")
        r = analyze_event(filepath, event_info)
        if r:
            results.append(r)

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output, index=False, encoding="utf-8-sig")

    # ── Özet tablo ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("ÖZET: FAY TİPİNE GÖRE USGS ANALİZ SONUÇLARI")
    print("=" * 90)

    show_cols = [
        "name", "mw", "fault_type", "mc", "n_above_mc",
        "b_value_3y", "b_w1", "b_w2", "b_w3",
        "b_trend_w3_to_w1", "b_decreasing",
        "ts_ratio_1y_vs_prev", "ts_accel_ratio_3m", "ts_z_score",
        "sp_spatial_focusing", "sp_dist_change_km",
        "dp_depth_change_km"
    ]

    show = df_out[[c for c in show_cols if c in df_out.columns]]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(show.to_string(index=False))

    # ── Fay tipi bazlı özet ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("FAY TİPİ BAZLI MEDYANLAR")
    print("─" * 70)

    numeric_cols = [c for c in show_cols
                    if c in df_out.columns
                    and df_out[c].dtype in [float, int, "float64", "int64"]]

    valid = df_out[df_out["n_above_mc"] >= 20]
    if len(valid) > 0:
        print(valid.groupby("fault_type")[numeric_cols].median().to_string())
    else:
        print("Yeterli veri yok (n_above_mc >= 20).")

    # ── b-değeri trend analizi ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("b-DEĞERİ TREND ANALİZİ (w3 → w2 → w1, yani eskiden yeniye)")
    print("─" * 70)
    for _, row in df_out.iterrows():
        b3 = row.get("b_w3", np.nan)
        b2 = row.get("b_w2", np.nan)
        b1 = row.get("b_w1", np.nan)
        trend = row.get("b_trend_w3_to_w1", np.nan)
        dec = row.get("b_decreasing", None)
        arrow = "↓" if dec else ("↑" if dec is False else "?")
        print(f"  {arrow} {row['name']:<30s} "
              f"b3={b3:.3f}  b2={b2:.3f}  b1={b1:.3f}  "
              f"trend={trend:.5f}" if all(pd.notna([b3, b2, b1, trend]))
              else f"  ? {row['name']:<30s}  (yetersiz veri)")

    # ── Kritik bulgular ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KRİTİK BULGULAR")
    print("=" * 70)

    # b azalan olaylar
    b_dec = df_out[df_out["b_decreasing"] == True]
    print(f"\nb-değeri azalan ({len(b_dec)}/{len(df_out)} olay):")
    for _, r in b_dec.iterrows():
        print(f"  {r['name']} ({r['fault_type']}) "
              f"b3→b1: {r.get('b_w3', np.nan):.3f}→{r.get('b_w1', np.nan):.3f}")

    # Mekânsal odaklanma
    sp_foc = df_out[df_out["sp_spatial_focusing"] == True]
    print(f"\nMekânsal odaklanma ({len(sp_foc)}/{len(df_out)} olay):")
    for _, r in sp_foc.iterrows():
        print(f"  {r['name']} ({r['fault_type']}) "
              f"dist_change={r.get('sp_dist_change_km', np.nan):.1f} km")

    # Hızlanma
    accel = df_out[df_out["ts_accel_ratio_3m"].notna() &
                   (df_out["ts_accel_ratio_3m"] > 1.5)]
    print(f"\nHızlanma (accel_3m > 1.5): {len(accel)}/{len(df_out)} olay")
    for _, r in accel.iterrows():
        print(f"  {r['name']} ({r['fault_type']}) "
              f"accel={r.get('ts_accel_ratio_3m', np.nan):.2f}x")

    print(f"\nKaydedildi: {args.output}")
    print(f"Toplam analiz edilen olay: {len(results)}")


if __name__ == "__main__":
    main()