#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pandas as pd
from datetime import timedelta

EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c

def magnitude_to_moment_nm(mw):
    # log10(M0[Nm]) = 1.5*Mw + 9.1
    return 10 ** (1.5 * mw + 9.1)

def b_value_aki(mags, dm=0.1):
    mags = np.array(mags, dtype=float)
    mags = mags[np.isfinite(mags)]
    if len(mags) < 20:
        return np.nan
    mmin = np.min(mags)
    mean_m = np.mean(mags)
    denom = mean_m - (mmin - dm / 2.0)
    if denom <= 0:
        return np.nan
    return np.log10(np.e) / denom

def linear_slope(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    try:
        return np.polyfit(x, y, 1)[0]
    except:
        return np.nan

def monthly_counts(event_times, main_time, months=36):
    # son 36 ay için ay bazlı sayım
    bins = []
    for i in range(months, 0, -1):
        end = main_time - pd.Timedelta(days=(i - 1) * 30)
        start = main_time - pd.Timedelta(days=i * 30)
        bins.append(((event_times >= start) & (event_times < end)).sum())
    return np.array(bins, dtype=int)

def local_xy_km(lat0, lon0, lats, lons):
    # yaklaşık lokal projeksiyon
    lat0r = np.radians(lat0)
    x = (lons - lon0) * 111.32 * np.cos(lat0r)
    y = (lats - lat0) * 110.57
    return x, y

def migration_slope_km_day(df_local, main_time, lat0, lon0):
    if len(df_local) < 10:
        return np.nan

    lats = df_local["eff_lat"].values
    lons = df_local["eff_lon"].values
    times = df_local["datetime_utc"]

    x, y = local_xy_km(lat0, lon0, lats, lons)
    pts = np.column_stack([x, y])

    try:
        pts_centered = pts - pts.mean(axis=0)
        cov = np.cov(pts_centered.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        proj = pts_centered @ axis

        days_before = (main_time - times).dt.total_seconds().values / 86400.0
        # zaman ileriye gittikçe ana depreme yaklaşma için negatiften kaçınmak adına
        t = -days_before
        return linear_slope(t, proj)
    except:
        return np.nan

def summarize_window(dfw, main_time, lat0, lon0):
    out = {}
    n = len(dfw)
    out["n_events"] = int(n)

    if n == 0:
        out.update({
            "max_mw": np.nan,
            "mean_mw": np.nan,
            "std_mw": np.nan,
            "b_value": np.nan,
            "cum_moment_nm": np.nan,
            "mean_depth_km": np.nan,
            "std_depth_km": np.nan,
            "depth_slope_km_day": np.nan,
            "mean_dist_km": np.nan,
            "std_dist_km": np.nan,
            "migration_slope_km_day": np.nan
        })
        return out

    out["max_mw"] = dfw["mw"].max()
    out["mean_mw"] = dfw["mw"].mean()
    out["std_mw"] = dfw["mw"].std()
    out["b_value"] = b_value_aki(dfw["mw"].values)

    if "scalar_moment_dyncm" in dfw.columns:
        m0_nm = []
        for _, r in dfw.iterrows():
            if pd.notna(r.get("scalar_moment_dyncm")):
                m0_nm.append(r["scalar_moment_dyncm"] / 1e7)
            elif pd.notna(r.get("mw")):
                m0_nm.append(magnitude_to_moment_nm(r["mw"]))
        out["cum_moment_nm"] = np.nansum(m0_nm) if len(m0_nm) else np.nan
    else:
        out["cum_moment_nm"] = np.nansum(magnitude_to_moment_nm(dfw["mw"].values))

    out["mean_depth_km"] = dfw["eff_depth_km"].mean()
    out["std_depth_km"] = dfw["eff_depth_km"].std()

    days_before = (main_time - dfw["datetime_utc"]).dt.total_seconds() / 86400.0
    out["depth_slope_km_day"] = linear_slope(-days_before.values, dfw["eff_depth_km"].values)

    dists = haversine_km(lat0, lon0, dfw["eff_lat"].values, dfw["eff_lon"].values)
    out["mean_dist_km"] = np.mean(dists)
    out["std_dist_km"] = np.std(dists)

    out["migration_slope_km_day"] = migration_slope_km_day(dfw, main_time, lat0, lon0)

    return out

def build_features(df, radii=(100, 200, 300)):
    df = df.copy()

    # datetime
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")

    # efektif konum/depth
    df["eff_lat"] = df["centroid_lat"].fillna(df["hypo_lat"])
    df["eff_lon"] = df["centroid_lon"].fillna(df["hypo_lon"])
    df["eff_depth_km"] = df["centroid_depth_km"].fillna(df["hypo_depth_km"])

    # kritik alan filtresi
    df = df[
        df["datetime_utc"].notna() &
        df["mw"].notna() &
        df["eff_lat"].notna() &
        df["eff_lon"].notna() &
        df["eff_depth_km"].notna()
    ].copy()

    df = df.sort_values("datetime_utc").reset_index(drop=True)

    majors = df[df["mw"] >= 7.0].copy().reset_index(drop=True)

    rows = []

    for i, m in majors.iterrows():
        main_time = m["datetime_utc"]
        lat0 = m["eff_lat"]
        lon0 = m["eff_lon"]

        df3 = df[
            (df["datetime_utc"] < main_time) &
            (df["datetime_utc"] >= main_time - pd.Timedelta(days=1095))
        ].copy()

        if len(df3) == 0:
            continue

        df3["distance_km"] = haversine_km(lat0, lon0, df3["eff_lat"].values, df3["eff_lon"].values)
        df3["days_before"] = (main_time - df3["datetime_utc"]).dt.total_seconds() / 86400.0

        for radius in radii:
            local = df3[df3["distance_km"] <= radius].copy()

            w1 = local[local["days_before"] <= 365].copy()
            w2 = local[local["days_before"] <= 730].copy()
            w3 = local[local["days_before"] <= 1095].copy()

            s1 = summarize_window(w1, main_time, lat0, lon0)
            s2 = summarize_window(w2, main_time, lat0, lon0)
            s3 = summarize_window(w3, main_time, lat0, lon0)

            # oranlar / trendler
            count_0_1y = len(w1)
            count_1_2y = len(local[(local["days_before"] > 365) & (local["days_before"] <= 730)])
            count_2_3y = len(local[(local["days_before"] > 730) & (local["days_before"] <= 1095)])

            prev24 = count_1_2y + count_2_3y
            quiescence_ratio = np.nan
            if prev24 > 0:
                quiescence_ratio = count_0_1y / (prev24 / 2.0)

            last90 = len(local[local["days_before"] <= 90])
            prev270 = len(local[(local["days_before"] > 90) & (local["days_before"] <= 360)])
            accel_90d = np.nan
            if prev270 > 0:
                accel_90d = (last90 / 90.0) / (prev270 / 270.0)

            # son 36 ay aylık sayım trendi
            mc = monthly_counts(local["datetime_utc"], main_time, months=36)
            monthly_slope_36m = linear_slope(np.arange(36), mc)

            row = {
                "main_event_id": m.get("event_id"),
                "main_datetime_utc": main_time,
                "main_region": m.get("region"),
                "main_mw": m.get("mw"),
                "main_fault_type": m.get("fault_type"),
                "main_lat": lat0,
                "main_lon": lon0,
                "main_depth_km": m.get("eff_depth_km"),
                "radius_km": radius,

                "count_0_1y": count_0_1y,
                "count_1_2y": count_1_2y,
                "count_2_3y": count_2_3y,
                "quiescence_ratio": quiescence_ratio,
                "accel_90d": accel_90d,
                "monthly_slope_36m": monthly_slope_36m,
                "monthly_counts_36m_json": json.dumps(mc.tolist()),

                # 1y
                "w1_n_events": s1["n_events"],
                "w1_max_mw": s1["max_mw"],
                "w1_mean_mw": s1["mean_mw"],
                "w1_std_mw": s1["std_mw"],
                "w1_b_value": s1["b_value"],
                "w1_cum_moment_nm": s1["cum_moment_nm"],
                "w1_mean_depth_km": s1["mean_depth_km"],
                "w1_std_depth_km": s1["std_depth_km"],
                "w1_depth_slope_km_day": s1["depth_slope_km_day"],
                "w1_mean_dist_km": s1["mean_dist_km"],
                "w1_std_dist_km": s1["std_dist_km"],
                "w1_migration_slope_km_day": s1["migration_slope_km_day"],

                # 2y
                "w2_n_events": s2["n_events"],
                "w2_max_mw": s2["max_mw"],
                "w2_mean_mw": s2["mean_mw"],
                "w2_std_mw": s2["std_mw"],
                "w2_b_value": s2["b_value"],
                "w2_cum_moment_nm": s2["cum_moment_nm"],
                "w2_mean_depth_km": s2["mean_depth_km"],
                "w2_std_depth_km": s2["std_depth_km"],
                "w2_depth_slope_km_day": s2["depth_slope_km_day"],
                "w2_mean_dist_km": s2["mean_dist_km"],
                "w2_std_dist_km": s2["std_dist_km"],
                "w2_migration_slope_km_day": s2["migration_slope_km_day"],

                # 3y
                "w3_n_events": s3["n_events"],
                "w3_max_mw": s3["max_mw"],
                "w3_mean_mw": s3["mean_mw"],
                "w3_std_mw": s3["std_mw"],
                "w3_b_value": s3["b_value"],
                "w3_cum_moment_nm": s3["cum_moment_nm"],
                "w3_mean_depth_km": s3["mean_depth_km"],
                "w3_std_depth_km": s3["std_depth_km"],
                "w3_depth_slope_km_day": s3["depth_slope_km_day"],
                "w3_mean_dist_km": s3["mean_dist_km"],
                "w3_std_dist_km": s3["std_dist_km"],
                "w3_migration_slope_km_day": s3["migration_slope_km_day"],
            }

            rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(majors)} ana deprem işlendi...")

    feat = pd.DataFrame(rows)
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="all_earthquakes.csv")
    ap.add_argument("--output", required=True, help="gcmt_precursor_features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    feat = build_features(df, radii=(100, 200, 300))
    feat.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\nTamamlandı.")
    print(f"Feature satırı: {len(feat):,}")
    print(f"Ana deprem sayısı: {feat['main_event_id'].nunique():,}")
    print(f"Yarıçaplar: {sorted(feat['radius_km'].unique().tolist())}")

    print("\nÖzet:")
    print(
        feat.groupby("radius_km")[["count_0_1y", "count_1_2y", "count_2_3y", "quiescence_ratio", "accel_90d"]]
        .median()
    )

if __name__ == "__main__":
    main()