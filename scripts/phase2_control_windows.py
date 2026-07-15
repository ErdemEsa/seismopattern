#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 2 Kontrol Pencereleri
==========================================
Büyük deprem ile sonuçlanmayan zaman pencerelerini oluşturup
aynı feature setini çıkarır.

Yöntem:
  Her Mw 7+ deprem için, aynı bölgeden (aynı lat/lon etrafında)
  ama büyük deprem OLMAYAN dönemlerden kontrol pencereleri seçilir.

  Kontrol penceresi seçim kuralları:
  1) Ana depremin konumunda (aynı lat/lon)
  2) Ana depremden EN AZ 4 yıl ÖNCE veya SONRA
     (ki ana depremin etkisinden uzak olsun)
  3) Kontrol penceresinin 3 yıllık aralığında Mw 7+ olay olmamalı
     (aynı bölge, 300 km içinde)
  4) Her ana deprem için 1-2 kontrol penceresi üretilmeye çalışılır
"""

import argparse
import json
import numpy as np
import pandas as pd
from datetime import timedelta

EARTH_RADIUS_KM = 6371.0


# ═══════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR (phase2_gcmt_features ile aynı)
# ═══════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


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
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    try:
        return np.polyfit(x, y, 1)[0]
    except:
        return np.nan


def monthly_counts(event_times, ref_time, months=36):
    bins = []
    for i in range(months, 0, -1):
        end = ref_time - pd.Timedelta(days=(i - 1) * 30)
        start = ref_time - pd.Timedelta(days=i * 30)
        bins.append(((event_times >= start) & (event_times < end)).sum())
    return np.array(bins, dtype=int)


def local_xy_km(lat0, lon0, lats, lons):
    lat0r = np.radians(lat0)
    x = (lons - lon0) * 111.32 * np.cos(lat0r)
    y = (lats - lat0) * 110.57
    return x, y


def magnitude_to_moment_nm(mw):
    return 10 ** (1.5 * mw + 9.1)


def migration_slope_km_day(df_local, ref_time, lat0, lon0):
    if len(df_local) < 10:
        return np.nan
    lats = df_local["eff_lat"].values
    lons = df_local["eff_lon"].values
    times = df_local["datetime_utc"]
    x, y = local_xy_km(lat0, lon0, lats, lons)
    pts = np.column_stack([x, y])
    try:
        pts_c = pts - pts.mean(axis=0)
        cov = np.cov(pts_c.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        proj = pts_c @ axis
        days_before = (ref_time - times).dt.total_seconds().values / 86400.0
        return linear_slope(-days_before, proj)
    except:
        return np.nan


def summarize_window(dfw, ref_time, lat0, lon0):
    out = {}
    n = len(dfw)
    out["n_events"] = int(n)

    if n == 0:
        for k in ["max_mw", "mean_mw", "std_mw", "b_value", "cum_moment_nm",
                   "mean_depth_km", "std_depth_km", "depth_slope_km_day",
                   "mean_dist_km", "std_dist_km", "migration_slope_km_day"]:
            out[k] = np.nan
        return out

    out["max_mw"] = dfw["mw"].max()
    out["mean_mw"] = dfw["mw"].mean()
    out["std_mw"] = dfw["mw"].std()
    out["b_value"] = b_value_aki(dfw["mw"].values)

    m0_list = []
    for _, r in dfw.iterrows():
        if pd.notna(r.get("scalar_moment_dyncm")):
            m0_list.append(r["scalar_moment_dyncm"] / 1e7)
        elif pd.notna(r.get("mw")):
            m0_list.append(magnitude_to_moment_nm(r["mw"]))
    out["cum_moment_nm"] = np.nansum(m0_list) if m0_list else np.nan

    out["mean_depth_km"] = dfw["eff_depth_km"].mean()
    out["std_depth_km"] = dfw["eff_depth_km"].std()

    days_before = (ref_time - dfw["datetime_utc"]).dt.total_seconds() / 86400.0
    out["depth_slope_km_day"] = linear_slope(-days_before.values, dfw["eff_depth_km"].values)

    dists = haversine_km(lat0, lon0, dfw["eff_lat"].values, dfw["eff_lon"].values)
    out["mean_dist_km"] = np.mean(dists)
    out["std_dist_km"] = np.std(dists)
    out["migration_slope_km_day"] = migration_slope_km_day(dfw, ref_time, lat0, lon0)

    return out


# ═══════════════════════════════════════════════════════════
# KONTROL PENCERESİ OLUŞTURUCU
# ═══════════════════════════════════════════════════════════

def find_control_windows(major_row, df, all_majors_df,
                         min_gap_years=4, n_controls=2,
                         check_radius_km=300):
    """
    Bir ana deprem için kontrol pencereleri bul.
    
    Kurallar:
    - Ana depremden en az min_gap_years yıl uzakta
    - 3 yıllık kontrol penceresi içinde aynı bölgede (check_radius_km)
      Mw 7+ deprem OLMAMALI
    - Kataloğun kapsadığı tarih aralığında olmalı
    """
    main_time = major_row["datetime_utc"]
    lat0 = major_row["eff_lat"]
    lon0 = major_row["eff_lon"]

    catalog_start = df["datetime_utc"].min()
    catalog_end = df["datetime_utc"].max()

    # Bu bölgedeki tüm Mw 7+ olayların tarihlerini bul
    other_majors = all_majors_df.copy()
    other_majors["dist_to_main"] = haversine_km(
        lat0, lon0,
        other_majors["eff_lat"].values,
        other_majors["eff_lon"].values
    )
    nearby_majors = other_majors[other_majors["dist_to_main"] <= check_radius_km]
    major_times = nearby_majors["datetime_utc"].values

    # Aday kontrol zamanları üret
    gap_days = int(min_gap_years * 365.25)
    candidates = []

    # Ana depremden ÖNCE kontrol pencereleri
    # ref_time = kontrol penceresinin "sonu" (yani büyük deprem olacakmış gibi düşündüğümüz an)
    for years_back in [5, 7, 10, 15, 20, 25, 30, 35, 40]:
        ref = main_time - pd.Timedelta(days=int(years_back * 365.25))

        # Kataloğun içinde mi?
        window_start = ref - pd.Timedelta(days=1095)
        if window_start < catalog_start:
            continue
        if ref > catalog_end:
            continue

        # Bu ref zamanının 3 yıllık penceresinde Mw 7+ var mı?
        conflict = False
        for mt in major_times:
            mt_ts = pd.Timestamp(mt)
            if window_start <= mt_ts <= ref:
                conflict = True
                break
            # ref zamanına çok yakın Mw 7+ var mı? (±gap/2)
            if abs((mt_ts - ref).total_seconds()) < gap_days * 86400 / 2:
                conflict = True
                break

        if not conflict:
            candidates.append(ref)

        if len(candidates) >= n_controls:
            break

    # Ana depremden SONRA da dene
    if len(candidates) < n_controls:
        for years_fwd in [5, 7, 10]:
            ref = main_time + pd.Timedelta(days=int(years_fwd * 365.25))

            window_start = ref - pd.Timedelta(days=1095)
            if window_start < catalog_start:
                continue
            if ref > catalog_end:
                continue

            conflict = False
            for mt in major_times:
                mt_ts = pd.Timestamp(mt)
                if window_start <= mt_ts <= ref:
                    conflict = True
                    break
                if abs((mt_ts - ref).total_seconds()) < gap_days * 86400 / 2:
                    conflict = True
                    break

            if not conflict:
                candidates.append(ref)

            if len(candidates) >= n_controls:
                break

    return candidates[:n_controls]


def build_control_features(df, all_majors_df, radii=(100, 200, 300),
                           min_gap_years=4, n_controls=2):
    """Tüm ana depremler için kontrol pencereleri oluştur ve feature çıkar"""

    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
    df["eff_lat"] = df["centroid_lat"].fillna(df["hypo_lat"])
    df["eff_lon"] = df["centroid_lon"].fillna(df["hypo_lon"])
    df["eff_depth_km"] = df["centroid_depth_km"].fillna(df["hypo_depth_km"])

    df = df.dropna(subset=["datetime_utc", "mw", "eff_lat", "eff_lon", "eff_depth_km"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    majors = df[df["mw"] >= 7.0].copy().reset_index(drop=True)
    majors["eff_lat"] = majors["centroid_lat"].fillna(majors["hypo_lat"])
    majors["eff_lon"] = majors["centroid_lon"].fillna(majors["hypo_lon"])
    majors["eff_depth_km"] = majors["centroid_depth_km"].fillna(majors["hypo_depth_km"])

    rows = []
    total_controls = 0
    no_control = 0

    for i, m in majors.iterrows():
        controls = find_control_windows(m, df, majors,
                                        min_gap_years=min_gap_years,
                                        n_controls=n_controls)

        if not controls:
            no_control += 1
            continue

        lat0 = m["eff_lat"]
        lon0 = m["eff_lon"]

        for ci, ref_time in enumerate(controls):
            total_controls += 1

            df3 = df[
                (df["datetime_utc"] < ref_time) &
                (df["datetime_utc"] >= ref_time - pd.Timedelta(days=1095))
            ].copy()

            if len(df3) == 0:
                continue

            df3["distance_km"] = haversine_km(lat0, lon0,
                                               df3["eff_lat"].values,
                                               df3["eff_lon"].values)
            df3["days_before"] = (ref_time - df3["datetime_utc"]).dt.total_seconds() / 86400.0

            for radius in radii:
                local = df3[df3["distance_km"] <= radius].copy()

                w1 = local[local["days_before"] <= 365]
                w2 = local[local["days_before"] <= 730]
                w3 = local[local["days_before"] <= 1095]

                s1 = summarize_window(w1, ref_time, lat0, lon0)
                s2 = summarize_window(w2, ref_time, lat0, lon0)
                s3 = summarize_window(w3, ref_time, lat0, lon0)

                count_0_1y = len(w1)
                count_1_2y = len(local[(local["days_before"] > 365) & (local["days_before"] <= 730)])
                count_2_3y = len(local[(local["days_before"] > 730) & (local["days_before"] <= 1095)])

                prev24 = count_1_2y + count_2_3y
                quiescence_ratio = count_0_1y / (prev24 / 2.0) if prev24 > 0 else np.nan

                last90 = len(local[local["days_before"] <= 90])
                prev270 = len(local[(local["days_before"] > 90) & (local["days_before"] <= 360)])
                accel_90d = (last90 / 90.0) / (prev270 / 270.0) if prev270 > 0 else np.nan

                mc = monthly_counts(local["datetime_utc"], ref_time, months=36)
                monthly_slope_36m = linear_slope(np.arange(36), mc)

                row = {
                    "label": "CONTROL",
                    "control_index": ci,
                    "parent_event_id": m.get("event_id"),
                    "parent_mw": m.get("mw"),
                    "parent_fault_type": m.get("fault_type"),
                    "ref_datetime_utc": ref_time,
                    "ref_lat": lat0,
                    "ref_lon": lon0,
                    "ref_depth_km": m.get("eff_depth_km"),
                    "radius_km": radius,

                    "count_0_1y": count_0_1y,
                    "count_1_2y": count_1_2y,
                    "count_2_3y": count_2_3y,
                    "quiescence_ratio": quiescence_ratio,
                    "accel_90d": accel_90d,
                    "monthly_slope_36m": monthly_slope_36m,
                    "monthly_counts_36m_json": json.dumps(mc.tolist()),
                }

                # w1, w2, w3 feature ekle
                for prefix, s in [("w1", s1), ("w2", s2), ("w3", s3)]:
                    for k, v in s.items():
                        row[f"{prefix}_{k}"] = v

                rows.append(row)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(majors)} ana deprem kontrol penceresi işlendi...")

    feat = pd.DataFrame(rows)

    print(f"\nKontrol penceresi oluşturma tamamlandı.")
    print(f"  Toplam kontrol penceresi: {total_controls}")
    print(f"  Kontrol bulunamayan ana deprem: {no_control}")
    print(f"  Toplam feature satırı: {len(feat):,}")

    return feat


def main():
    import json  # summarize içinde json.dumps kullanılmıyor ama monthly_counts var

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="all_earthquakes.csv")
    ap.add_argument("--output", required=True, help="gcmt_control_features.csv")
    ap.add_argument("--n_controls", type=int, default=2)
    ap.add_argument("--min_gap_years", type=float, default=4.0)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    feat = build_control_features(
        df,
        all_majors_df=df[df["mw"] >= 7.0].copy() if "mw" in df.columns else pd.DataFrame(),
        radii=(100, 200, 300),
        min_gap_years=args.min_gap_years,
        n_controls=args.n_controls
    )

    feat.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\nKaydedildi: {args.output}")
    print(f"Satır: {len(feat):,}")

    if len(feat) > 0 and "quiescence_ratio" in feat.columns:
        print("\nKontrol penceresi özet:")
        print(feat.groupby("radius_km")[
            ["count_0_1y", "count_1_2y", "count_2_3y",
             "quiescence_ratio", "accel_90d"]
        ].median().to_string())


if __name__ == "__main__":
    main()