#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 2.1: Basitlestirilmis Coulomb Stres
============================================================
Buyuk depremlerin komsu faylara olan stres etkisini hesaplar.

Yontem:
  Okada (1992) tam cozumu yerine, Wells & Coppersmith (1994)
  kirilma boyutlari + uzaklik bazli stres azalimi kullanilir.

  Bu tam Coulomb degil ama:
  - Buyuk depremlerden sonra hangi bolgelerde
    stres arttigi / azaldigi konusunda fikir verir
  - Feature olarak modele girdigi icin AUC'ye katkisi olur

Referanslar:
  - King et al. (1994) Static stress changes
  - Wells & Coppersmith (1994) fault dimensions
  - Stein (1999) role of stress transfer
"""

import math
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta


# Sabitler
MU = 3.3e10          # Rijidite (Pa)
EARTH_R = 6371.0     # km


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(a))


def mw_to_moment(mw):
    """Mw → Sismik Moment (N·m)"""
    return 10 ** (1.5 * mw + 9.05)


def wells_coppersmith(mw, fault_type="ALL"):
    """
    Wells & Coppersmith (1994) ampirik iliskileri.
    Mw'den kirilma boyutlarini tahmin eder.

    Returns: (length_km, width_km, slip_m)
    """
    # log10(L) = a + b * M
    if fault_type == "STRIKE_SLIP":
        a_l, b_l = -3.55, 0.74
        a_w, b_w = -0.76, 0.27
    elif fault_type == "REVERSE":
        a_l, b_l = -2.86, 0.63
        a_w, b_w = -1.61, 0.41
    elif fault_type == "NORMAL":
        a_l, b_l = -2.01, 0.50
        a_w, b_w = -1.14, 0.35
    else:  # ALL
        a_l, b_l = -3.22, 0.69
        a_w, b_w = -1.01, 0.32

    length_km = 10 ** (a_l + b_l * mw)
    width_km  = 10 ** (a_w + b_w * mw)

    # Ortalama kayma: M0 = mu * L * W * D → D = M0 / (mu * L * W)
    m0 = mw_to_moment(mw)
    area = length_km * 1000 * width_km * 1000  # m²
    slip_m = m0 / (MU * area) if area > 0 else 0

    return round(length_km, 1), round(width_km, 1), round(slip_m, 2)


def simple_stress_change(mw, distance_km, fault_type="ALL"):
    """
    Basitlestirilmis stres degisimi hesabi (bar cinsinden).

    Yaklasik formul:
      delta_sigma ≈ C * M0 / r³

    C = geometrik sabit (~7.5e-7 bar icin km ve N·m birimlerinde)

    Bu formul:
    - Uzak alanda (r >> kirilma boyu) gecerlidir
    - Yakin alanda (r < kirilma boyu) asiri tahmin verir
    - Yonu (artis/azalis) vermez, sadece buyukluk

    Returns: stress_change_bar (mutlak deger)
    """
    if distance_km <= 0:
        distance_km = 0.1

    m0 = mw_to_moment(mw)

    # Kirilma boyutu
    length_km, width_km, _ = wells_coppersmith(mw, fault_type)

    # Yakin alan duzeltmesi
    # r < L ise, kirilma icinde → stres drop (~30 bar tipik)
    if distance_km < length_km * 0.5:
        return 30.0  # Tipik stres drop (yakin alan)

    # Uzak alan: 1/r³ azalim
    r_m = distance_km * 1000  # km → m
    C = 7.5e-7  # ampirik sabit

    stress = C * m0 / (r_m ** 3)

    # bar'a cevir (1 Pa = 1e-5 bar)
    stress_bar = stress * 1e-5

    return round(stress_bar, 6)


def compute_cumulative_cff(lat, lon, ref_time, catalog_df,
                            min_mw=6.0, max_dist_km=500,
                            max_years_back=50):
    """
    Bir noktadaki kumulatif Coulomb stres degisimi.

    ref_time'dan onceki max_years_back yil icindeki
    min_mw+ depremlerin toplam stres etkisi.

    Returns: dict with CFF features
    """
    if isinstance(ref_time, str):
        ref_time = datetime.strptime(ref_time, "%Y-%m-%d")

    df = catalog_df.copy()

    # Zaman sutunu
    time_col = None
    for c in ["datetime_utc", "time"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        return {"cff_total": 0, "cff_n_sources": 0}

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Buyukluk sutunu
    mag_col = "mw" if "mw" in df.columns else "magnitude"

    # Koordinat sutunlari
    lat_col = next((c for c in ["eff_lat", "centroid_lat", "lat", "hypo_lat"]
                    if c in df.columns), None)
    lon_col = next((c for c in ["eff_lon", "centroid_lon", "lon", "hypo_lon"]
                    if c in df.columns), None)

    if not all([time_col, mag_col, lat_col, lon_col]):
        return {"cff_total": 0, "cff_n_sources": 0}

    # Filtrele: ref_time'dan once, min_mw+
    cutoff = ref_time - timedelta(days=int(max_years_back * 365.25))
    mask = (
        (df[time_col] >= cutoff) &
        (df[time_col] < ref_time) &
        (df[mag_col] >= min_mw) &
        df[lat_col].notna() &
        df[lon_col].notna()
    )
    sources = df[mask].copy()

    if len(sources) == 0:
        return {
            "cff_total": 0.0,
            "cff_n_sources": 0,
            "cff_max_single": 0.0,
            "cff_nearest_source_km": None,
            "cff_nearest_source_mw": None,
            "cff_nearest_source_date": None,
        }

    # Her kaynak icin stres hesapla
    total_stress = 0.0
    max_single = 0.0
    nearest_dist = float('inf')
    nearest_mw = None
    nearest_date = None
    n_sources = 0

    for _, src in sources.iterrows():
        src_lat = float(src[lat_col])
        src_lon = float(src[lon_col])
        src_mw  = float(src[mag_col])
        src_time= src[time_col]

        dist = haversine_km(lat, lon, src_lat, src_lon)

        if dist > max_dist_km:
            continue

        # Fay tipi (varsa)
        ft = src.get("fault_type", "ALL")
        if pd.isna(ft) or ft not in ["STRIKE_SLIP", "REVERSE", "NORMAL"]:
            ft = "ALL"

        stress = simple_stress_change(src_mw, dist, ft)

        # Zaman azalimi: eski olaylar daha az etkili
        years_ago = (ref_time - src_time).days / 365.25
        time_decay = 1.0 / (1.0 + years_ago / 10.0)

        weighted_stress = stress * time_decay
        total_stress += weighted_stress
        n_sources += 1

        if weighted_stress > max_single:
            max_single = weighted_stress

        if dist < nearest_dist:
            nearest_dist = dist
            nearest_mw = src_mw
            nearest_date = str(src_time)[:10] if pd.notna(src_time) else None

    return {
        "cff_total": round(total_stress, 6),
        "cff_n_sources": n_sources,
        "cff_max_single": round(max_single, 6),
        "cff_nearest_source_km": round(nearest_dist, 1) if nearest_dist < float('inf') else None,
        "cff_nearest_source_mw": nearest_mw,
        "cff_nearest_source_date": nearest_date,
        "cff_log_total": round(math.log10(max(total_stress, 1e-10)), 4),
    }


def compute_cff_for_app(lat, lon, ref_date=None):
    """
    app.py'den cagirilacak fonksiyon.
    GCMT katalogunu kullanarak CFF hesaplar.
    """
    catalog_path = Path("output/all_earthquakes.csv")

    if not catalog_path.exists():
        return {"error": "Katalog bulunamadi"}

    df = pd.read_csv(catalog_path, low_memory=False)

    # Efektif koordinatlar
    if "eff_lat" not in df.columns:
        df["eff_lat"] = df.get("centroid_lat", df.get("hypo_lat"))
    if "eff_lon" not in df.columns:
        df["eff_lon"] = df.get("centroid_lon", df.get("hypo_lon"))

    ref_time = datetime.strptime(ref_date, "%Y-%m-%d") if ref_date else datetime.utcnow()

    result = compute_cumulative_cff(
        lat, lon, ref_time, df,
        min_mw=6.0, max_dist_km=500, max_years_back=50
    )

    return result


# =========================================================
# TEST
# =========================================================

def test():
    print("=" * 60)
    print("COULOMB STRES TEST")
    print("=" * 60)

    # Wells & Coppersmith testi
    print("\nWells & Coppersmith boyutlari:")
    for mw in [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]:
        l, w, s = wells_coppersmith(mw)
        print(f"  Mw {mw}: L={l:>7.1f} km, W={w:>5.1f} km, Slip={s:.2f} m")

    # Stres azalimi testi
    print("\nStres azalimi (Mw 7.5):")
    for dist in [10, 20, 50, 100, 200, 300, 500]:
        s = simple_stress_change(7.5, dist)
        print(f"  {dist:>4} km: {s:.6f} bar")

    # Gercek lokasyonlar
    catalog_path = Path("output/all_earthquakes.csv")
    if not catalog_path.exists():
        print("\nKatalog bulunamadi, CFF testi atlanıyor.")
        return

    print("\nKumulatif CFF (gercek katalog):")
    df = pd.read_csv(catalog_path, low_memory=False)
    if "eff_lat" not in df.columns:
        df["eff_lat"] = df.get("centroid_lat", df.get("hypo_lat"))
    if "eff_lon" not in df.columns:
        df["eff_lon"] = df.get("centroid_lon", df.get("hypo_lon"))

    locations = [
        ("Istanbul",       41.01,  28.98, None),
        ("Kahramanmaras",  37.22,  37.02, "2023-01-06"),
        ("Izmit oncesi",   40.75,  29.86, "1999-07-17"),
        ("Tohoku oncesi",  38.30, 142.37, "2011-02-11"),
        ("Cascadia",       45.50,-125.00, None),
    ]

    for name, lat, lon, ref in locations:
        ref_time = datetime.strptime(ref, "%Y-%m-%d") if ref else datetime.utcnow()
        result = compute_cumulative_cff(
            lat, lon, ref_time, df,
            min_mw=6.0, max_dist_km=500
        )
        print(f"\n  {name}:")
        print(f"    CFF toplam      : {result['cff_total']:.6f} bar")
        print(f"    Kaynak sayisi   : {result['cff_n_sources']}")
        print(f"    Maks tekil      : {result['cff_max_single']:.6f} bar")
        print(f"    En yakin kaynak : {result.get('cff_nearest_source_km')} km "
              f"(Mw {result.get('cff_nearest_source_mw')})")
        print(f"    log10(CFF)      : {result.get('cff_log_total')}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--refdate", type=str, default=None)
    args = ap.parse_args()

    if args.test:
        test()
    elif args.lat and args.lon:
        result = compute_cff_for_app(args.lat, args.lon, args.refdate)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Kullanim:")
        print("  python scripts/coulomb_simple.py --test")
        print("  python scripts/coulomb_simple.py --lat 41.01 --lon 28.98")


if __name__ == "__main__":
    main()