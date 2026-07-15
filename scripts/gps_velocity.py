#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 2.2: GPS Deformasyon Verisi
====================================================
Nevada Geodetic Laboratory MIDAS velocity field kullanarak
bolgesel deformasyon hizi ve strain rate tahmini.

Veri: http://geodesy.unr.edu/velocities/
Format: MIDAS (Median Interannual Difference Adjusted for Skewness)
"""

import math
import requests
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

GPS_DIR = Path("data/gps")
GPS_DIR.mkdir(parents=True, exist_ok=True)

MIDAS_URL = "http://geodesy.unr.edu/velocities/midas.IGS14.txt"
MIDAS_FILE = GPS_DIR / "midas_velocities.txt"

EARTH_R = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(a))


# =========================================================
# VERI INDIRME VE PARSE
# =========================================================

def download_midas():
    """NGL MIDAS velocity field indir (fallback URL + user-agent + daha uzun timeout)."""
    if MIDAS_FILE.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(
            MIDAS_FILE.stat().st_mtime
        )).days
        if age_days < 30:
            print(f"  MIDAS dosyasi mevcut ({age_days} gun once)")
            return True

    urls = [
        "https://geodesy.unr.edu/velocities/midas.IGS14.txt",
        "http://geodesy.unr.edu/velocities/midas.IGS14.txt",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SeismoPattern/1.0"
    }

    for url in urls:
        print(f"  MIDAS indiriliyor: {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            resp.raise_for_status()

            if len(resp.text) < 1000:
                print("  Yanit cok kisa, atlandi")
                continue

            MIDAS_FILE.write_text(resp.text, encoding="utf-8")
            print(f"  Kaydedildi: {MIDAS_FILE} ({len(resp.text)//1024} KB)")
            return True

        except Exception as e:
            print(f"  URL basarisiz: {e}")

    print("  Tum MIDAS URL denemeleri basarisiz.")
    return False


def parse_midas():
    """
    MIDAS dosyasini dogru kolon indeksleriyle parse et.
    
    Gozlenen format:
      0  : site
      1  : MIDAS5
      2  : start_year
      3  : end_year
      4  : duration_years
      5-7: gozlem sayilari / kalite alanlari
      8  : Ve (m/yr)
      9  : Vn (m/yr)
      10 : Vu (m/yr)
      11 : Se (m/yr)
      12 : Sn (m/yr)
      13 : Su (m/yr)
      ...
      23 : kalite flag
      24 : latitude
      25 : longitude
      26 : height
    """
    if not MIDAS_FILE.exists():
        if not download_midas():
            return pd.DataFrame()

    txt = MIDAS_FILE.read_text(encoding="utf-8", errors="replace")

    low = txt.lower()
    if "<html" in low or "<!doctype" in low:
        print("  UYARI: MIDAS dosyasi HTML/redirect gibi gorunuyor.")
        return pd.DataFrame()

    lines = txt.splitlines()
    records = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("*") or line.startswith("%"):
            continue

        parts = line.split()

        # Beklenen satir uzunlugu
        if len(parts) < 27:
            continue

        try:
            site = parts[0]

            # Enlem-boylam son kolonlarda
            lat = float(parts[24])
            lon = float(parts[25])
            height_m = float(parts[26])

            # Boylam normalize et
            if lon > 180:
                lon -= 360
            if lon < -180:
                lon += 360

            # Mantik kontrolu
            if not (-90 <= lat <= 90):
                continue
            if not (-180 <= lon <= 180):
                continue

            # Hızlar m/yr → mm/yr
            ve = float(parts[8])  * 1000.0
            vn = float(parts[9])  * 1000.0
            vu = float(parts[10]) * 1000.0

            # Belirsizlikler m/yr → mm/yr
            se = float(parts[11]) * 1000.0
            sn = float(parts[12]) * 1000.0
            su = float(parts[13]) * 1000.0

            vh = math.sqrt(ve**2 + vn**2)
            azimuth = math.degrees(math.atan2(ve, vn)) % 360

            records.append({
                "site": site,
                "lat": lat,
                "lon": lon,
                "height_m": round(height_m, 2),
                "ve_mm_yr": round(ve, 3),
                "vn_mm_yr": round(vn, 3),
                "vu_mm_yr": round(vu, 3),
                "se_mm_yr": round(se, 3),
                "sn_mm_yr": round(sn, 3),
                "su_mm_yr": round(su, 3),
                "vh_mm_yr": round(vh, 2),
                "azimuth_deg": round(azimuth, 1),
                "start_year": float(parts[2]),
                "end_year": float(parts[3]),
                "duration_years": float(parts[4]),
                "quality_flag": int(float(parts[23])),
            })

        except Exception:
            # Bozuk satiri atla
            continue

    df = pd.DataFrame(records)

    print(f"  MIDAS parse edildi: {len(df)} istasyon")
    if len(df) > 0:
        print(f"  Ornek istasyonlar:")
        print(df.head(3).to_string(index=False))

    return df


# =========================================================
# GPS VELOCITY DATABASE
# =========================================================

class GPSVelocityDB:
    """GPS velocity field veritabani."""

    def __init__(self):
        self.stations = pd.DataFrame()
        self.spatial_index = {}
        self.grid_size = 2.0

    def load(self):
        """MIDAS verisini yukle ve indeksle."""
        self.stations = parse_midas()

        if len(self.stations) == 0:
            return

        # Spatial index
        for i, row in self.stations.iterrows():
            gx = int(row["lon"] / self.grid_size)
            gy = int(row["lat"] / self.grid_size)
            key = (gx, gy)
            if key not in self.spatial_index:
                self.spatial_index[key] = []
            self.spatial_index[key].append(i)

        print(f"  GPS istasyon: {len(self.stations)}")
        print(f"  Grid hucre: {len(self.spatial_index)}")

    def find_nearby(self, lat, lon, max_dist_km=300, max_results=20):
        """Yakin GPS istasyonlarini bul."""
        if len(self.stations) == 0:
            return pd.DataFrame()

        search_radius = int(max_dist_km / (self.grid_size * 111)) + 1
        gx = int(lon / self.grid_size)
        gy = int(lat / self.grid_size)

        candidate_indices = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                key = (gx + dx, gy + dy)
                if key in self.spatial_index:
                    candidate_indices.extend(self.spatial_index[key])

        if not candidate_indices:
            return pd.DataFrame()

        nearby = self.stations.loc[candidate_indices].copy()
        nearby["dist_km"] = nearby.apply(
            lambda r: haversine_km(lat, lon, r["lat"], r["lon"]),
            axis=1
        )
        nearby = nearby[nearby["dist_km"] <= max_dist_km]
        nearby = nearby.sort_values("dist_km")

        return nearby.head(max_results)

    def compute_strain_features(self, lat, lon, max_dist_km=300):
        """
        Bolgesel deformasyon feature'lari hesapla.

        Hesaplanan metrikler:
        1. Ortalama yatay hiz
        2. Hiz gradyani (strain rate yaklasimi)
        3. Hiz varyasyonu (deformasyon karmasikligi)
        4. Dikey hareket trendi
        """
        nearby = self.find_nearby(lat, lon, max_dist_km)

        if len(nearby) == 0:
            return {
                "gps_n_stations": 0,
                "gps_mean_vh_mm_yr": None,
                "gps_max_vh_mm_yr": None,
                "gps_std_vh_mm_yr": None,
                "gps_mean_vu_mm_yr": None,
                "gps_strain_rate": None,
                "gps_velocity_gradient": None,
                "gps_azimuth_variance": None,
                "gps_nearest_dist_km": None,
                "gps_nearest_site": None,
                "gps_nearest_vh": None,
            }

        n = len(nearby)

        # Temel istatistikler
        mean_vh = round(float(nearby["vh_mm_yr"].mean()), 2)
        max_vh = round(float(nearby["vh_mm_yr"].max()), 2)
        std_vh = round(float(nearby["vh_mm_yr"].std()), 2) if n > 1 else 0
        mean_vu = round(float(nearby["vu_mm_yr"].mean()), 2)

        # En yakin istasyon
        nearest = nearby.iloc[0]
        nearest_dist = round(float(nearest["dist_km"]), 1)
        nearest_site = nearest["site"]
        nearest_vh = round(float(nearest["vh_mm_yr"]), 2)

        # Hiz gradyani (strain rate yaklasimi)
        # Farkli mesafelerdeki istasyonlar arasindaki hiz farki
        strain_rate = None
        velocity_gradient = None

        if n >= 3:
            # En yakin 3+ istasyon arasindaki hiz farki / mesafe farki
            vhs = nearby["vh_mm_yr"].values[:min(n, 10)]
            dists = nearby["dist_km"].values[:min(n, 10)]

            if len(vhs) >= 2 and dists[-1] > dists[0]:
                # Basit gradient: (v_uzak - v_yakin) / (d_uzak - d_yakin)
                grad = abs(vhs[-1] - vhs[0]) / (dists[-1] - dists[0] + 1)
                velocity_gradient = round(float(grad), 4)

                # Strain rate: velocity gradient / mesafe (nano-strain/yil)
                # 1 mm/yr / km ≈ 10^-6 strain/yr = 1 micro-strain/yr
                strain_rate = round(float(grad * 1e-3), 6)  # mm/yr/km → strain/yr

        # Azimuth varyasyonu (hareket yonu ne kadar degisiyor?)
        azimuth_var = None
        if n >= 3:
            azimuths = nearby["azimuth_deg"].values[:min(n, 10)]
            # Dairesel varyans
            sin_sum = np.sum(np.sin(np.radians(azimuths)))
            cos_sum = np.sum(np.cos(np.radians(azimuths)))
            R = math.sqrt(sin_sum**2 + cos_sum**2) / len(azimuths)
            azimuth_var = round(float(1 - R), 4)  # 0 = hep ayni yon, 1 = rastgele

        return {
            "gps_n_stations": n,
            "gps_mean_vh_mm_yr": mean_vh,
            "gps_max_vh_mm_yr": max_vh,
            "gps_std_vh_mm_yr": std_vh,
            "gps_mean_vu_mm_yr": mean_vu,
            "gps_strain_rate": strain_rate,
            "gps_velocity_gradient": velocity_gradient,
            "gps_azimuth_variance": azimuth_var,
            "gps_nearest_dist_km": nearest_dist,
            "gps_nearest_site": nearest_site,
            "gps_nearest_vh": nearest_vh,
        }


# =========================================================
# GLOBAL INSTANCE
# =========================================================

_GPS_DB = None

def get_gps_db():
    global _GPS_DB
    if _GPS_DB is None:
        _GPS_DB = GPSVelocityDB()
        _GPS_DB.load()
    return _GPS_DB


def get_gps_info_for_app(lat, lon):
    """app.py'den cagirilacak fonksiyon."""
    db = get_gps_db()
    if len(db.stations) == 0:
        return None
    return db.compute_strain_features(lat, lon)


# =========================================================
# TEST
# =========================================================

def test():
    print("=" * 60)
    print("GPS VELOCITY TEST")
    print("=" * 60)

    db = get_gps_db()

    if len(db.stations) == 0:
        print("HATA: GPS verisi yuklenemedi!")
        return

    print(f"\nToplam istasyon: {len(db.stations)}")
    print(f"Lat araligi: {db.stations['lat'].min():.1f} - {db.stations['lat'].max():.1f}")
    print(f"Lon araligi: {db.stations['lon'].min():.1f} - {db.stations['lon'].max():.1f}")
    print(f"Ort. yatay hiz: {db.stations['vh_mm_yr'].mean():.1f} mm/yr")

    locations = [
        ("Istanbul",       41.01,  28.98),
        ("Kahramanmaras",  37.22,  37.02),
        ("Izmit",          40.75,  29.86),
        ("San Andreas",    34.05,-118.25),
        ("Tohoku",         38.30, 142.37),
        ("Cascadia",       47.61,-122.33),
        ("Nankai",         34.69, 135.50),
        ("Lima",          -12.05, -77.04),
        ("Nepal",          28.23,  84.73),
    ]

    for name, lat, lon in locations:
        print(f"\n{'--'*30}")
        print(f"  {name} ({lat}, {lon})")

        feats = db.compute_strain_features(lat, lon)

        print(f"  GPS istasyon    : {feats['gps_n_stations']}")
        print(f"  Ort. yatay hiz  : {feats['gps_mean_vh_mm_yr']} mm/yr")
        print(f"  Maks yatay hiz  : {feats['gps_max_vh_mm_yr']} mm/yr")
        print(f"  Std yatay hiz   : {feats['gps_std_vh_mm_yr']} mm/yr")
        print(f"  Ort. dikey hiz  : {feats['gps_mean_vu_mm_yr']} mm/yr")
        print(f"  Strain rate     : {feats['gps_strain_rate']}")
        print(f"  Hiz gradyani    : {feats['gps_velocity_gradient']}")
        print(f"  Azimuth varyans : {feats['gps_azimuth_variance']}")
        print(f"  En yakin ist.   : {feats['gps_nearest_site']} "
              f"({feats['gps_nearest_dist_km']} km, "
              f"{feats['gps_nearest_vh']} mm/yr)")

        nearby = db.find_nearby(lat, lon, max_dist_km=200, max_results=5)
        if len(nearby) > 0:
            print(f"\n  En yakin 5 istasyon:")
            for _, s in nearby.head(5).iterrows():
                print(f"    {s['site']:<6} "
                      f"{s['dist_km']:>6.1f} km  "
                      f"Vh={s['vh_mm_yr']:>6.1f} mm/yr  "
                      f"Az={s['azimuth_deg']:>5.1f} deg  "
                      f"Vu={s['vu_mm_yr']:>+5.1f} mm/yr")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--download", action="store_true",
                    help="MIDAS verisini indir")
    args = ap.parse_args()

    if args.download:
        download_midas()
    elif args.test:
        test()
    elif args.lat and args.lon:
        import json
        result = get_gps_info_for_app(args.lat, args.lon)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Kullanim:")
        print("  python scripts/gps_velocity.py --download")
        print("  python scripts/gps_velocity.py --test")
        print("  python scripts/gps_velocity.py --lat 41.01 --lon 28.98")


if __name__ == "__main__":
    main()