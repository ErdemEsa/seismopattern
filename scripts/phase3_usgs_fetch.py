#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - USGS Bölgesel Katalog Çekici
=============================================
Seçilmiş Mw 7.5+ büyük depremler için
USGS API'sinden önceki 3 yıllık Mw 2.5+ verileri çeker.

USGS ComCat API:
  https://earthquake.usgs.gov/fdsnws/event/1/

Neden Mw 7.5+ seçiyoruz?
  - Bölgesel ağların daha uzun süredir aktif olduğu dönemleri kapsamak için
  - 7.5+ olaylar için USGS'nin yerel ağ verisi daha eksiksiz
  - Her bölgede farklı completeness threshold var (ortalama ~Mw 2.5)
"""

import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

API_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def fetch_usgs(lat, lon, radius_km, start_time, end_time,
               min_magnitude=2.5, max_results=10000):
    """USGS API'den deprem verisi çek"""
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_magnitude,
        "orderby": "time-asc",
        "limit": max_results,
    }

    try:
        resp = requests.get(API_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            return pd.DataFrame()

        records = []
        for f in features:
            p = f["properties"]
            g = f["geometry"]["coordinates"]
            records.append({
                "usgs_id": f["id"],
                "datetime_utc": pd.to_datetime(p["time"], unit="ms", utc=True).tz_localize(None),
                "lat": g[1],
                "lon": g[0],
                "depth_km": g[2],
                "magnitude": p.get("mag"),
                "mag_type": p.get("magType"),
                "place": p.get("place"),
                "status": p.get("status"),
            })

        return pd.DataFrame(records)

    except requests.exceptions.RequestException as e:
        print(f"    API hatası: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    Parse hatası: {e}")
        return pd.DataFrame()


# Öncelikli büyük depremler listesi
# Farklı fay tipleri, bölgeler ve dönemlerden seçildi
PRIORITY_EVENTS = [
    # Format: (event_id, isim, tarih, lat, lon, mw, fault_type)
    ("tohoku_2011",       "Tohoku Japonya 2011",          "2011-03-11", 38.297, 142.373, 9.1, "REVERSE"),
    ("sumatra_2004",      "Sumatra 2004",                  "2004-12-26",  3.295,  95.982, 9.1, "REVERSE"),
    ("chile_2010",        "Maule Şili 2010",               "2010-02-27",-35.846, -72.719, 8.8, "REVERSE"),
    ("alaska_1964",       "Alaska 1964",                   "1964-03-28", 61.05, -147.07,  9.2, "REVERSE"),
    ("chile_2015",        "Illapel Şili 2015",             "2015-09-16",-31.573, -71.674, 8.3, "REVERSE"),
    ("sumatra_2012",      "N.Sumatra 2012",                "2012-04-11",  2.327,  93.063, 8.6, "STRIKE_SLIP"),
    ("kahramanmaras_2023","Kahramanmaraş 2023",            "2023-02-06", 37.220,  37.019, 7.8, "STRIKE_SLIP"),
    ("izmit_1999",        "İzmit 1999",                    "1999-08-17", 40.748,  29.864, 7.6, "STRIKE_SLIP"),
    ("landers_1992",      "Landers ABD 1992",              "1992-06-28", 34.200,-116.437, 7.3, "STRIKE_SLIP"),
    ("ridgecrest_2019",   "Ridgecrest ABD 2019",           "2019-07-06", 35.770,-117.599, 7.1, "STRIKE_SLIP"),
    ("north_iran_1990",   "Kuzey İran 1990",               "1990-06-20", 36.957,  49.409, 7.4, "REVERSE"),
    ("bam_2003",          "Bam İran 2003",                 "2003-12-26", 29.010,  58.310, 6.6, "STRIKE_SLIP"),
    ("nias_2005",         "Nias Endonezya 2005",           "2005-03-28",  1.665,  97.004, 8.6, "REVERSE"),
    ("kuril_2006",        "Kuril 2006",                    "2006-11-15", 46.607, 153.266, 8.3, "REVERSE"),
    ("samoa_2009",        "Samoa 2009",                    "2009-09-29",-15.489,-172.095, 8.1, "NORMAL"),
    ("haiti_2010",        "Haiti 2010",                    "2010-01-12", 18.443, -72.571, 7.0, "STRIKE_SLIP"),
    ("nepal_2015",        "Nepal 2015",                    "2015-04-25", 28.231,  84.731, 7.8, "REVERSE"),
    ("new_zealand_2016",  "Kaikoura YZ 2016",              "2016-11-13",-42.757, 173.054, 7.8, "STRIKE_SLIP"),
    ("mexico_2017",       "Puebla Meksika 2017",           "2017-09-19", 18.584, -98.399, 7.1, "NORMAL"),
    ("palu_2018",         "Palu Endonezya 2018",           "2018-09-28", -0.256, 119.846, 7.5, "STRIKE_SLIP"),
]


def fetch_all_priority_events(output_dir="output/usgs_regional",
                               years_back=3,
                               radius_km=300,
                               min_magnitude=2.5,
                               delay_seconds=3):
    """Tüm öncelikli olaylar için USGS verisi çek"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    summary = []

    for event_id, name, date_str, lat, lon, mw, fault_type in PRIORITY_EVENTS:
        main_time = datetime.strptime(date_str, "%Y-%m-%d")
        start_time = main_time - timedelta(days=int(years_back * 365.25))
        end_time = main_time - timedelta(hours=1)  # Ana depremi dahil etme

        out_file = Path(output_dir) / f"{event_id}_precursor.csv"

        print(f"\n{'─'*60}")
        print(f"Çekiliyor: {name} ({mw} Mw, {date_str})")
        print(f"  Konum   : {lat:.3f}°N, {lon:.3f}°E")
        print(f"  Pencere : {start_time.date()} → {end_time.date()}")
        print(f"  Yarıçap : {radius_km} km")
        print(f"  Min mag : {min_magnitude}")

        if out_file.exists():
            existing = pd.read_csv(out_file)
            print(f"  Zaten var: {len(existing)} kayıt - atlanıyor")
            summary.append({
                "event_id": event_id,
                "name": name,
                "date": date_str,
                "mw": mw,
                "fault_type": fault_type,
                "lat": lat,
                "lon": lon,
                "n_precursors": len(existing),
                "status": "cached"
            })
            continue

        df = fetch_usgs(lat, lon, radius_km, start_time, end_time,
                        min_magnitude=min_magnitude)

        if df.empty:
            print(f"  Veri bulunamadı veya API hatası.")
            summary.append({
                "event_id": event_id,
                "name": name,
                "date": date_str,
                "mw": mw,
                "fault_type": fault_type,
                "lat": lat,
                "lon": lon,
                "n_precursors": 0,
                "status": "no_data"
            })
        else:
            df["parent_event_id"] = event_id
            df["parent_name"] = name
            df["parent_date"] = date_str
            df["parent_mw"] = mw
            df["parent_fault_type"] = fault_type
            df["parent_lat"] = lat
            df["parent_lon"] = lon
            df["days_before"] = (main_time - df["datetime_utc"]).dt.total_seconds() / 86400.0
            df["distance_km"] = haversine_km(lat, lon, df["lat"].values, df["lon"].values)

            df.to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"  Kaydedildi: {len(df)} kayıt → {out_file}")

            summary.append({
                "event_id": event_id,
                "name": name,
                "date": date_str,
                "mw": mw,
                "fault_type": fault_type,
                "lat": lat,
                "lon": lon,
                "n_precursors": len(df),
                "status": "fetched"
            })

        time.sleep(delay_seconds)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/fetch_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("USGS VERİ ÇEKME TAMAMLANDI")
    print(f"{'='*60}")
    print(summary_df[["name", "date", "mw", "fault_type", "n_precursors", "status"]].to_string(index=False))

    return summary_df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=300)
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--minmag", type=float, default=2.5)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--output", default="output/usgs_regional")
    args = ap.parse_args()

    fetch_all_priority_events(
        output_dir=args.output,
        years_back=args.years,
        radius_km=args.radius,
        min_magnitude=args.minmag,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()