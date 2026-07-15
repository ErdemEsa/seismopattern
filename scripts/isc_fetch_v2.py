#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - ISC Katalog Çekici v2
ISC API URL ve parametre düzeltmesi + Mc otomasyonu
"""

import time
import requests
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta, timezone

OUTPUT_DIR = Path("output/isc_regional")
OUTPUT_DIR.mkdir(exist_ok=True)

# ISC'nin doğru endpoint'i
ISC_URLS = [
    "https://isc-mirror.iris.washington.edu/fdsnws/event/1/query",
    "http://www.isc.ac.uk/fdsnws/event/1/query",
]

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fetch_isc_v2(lat, lon, radius_km, start_time, end_time,
                 min_mag=2.5, timeout=90):
    """
    ISC FDSN - düzeltilmiş parametreler.
    maxradiuskm parametresi km cinsinden (bazı sürümler destekler).
    Desteklemiyorsa maxradius (derece) kullan.
    """
    radius_deg = radius_km / 111.195

    for base_url in ISC_URLS:
        # Önce km cinsinden dene
        for radius_param, radius_val in [
            ("maxradiuskm", radius_km),
            ("maxradius",   radius_deg),
        ]:
            params = {
                "format":       "text",
                "starttime":    start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "endtime":      end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "latitude":     round(lat, 4),
                "longitude":    round(lon, 4),
                radius_param:   round(radius_val, 4),
                "minmagnitude": min_mag,
                "orderby":      "time-asc",
            }

            try:
                print(f"    ISC deneniyor: {base_url.split('/')[2]} "
                      f"({radius_param}={radius_val:.1f})")
                resp = requests.get(base_url, params=params,
                                    timeout=timeout)
                resp.raise_for_status()

                if "EventID" in resp.text or "Time" in resp.text:
                    df = parse_isc_text(resp.text)
                    if df is not None and len(df) > 0:
                        print(f"    ISC başarılı: {len(df)} olay")
                        return df, None
                else:
                    # Boş yanıt
                    continue

            except requests.exceptions.Timeout:
                print(f"    Zaman aşımı ({base_url.split('/')[2]})")
                continue
            except requests.exceptions.ConnectionError:
                print(f"    Bağlantı hatası ({base_url.split('/')[2]})")
                continue
            except Exception as e:
                print(f"    Hata: {str(e)[:60]}")
                continue

    return None, "ISC erişilemez"


def parse_isc_text(text):
    """ISC pipe-delimited text formatını DataFrame'e çevir"""
    lines = text.strip().split("\n")
    records = []
    header = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Yorum satırları
        if line.startswith("#"):
            # Header tespiti
            if "|" in line and "EventID" in line:
                header = [h.strip().lower()
                          for h in line.lstrip("#").split("|")]
            continue

        if "|" not in line:
            continue

        # Header satırı (# olmadan)
        if "EventID" in line or "eventid" in line.lower():
            header = [h.strip().lower() for h in line.split("|")]
            continue

        parts = [p.strip() for p in line.split("|")]

        try:
            # Genel parse
            if header and len(parts) >= 5:
                row = dict(zip(header, parts))
                rec = {
                    "event_id": row.get("eventid", ""),
                    "time":     pd.to_datetime(row.get("time", ""),
                                               errors="coerce"),
                    "lat":      pd.to_numeric(row.get("latitude", ""),
                                              errors="coerce"),
                    "lon":      pd.to_numeric(row.get("longitude", ""),
                                              errors="coerce"),
                    "depth_km": pd.to_numeric(row.get("depth", ""),
                                              errors="coerce"),
                    "magnitude":pd.to_numeric(
                        row.get("magnitude", row.get("mag", "")),
                        errors="coerce"
                    ),
                    "source":   "ISC",
                }
            elif len(parts) >= 10:
                # Pozisyon bazlı fallback
                rec = {
                    "event_id": parts[0],
                    "time":     pd.to_datetime(parts[1], errors="coerce"),
                    "lat":      pd.to_numeric(parts[2], errors="coerce"),
                    "lon":      pd.to_numeric(parts[3], errors="coerce"),
                    "depth_km": pd.to_numeric(parts[4], errors="coerce"),
                    "magnitude":pd.to_numeric(parts[10], errors="coerce")
                                if len(parts) > 10 else np.nan,
                    "source":   "ISC",
                }
            else:
                continue

            if pd.notna(rec["time"]) and pd.notna(rec["magnitude"]):
                records.append(rec)

        except Exception:
            continue

    return pd.DataFrame(records) if records else None


def fetch_usgs(lat, lon, radius_km, start_time, end_time,
               min_mag=2.5, timeout=60):
    """USGS FDSN veri çekici"""
    params = {
        "format":       "geojson",
        "starttime":    start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime":      end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude":     lat,
        "longitude":    lon,
        "maxradiuskm":  radius_km,
        "minmagnitude": min_mag,
        "orderby":      "time-asc",
        "limit":        10000,
    }
    try:
        resp = requests.get(USGS_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        if not features:
            return pd.DataFrame()
        records = []
        for f in features:
            p, g = f["properties"], f["geometry"]["coordinates"]
            records.append({
                "event_id":  f["id"],
                "time":      pd.to_datetime(
                    p["time"], unit="ms", utc=True
                ).tz_localize(None),
                "lat":       g[1], "lon": g[0],
                "depth_km":  g[2],
                "magnitude": p.get("mag"),
                "mag_type":  p.get("magType", ""),
                "source":    "USGS",
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"    USGS hatası: {str(e)[:60]}")
        return pd.DataFrame()


def deduplicate(df, time_tol=120, dist_tol=80):
    """Zaman ve mekân bazlı duplikat temizle"""
    if len(df) == 0:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    keep = [True] * len(df)
    for i in range(1, len(df)):
        if not keep[i]:
            continue
        for j in range(i-1, max(0, i-30), -1):
            if not keep[j]:
                continue
            dt = abs((df.loc[i, "time"] -
                      df.loc[j, "time"]).total_seconds())
            if dt > time_tol:
                break
            dlat = df.loc[i, "lat"] - df.loc[j, "lat"]
            dlon = df.loc[i, "lon"] - df.loc[j, "lon"]
            dist = (dlat**2 + dlon**2)**0.5 * 111
            if dist < dist_tol:
                keep[i if df.loc[j,"source"]=="ISC" else j] = False
                break
    return df[keep].reset_index(drop=True)


def estimate_mc(mags, dm=0.1):
    """
    Gelişmiş Mc tahmini:
    1) MaxC
    2) Enerji bazlı kontrol
    3) Sonuç: muhafazakar seç
    """
    mags = np.array(mags)
    mags = mags[np.isfinite(mags)]
    if len(mags) < 10:
        return float(np.nanmin(mags)) if len(mags) > 0 else 3.0

    bins = np.arange(
        np.floor(mags.min()*10)/10,
        np.ceil(mags.max()*10)/10 + dm, dm
    )
    counts, edges = np.histogram(mags, bins=bins)
    if len(counts) == 0:
        return mags.min()

    mc_maxc = edges[np.argmax(counts)] + dm/2

    # Muhafazakar: MaxC + 0.2 düzeltmesi (Woessner & Wiemer 2005)
    mc_conservative = mc_maxc + 0.2

    # Veri yeterliyse düşür
    above = mags[mags >= mc_maxc]
    if len(above) >= 30:
        return round(mc_maxc, 1)
    else:
        return round(mc_conservative, 1)


def b_value_mle(mags, mc, dm=0.05):
    """MLE b-değeri + güven aralığı"""
    mags = np.array(mags)
    above = mags[np.isfinite(mags) & (mags >= mc)]
    n = len(above)
    if n < 20:
        return np.nan, np.nan, np.nan, n
    mean_m = np.mean(above)
    denom = mean_m - (mc - dm/2)
    if denom <= 0:
        return np.nan, np.nan, np.nan, n
    b = np.log10(np.e) / denom
    b_std = 2.3 * b**2 * np.std(above, ddof=1) / np.sqrt(n*(n-1))
    return round(b,4), round(b-1.96*b_std,4), round(b+1.96*b_std,4), n


def fetch_and_analyze(lat, lon, radius_km=300, min_mag=2.5,
                      ref_date=None, event_id=None,
                      use_cache=True):
    """
    Tam veri çekme + feature hesaplama pipeline.
    ISC öncelikli, USGS fallback.
    """
    if ref_date:
        end_time = datetime.strptime(ref_date, "%Y-%m-%d")
    else:
        end_time = now_utc()

    start_time = end_time - timedelta(days=1095)

    # Önbellek kontrolü
    cache_file = None
    if event_id and use_cache:
        cache_file = OUTPUT_DIR / f"{event_id}_v2.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            df["time"] = pd.to_datetime(df["time"])
            print(f"  Önbellekten: {len(df)} kayıt ({cache_file.name})")
            source = "cache"
        else:
            df = None
    else:
        df = None

    if df is None:
        # ISC dene
        isc_df, isc_err = fetch_isc_v2(
            lat, lon, radius_km, start_time, end_time, min_mag
        )

        # USGS her zaman çek (tamamlayıcı)
        print(f"    USGS çekiliyor...")
        usgs_df = fetch_usgs(
            lat, lon, radius_km, start_time, end_time, min_mag
        )
        print(f"    USGS: {len(usgs_df)} olay")

        # Birleştir
        frames = []
        if isc_df is not None and len(isc_df) > 0:
            frames.append(isc_df.dropna(subset=["time","magnitude"]))
        if len(usgs_df) > 0:
            frames.append(usgs_df.dropna(subset=["time","magnitude"]))

        if not frames:
            return None, None, "Veri bulunamadı"

        df = pd.concat(frames, ignore_index=True)
        df = deduplicate(df)
        source = ("ISC+USGS" if isc_df is not None and len(isc_df) > 0
                  else "USGS")

        if cache_file:
            df.to_csv(cache_file, index=False)
            print(f"  Önbelleğe kaydedildi: {cache_file.name}")

    n_total = len(df)
    if n_total == 0:
        return None, None, "Boş katalog"

    df["days"] = (end_time - df["time"]).dt.total_seconds() / 86400.0
    df = df[df["days"] >= 0].copy()

    # Mc hesapla
    mc = estimate_mc(df["magnitude"].values)
    df_mc = df[df["magnitude"] >= mc].copy()
    n_mc = len(df_mc)

    # Zaman pencereleri
    w1 = df_mc[df_mc["days"] <= 365]
    w2 = df_mc[(df_mc["days"] > 365) & (df_mc["days"] <= 730)]
    w3 = df_mc[df_mc["days"] > 730]
    c0, c1, c2 = len(w1), len(w2), len(w3)

    # b-değerleri
    b1, b1l, b1u, n1 = b_value_mle(w1["magnitude"].values, mc)
    b2, b2l, b2u, n2 = b_value_mle(w2["magnitude"].values, mc)
    b3v, b3l, b3u, n3 = b_value_mle(w3["magnitude"].values, mc)
    b_all, bal, bau, na = b_value_mle(df_mc["magnitude"].values, mc)

    # b trendi
    b_vals = [v for v in [b3v, b2, b1] if not np.isnan(v)]
    b_trend = (round(float(np.polyfit(range(len(b_vals)),
                                       b_vals, 1)[0]), 5)
               if len(b_vals) >= 2 else np.nan)
    b_decreasing = bool(b_trend < 0) if not np.isnan(b_trend) else None

    # Quiescence
    prev = (c1+c2)/2.0
    qr = c0/prev if prev > 0 else np.nan

    # Hızlanma
    n90  = len(w1[w1["days"] <= 90])
    n275 = len(w1[(w1["days"] > 90) & (w1["days"] <= 365)])
    accel = round((n90/90.0)/(n275/275.0), 3) if n275 > 0 else np.nan

    # Z-skorlar
    long_rate = n_mc / 3.0
    z_rate = (c0 - long_rate) / (long_rate**0.5) if long_rate > 0 else 0
    z_b = ((b_all - b1) / max(0.15, abs(b_all)*0.1)
           if not np.isnan(b_all) and not np.isnan(b1) else 0)

    # Mekansal
    import math
    def hav(a, b, c, d):
        a,b,c,d = map(math.radians,[a,b,c,d])
        dl,dn = c-a, d-b
        x = math.sin(dl/2)**2 + math.cos(a)*math.cos(c)*math.sin(dn/2)**2
        return 6371*2*math.asin(math.sqrt(x))

    dw1 = ([hav(lat,lon,r.lat,r.lon) for r in w1.itertuples()]
           if len(w1) > 0 else [100])
    dall= ([hav(lat,lon,r.lat,r.lon) for r in df_mc.itertuples()]
           if len(df_mc) > 0 else [130])

    d1_m = round(float(np.mean(dw1)), 1)
    d3_m = round(float(np.mean(dall)), 1)
    z_dist = (d3_m - d1_m) / max(d3_m*0.2, 10) if d3_m > 0 else 0

    # b kalitesi
    b_quality = ("HIGH" if n1 >= 50 else
                 "MEDIUM" if n1 >= 20 else "LOW")

    feats = {
        "count_0_1y":       c0,
        "count_1_2y":       c1,
        "count_2_3y":       c2,
        "quiescence_ratio": round(float(qr), 4) if not np.isnan(qr) else None,
        "accel_90d":        accel,
        "monthly_slope_36m":0,
        "w1_max_mw":  float(w1["magnitude"].max()) if len(w1) > 0 else None,
        "w1_mean_mw": float(w1["magnitude"].mean()) if len(w1) > 0 else None,
        "w1_std_mw":  float(w1["magnitude"].std()) if len(w1) > 1 else 0.5,
        "w3_max_mw":  float(df_mc["magnitude"].max()) if len(df_mc) > 0 else None,
        "w3_mean_mw": float(df_mc["magnitude"].mean()) if len(df_mc) > 0 else None,
        "w1_b_value": b1,   "w2_b_value": b2,   "w3_b_value": b3v,
        "b_all_3y":   b_all,"b_trend":   b_trend,"b_decreasing": b_decreasing,
        "w1_mean_dist_km":  d1_m,
        "w3_mean_dist_km":  d3_m,
        "w1_std_dist_km":   round(float(np.std(dw1)), 1) if len(dw1) > 1 else 50,
        "w1_mean_depth_km": float(w1["depth_km"].mean()) if len(w1) > 0 else None,
        "w3_mean_depth_km": float(df_mc["depth_km"].mean()) if len(df_mc) > 0 else None,
        "w1_std_depth_km":  float(w1["depth_km"].std()) if len(w1) > 1 else 10,
        "w1_n_events": c0, "w3_n_events": n_mc,
        "w1_migration_slope_km_day": 0,
        "w3_migration_slope_km_day": 0,
        "z_rate_1y":    round(float(z_rate), 4),
        "z_rate_3y":    0,
        "z_b_value_1y": round(float(z_b), 4),
        "z_b_value_3y": 0,
        "z_max_mw_1y":  0,
        "z_depth_1y":   0,
        "z_dist_1y":    round(float(z_dist), 4),
    }

    meta = {
        "n_total":    n_total,
        "n_above_mc": n_mc,
        "mc":         mc,
        "source":     source if "source" in dir() else "USGS",
        "b_quality":  b_quality,
        "b1_n":       n1,
        "period":     (f"{start_time:%Y-%m-%d} → "
                       f"{end_time:%Y-%m-%d}"),
    }

    return feats, meta, None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat",     type=float, required=True)
    ap.add_argument("--lon",     type=float, required=True)
    ap.add_argument("--radius",  type=float, default=300)
    ap.add_argument("--minmag",  type=float, default=2.5)
    ap.add_argument("--refdate", type=str,   default=None)
    ap.add_argument("--id",      type=str,   default=None,
                    help="Olay ID (önbellek için)")
    ap.add_argument("--no-cache",action="store_true")
    args = ap.parse_args()

    print(f"Analiz: {args.lat}°N, {args.lon}°E | "
          f"{args.radius}km | Mw{args.minmag}+")
    if args.refdate:
        print(f"Referans tarih: {args.refdate}")

    feats, meta, err = fetch_and_analyze(
        args.lat, args.lon, args.radius, args.minmag,
        args.refdate, args.id,
        use_cache=not args.no_cache
    )

    if err:
        print(f"HATA: {err}")
        return

    print(f"\nSonuç:")
    print(f"  Kaynak     : {meta['source']}")
    print(f"  n_total    : {meta['n_total']}")
    print(f"  Mc         : {meta['mc']}")
    print(f"  n(Mc+)     : {meta['n_above_mc']}")
    print(f"  b_quality  : {meta['b_quality']} (n={meta['b1_n']})")
    print(f"  b(1y)      : {feats.get('w1_b_value')}")
    print(f"  b(3y)      : {feats.get('b_all_3y')}")
    print(f"  b_trend    : {feats.get('b_trend')}")
    print(f"  b_azalan   : {feats.get('b_decreasing')}")
    print(f"  Quiescence : {feats.get('quiescence_ratio')}")
    print(f"  Hızlanma   : {feats.get('accel_90d')}")
    print(f"  z_rate_1y  : {feats.get('z_rate_1y')}")
    print(f"  z_b_1y     : {feats.get('z_b_value_1y')}")


if __name__ == "__main__":
    main()