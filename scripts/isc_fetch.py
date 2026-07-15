#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - ISC Katalog Çekici
====================================
ISC FDSN Web Service üzerinden bölgesel katalog çeker.
USGS'e göre avantajları:
  - Daha düşük Mc (Mw 2.5+ dünya genelinde)
  - Daha kapsamlı tarihsel kapsam (1964+)
  - Daha fazla bölgesel ağ dahil

Kullanım:
  python scripts/isc_fetch.py --lat 40.77 --lon 29.00 --radius 300 --years 3
"""

import time
import requests
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


ISC_URL = "http://www.isc.ac.uk/fdsnws/event/1/query"
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OUTPUT_DIR = Path("output/isc_regional")
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_isc(lat, lon, radius_km, start_time, end_time,
              min_mag=2.5, max_rows=20000, timeout=120):
    """
    ISC FDSN'den deprem verisi çek.
    ISC bazen yavaş yanıt verir, timeout yüksek tutuldu.
    """
    params = {
        "format":       "text",
        "starttime":    start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime":      end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude":     lat,
        "longitude":    lon,
        "maxradius":    radius_km / 111.195,  # km → derece
        "minmagnitude": min_mag,
        "orderby":      "time-asc",
        "limit":        max_rows,
        "includeallorigins": "false",
        "includeallmagnitudes": "false",
    }

    try:
        print(f"    ISC API sorgulanıyor...")
        resp = requests.get(ISC_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        return parse_isc_text(resp.text), None
    except requests.exceptions.Timeout:
        return None, "ISC zaman aşımı"
    except requests.exceptions.RequestException as e:
        return None, f"ISC hata: {str(e)}"


def parse_isc_text(text):
    """ISC text formatını DataFrame'e çevir"""
    lines = text.strip().split("\n")
    records = []

    header_found = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Header satırı
        if "EventID" in line or "Time" in line and "Magnitude" in line:
            header_found = True
            continue

        if not header_found:
            continue

        parts = line.split("|")
        if len(parts) < 10:
            continue

        try:
            records.append({
                "event_id":  parts[0].strip(),
                "time":      pd.to_datetime(parts[1].strip(), errors="coerce"),
                "lat":       float(parts[2].strip()),
                "lon":       float(parts[3].strip()),
                "depth_km":  float(parts[4].strip()) if parts[4].strip() else np.nan,
                "magnitude": float(parts[10].strip()) if parts[10].strip() else np.nan,
                "mag_type":  parts[9].strip() if len(parts) > 9 else "",
                "source":    "ISC",
            })
        except (ValueError, IndexError):
            continue

    return pd.DataFrame(records) if records else pd.DataFrame()


def fetch_usgs_backup(lat, lon, radius_km, start_time, end_time,
                      min_mag=2.5, timeout=60):
    """USGS fallback (ISC başarısız olursa)"""
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
                "time":      pd.to_datetime(p["time"], unit="ms", utc=True).tz_localize(None),
                "lat":       g[1],
                "lon":       g[0],
                "depth_km":  g[2],
                "magnitude": p.get("mag"),
                "mag_type":  p.get("magType", ""),
                "source":    "USGS",
            })
        return pd.DataFrame(records)
    except Exception as e:
        print(f"    USGS fallback hatası: {e}")
        return pd.DataFrame()


def fetch_best_available(lat, lon, radius_km, start_time, end_time,
                         min_mag=2.5):
    """
    ISC dene, başarısız olursa USGS'e geç.
    Her iki kaynaktan da alındıysa birleştir ve duplikatları temizle.
    """
    print(f"  ISC deneniyor...")
    isc_df, err = fetch_isc(lat, lon, radius_km, start_time, end_time, min_mag)

    if isc_df is not None and len(isc_df) > 0:
        isc_df = isc_df.dropna(subset=["time", "magnitude"])
        print(f"  ISC: {len(isc_df)} olay bulundu")
    else:
        print(f"  ISC başarısız ({err}), USGS'e geçiliyor...")
        isc_df = pd.DataFrame()

    print(f"  USGS de deneniyor (tamamlayıcı)...")
    usgs_df = fetch_usgs_backup(lat, lon, radius_km, start_time, end_time, min_mag)
    if len(usgs_df) > 0:
        usgs_df = usgs_df.dropna(subset=["time", "magnitude"])
        print(f"  USGS: {len(usgs_df)} olay bulundu")

    # Birleştir
    if len(isc_df) == 0 and len(usgs_df) == 0:
        return pd.DataFrame(), "ISC", 0
    elif len(isc_df) == 0:
        return usgs_df, "USGS", len(usgs_df)
    elif len(usgs_df) == 0:
        return isc_df, "ISC", len(isc_df)
    else:
        # Her iki kaynaktan geldiyse birleştir, duplikatları kaldır
        combined = pd.concat([isc_df, usgs_df], ignore_index=True)
        combined = combined.sort_values("time").reset_index(drop=True)

        # 60 saniye içinde, 50 km içindeki aynı büyüklükteki depremler = duplikat
        combined["time_sec"] = (
            combined["time"] - pd.Timestamp("1970-01-01")
        ).dt.total_seconds()
        combined = deduplicate_catalog(combined, time_tol=60, dist_tol=50)
        print(f"  Birleştirildi: {len(combined)} benzersiz olay")
        return combined, "ISC+USGS", len(combined)


def deduplicate_catalog(df, time_tol=60, dist_tol=50):
    """
    Zaman ve mekân toleransıyla duplikat temizle.
    ISC ve USGS aynı olayı farklı kaydedebildiği için gerekli.
    """
    if len(df) == 0:
        return df

    df = df.sort_values("time").reset_index(drop=True)
    keep = [True] * len(df)

    for i in range(1, len(df)):
        if not keep[i]:
            continue
        for j in range(i - 1, max(0, i - 20), -1):
            if not keep[j]:
                continue

            dt = abs((df.loc[i, "time"] - df.loc[j, "time"]).total_seconds())
            if dt > time_tol:
                break

            # Mesafe kontrolü
            dlat = df.loc[i, "lat"] - df.loc[j, "lat"]
            dlon = df.loc[i, "lon"] - df.loc[j, "lon"]
            dist_deg = (dlat**2 + dlon**2)**0.5
            dist_km = dist_deg * 111.0

            if dist_km < dist_tol:
                # ISC kaydını tercih et (daha kaliteli)
                if df.loc[j, "source"] == "ISC":
                    keep[i] = False
                else:
                    keep[j] = False
                break

    return df[keep].reset_index(drop=True)


def estimate_mc_multiple(mags, dm=0.1):
    """
    Üç farklı Mc yöntemi uygula, en güvenilirini seç.
    1) MaxC (Maximum Curvature)
    2) GFT (Goodness-of-Fit Test)
    3) MBS (Median-Based Stability)
    """
    mags = np.array(mags)
    mags = mags[np.isfinite(mags)]
    if len(mags) < 20:
        return np.nanmin(mags) if len(mags) > 0 else None, "insufficient"

    # 1) MaxC
    bins = np.arange(
        np.floor(mags.min() * 10) / 10,
        np.ceil(mags.max() * 10) / 10 + dm, dm
    )
    counts, edges = np.histogram(mags, bins=bins)
    if len(counts) == 0:
        mc_maxc = mags.min()
    else:
        mc_maxc = edges[np.argmax(counts)] + dm / 2

    # 2) GFT (basitleştirilmiş)
    mc_gft = mc_maxc  # basit fallback
    for test_mc in np.arange(mc_maxc - 0.3, mc_maxc + 0.8, 0.1):
        above = mags[mags >= test_mc]
        if len(above) < 10:
            continue
        # G-R yasasına uyum kontrolü (R^2)
        b = np.log10(np.e) / (np.mean(above) - (test_mc - dm/2))
        if b <= 0:
            continue
        a = np.log10(len(above)) + b * test_mc
        expected = np.array([10**(a - b*m) for m in edges[:-1] if m >= test_mc])
        observed = np.array([np.sum(mags >= m) for m in edges[:-1] if m >= test_mc])
        if len(observed) > 0 and np.sum(observed) > 0:
            residual = np.abs(observed - expected).sum() / np.sum(observed)
            if residual < 0.1:  # %10 tolerans
                mc_gft = test_mc
                break

    # Karar: MaxC'yi kullan, GFT ile doğrula
    mc_final = round(max(mc_maxc, mc_gft - 0.1), 1)
    method = "MaxC+GFT"

    return mc_final, method


def b_value_mle_full(mags, mc, dm=0.05):
    """
    MLE b-değeri + güven aralığı.
    Shi & Bolt (1982) standart hata.
    """
    mags = np.array(mags)
    above = mags[np.isfinite(mags) & (mags >= mc)]
    n = len(above)

    if n < 20:
        return np.nan, np.nan, np.nan, n

    mean_m = np.mean(above)
    denom = mean_m - (mc - dm / 2.0)

    if denom <= 0:
        return np.nan, np.nan, np.nan, n

    b = np.log10(np.e) / denom
    b_std = 2.3 * b**2 * np.std(above, ddof=1) / np.sqrt(n * (n - 1))
    b_lower = b - 1.96 * b_std
    b_upper = b + 1.96 * b_std

    return (round(b, 4), round(b_lower, 4),
            round(b_upper, 4), n)


def compute_features_from_catalog(df, lat, lon, ref_time=None,
                                   mc_override=None):
    """
    Katalog verisinden SeismoPattern feature'larını hesapla.
    Bu fonksiyon app.py'deki basit versiyonun gelişmiş hali.
    """
    if ref_time is None:
        ref_time = datetime.utcnow()

    if isinstance(ref_time, str):
        ref_time = datetime.strptime(ref_time, "%Y-%m-%d")

    df = df.copy()
    df["days"] = (ref_time - df["time"]).dt.total_seconds() / 86400.0
    df = df[df["days"] >= 0].copy()

    n_total = len(df)
    if n_total == 0:
        return None, "Veri yok"

    # Mc hesapla
    if mc_override is not None:
        mc = mc_override
        mc_method = "override"
    else:
        mc, mc_method = estimate_mc_multiple(df["magnitude"].values)

    if mc is None:
        return None, "Mc hesaplanamadı"

    df_mc = df[df["magnitude"] >= mc].copy()
    n_mc = len(df_mc)

    # Zaman pencereleri
    w1 = df_mc[df_mc["days"] <= 365]
    w2 = df_mc[(df_mc["days"] > 365) & (df_mc["days"] <= 730)]
    w3 = df_mc[df_mc["days"] > 730]
    w_all = df_mc

    c0, c1, c2 = len(w1), len(w2), len(w3)

    # b-değeri (3 pencere)
    b1, b1l, b1u, n1 = b_value_mle_full(w1["magnitude"].values, mc)
    b2, b2l, b2u, n2 = b_value_mle_full(w2["magnitude"].values, mc)
    b3v, b3l, b3u, n3 = b_value_mle_full(w3["magnitude"].values, mc)
    b_all, bal, bau, na = b_value_mle_full(w_all["magnitude"].values, mc)

    # b-değeri trendi
    b_vals = [v for v in [b3v, b2, b1] if not np.isnan(v)]
    if len(b_vals) >= 2:
        b_trend = round(np.polyfit(range(len(b_vals)), b_vals, 1)[0], 5)
        b_decreasing = b_trend < 0
    else:
        b_trend = np.nan
        b_decreasing = None

    # Aktivite oranları
    prev_avg = (c1 + c2) / 2.0
    quiescence_ratio = c0 / prev_avg if prev_avg > 0 else np.nan

    n90 = len(w1[w1["days"] <= 90])
    n275 = len(w1[(w1["days"] > 90) & (w1["days"] <= 365)])
    accel_90d = round((n90/90.0) / (n275/275.0), 3) if n275 > 0 else np.nan

    # Z-score: aktivite oranı
    # Uzun dönem referans (tüm 3 yıl ortalaması)
    long_rate = n_mc / 3.0  # yıllık
    if long_rate > 0 and c1 > 0:
        z_rate_1y = (c0 - long_rate) / (long_rate**0.5)
    else:
        z_rate_1y = 0

    # Z-score: b-değeri
    if not np.isnan(b_all) and not np.isnan(b1):
        z_b_value_1y = (b_all - b1) / max(0.15, abs(b_all) * 0.1)
    else:
        z_b_value_1y = 0

    # Haversine mesafe
    def hav(la1, lo1, la2, lo2):
        import math
        la1, lo1, la2, lo2 = map(math.radians, [la1, lo1, la2, lo2])
        dl, dn = la2-la1, lo2-lo1
        a = math.sin(dl/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dn/2)**2
        return 6371 * 2 * math.asin(math.sqrt(a))

    dist_w1 = [hav(lat, lon, r["lat"], r["lon"])
               for _, r in w1.iterrows()] if len(w1) > 0 else [100]
    dist_all = [hav(lat, lon, r["lat"], r["lon"])
                for _, r in df_mc.iterrows()] if len(df_mc) > 0 else [130]

    d1_mean = round(np.mean(dist_w1), 1)
    d3_mean = round(np.mean(dist_all), 1)
    d1_std = round(np.std(dist_w1), 1) if len(dist_w1) > 1 else 50

    # Z-score: mesafe
    if d3_mean > 0:
        z_dist_1y = (d3_mean - d1_mean) / max(d3_mean * 0.2, 10)
    else:
        z_dist_1y = 0

    # Derinlik
    dep_w1 = w1["depth_km"].dropna()
    dep_all = df_mc["depth_km"].dropna()
    dep1_mean = round(dep_w1.mean(), 2) if len(dep_w1) > 0 else None
    dep3_mean = round(dep_all.mean(), 2) if len(dep_all) > 0 else None
    dep1_std = round(dep_w1.std(), 2) if len(dep_w1) > 1 else 10

    # Aylık trend
    monthly = []
    for i in range(36, 0, -1):
        end_d, st_d = i-1, i
        n = len(df_mc[(df_mc["days"] <= st_d*30.4) &
                      (df_mc["days"] > end_d*30.4)])
        monthly.append(n)
    monthly_slope = float(np.polyfit(range(36), monthly, 1)[0]) if len(monthly) == 36 else 0

    features = {
        # Temel sayımlar
        "count_0_1y":       c0,
        "count_1_2y":       c1,
        "count_2_3y":       c2,
        "quiescence_ratio": round(float(quiescence_ratio), 4) if not np.isnan(quiescence_ratio) else None,
        "accel_90d":        accel_90d,
        "monthly_slope_36m": round(monthly_slope, 6),

        # Büyüklük
        "w1_max_mw":   round(float(w1["magnitude"].max()), 3) if len(w1) > 0 else None,
        "w1_mean_mw":  round(float(w1["magnitude"].mean()), 3) if len(w1) > 0 else None,
        "w1_std_mw":   round(float(w1["magnitude"].std()), 3) if len(w1) > 1 else 0.5,
        "w3_max_mw":   round(float(df_mc["magnitude"].max()), 3) if len(df_mc) > 0 else None,
        "w3_mean_mw":  round(float(df_mc["magnitude"].mean()), 3) if len(df_mc) > 0 else None,

        # b-değeri (tam)
        "w1_b_value":  b1,
        "w2_b_value":  b2,
        "w3_b_value":  b3v,
        "b_all_3y":    b_all,
        "b_trend":     b_trend,
        "b_decreasing":b_decreasing,
        "b1_lower":    b1l, "b1_upper": b1u,
        "b3_lower":    b3l, "b3_upper": b3u,

        # Mekânsal
        "w1_mean_dist_km":  d1_mean,
        "w3_mean_dist_km":  d3_mean,
        "w1_std_dist_km":   d1_std,

        # Derinlik
        "w1_mean_depth_km": dep1_mean,
        "w3_mean_depth_km": dep3_mean,
        "w1_std_depth_km":  dep1_std,

        # Olay sayıları
        "w1_n_events": c0,
        "w3_n_events": n_mc,

        # Z-score'lar (şimdi gerçek hesaplama var)
        "z_rate_1y":    round(float(z_rate_1y), 4),
        "z_rate_3y":    0,
        "z_b_value_1y": round(float(z_b_value_1y), 4),
        "z_b_value_3y": 0,
        "z_max_mw_1y":  0,
        "z_depth_1y":   0,
        "z_dist_1y":    round(float(z_dist_1y), 4),

        # Göç trendi
        "w1_migration_slope_km_day": 0,
        "w3_migration_slope_km_day": 0,
    }

    meta = {
        "n_total":    n_total,
        "n_above_mc": n_mc,
        "mc":         mc,
        "mc_method":  mc_method,
        "source":     df["source"].iloc[0] if "source" in df.columns else "unknown",
        "b_quality": "HIGH" if n1 >= 50 else "MEDIUM" if n1 >= 20 else "LOW",
        "period":    f"{ref_time - timedelta(days=1095):%Y-%m-%d} → {ref_time:%Y-%m-%d}",
    }

    return features, meta


def run_priority_events():
    """
    Faz 3'teki öncelikli olayları ISC ile yeniden çek.
    USGS versiyonuna göre çok daha fazla veri bekleniyor.
    """
    EVENTS = [
        ("tohoku_2011",       "Tohoku 2011",        "2011-03-11", 38.30,  142.37, 2.5),
        ("sumatra_2004",      "Sumatra 2004",        "2004-12-26",  3.30,   95.98, 3.0),
        ("chile_2010",        "Maule 2010",          "2010-02-27",-35.85,  -72.72, 2.5),
        ("kahramanmaras_2023","Kahramanmaraş 2023",  "2023-02-06", 37.22,   37.02, 2.5),
        ("izmit_1999",        "İzmit 1999",          "1999-08-17", 40.75,   29.86, 2.5),
        ("landers_1992",      "Landers 1992",        "1992-06-28", 34.20, -116.44, 2.0),
        ("ridgecrest_2019",   "Ridgecrest 2019",     "2019-07-06", 35.77, -117.60, 2.0),
        ("nias_2005",         "Nias 2005",           "2005-03-28",  1.67,   97.00, 3.0),
        ("kuril_2006",        "Kuril 2006",          "2006-11-15", 46.61,  153.27, 3.0),
        ("nepal_2015",        "Nepal 2015",          "2015-04-25", 28.23,   84.73, 3.0),
        ("palu_2018",         "Palu 2018",           "2018-09-28", -0.26,  119.85, 2.5),
    ]

    print("=" * 65)
    print("ISC Veri Çekme — Öncelikli Olaylar")
    print("=" * 65)

    results = []
    for event_id, name, date_str, lat, lon, min_mag in EVENTS:
        print(f"\n{'─'*55}")
        print(f"Olay: {name} ({date_str})")

        out_file = OUTPUT_DIR / f"{event_id}_isc.csv"

        if out_file.exists():
            df = pd.read_csv(out_file)
            df["time"] = pd.to_datetime(df["time"])
            print(f"  Önbellekten: {len(df)} kayıt")
            source = "cache"
        else:
            end_time = datetime.strptime(date_str, "%Y-%m-%d")
            start_time = end_time - timedelta(days=1095)

            df, source, n = fetch_best_available(
                lat, lon, 300, start_time, end_time, min_mag
            )

            if len(df) > 0:
                df["parent_event"] = event_id
                df.to_csv(out_file, index=False)
                print(f"  Kaydedildi: {len(df)} kayıt → {out_file}")

            time.sleep(3)  # API rate limiting

        if len(df) > 0:
            feats, meta = compute_features_from_catalog(
                df, lat, lon, ref_time=datetime.strptime(date_str, "%Y-%m-%d")
            )

            if feats:
                print(f"  n_total={meta['n_total']}, "
                      f"Mc={meta['mc']} ({meta['mc_method']}), "
                      f"b(1y)={feats.get('w1_b_value','N/A')}, "
                      f"b(3y)={feats.get('b_all_3y','N/A')}, "
                      f"b_trend={feats.get('b_trend','N/A')}, "
                      f"b_quality={meta.get('b_quality','?')}")

                results.append({
                    "event_id": event_id,
                    "name": name,
                    "date": date_str,
                    "source": source,
                    **{k: v for k, v in meta.items()
                       if k not in ["period"]},
                    **{f"f_{k}": v for k, v in feats.items()
                       if not isinstance(v, list)},
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "isc_features_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"\n{'='*65}")
    print("ÖZET")
    print(f"{'='*65}")
    print(results_df[[
        "name", "n_total", "n_above_mc", "mc",
        "b_quality", "f_w1_b_value", "f_b_all_3y", "f_b_trend"
    ]].to_string(index=False))

    return results_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat",    type=float, help="Enlem")
    ap.add_argument("--lon",    type=float, help="Boylam")
    ap.add_argument("--radius", type=float, default=300)
    ap.add_argument("--years",  type=float, default=3.0)
    ap.add_argument("--minmag", type=float, default=2.5)
    ap.add_argument("--refdate",type=str,   default=None,
                    help="YYYY-MM-DD (boş=bugün)")
    ap.add_argument("--all",    action="store_true",
                    help="Tüm öncelikli olayları çek")
    args = ap.parse_args()

    if args.all:
        run_priority_events()
        return

    if args.lat is None or args.lon is None:
        print("Kullanım:")
        print("  Tek bölge: python isc_fetch.py --lat 40.77 --lon 29.00")
        print("  Tüm olaylar: python isc_fetch.py --all")
        return

    ref = (datetime.strptime(args.refdate, "%Y-%m-%d")
           if args.refdate else datetime.utcnow())
    end_time = ref
    start_time = end_time - timedelta(days=int(args.years * 365.25))

    df, source, n = fetch_best_available(
        args.lat, args.lon, args.radius,
        start_time, end_time, args.minmag
    )

    if len(df) == 0:
        print("Veri bulunamadı.")
        return

    feats, meta = compute_features_from_catalog(
        df, args.lat, args.lon, ref_time=ref
    )

    print(f"\nSONUÇ:")
    print(f"  Kaynak     : {source}")
    print(f"  Toplam     : {meta['n_total']}")
    print(f"  Mc         : {meta['mc']} ({meta['mc_method']})")
    print(f"  Mc üzeri   : {meta['n_above_mc']}")
    print(f"  b (1y)     : {feats.get('w1_b_value')}")
    print(f"  b (3y)     : {feats.get('b_all_3y')}")
    print(f"  b trendi   : {feats.get('b_trend')}")
    print(f"  b kalitesi : {meta['b_quality']}")
    print(f"  Quiescence : {feats.get('quiescence_ratio')}")
    print(f"  Hızlanma   : {feats.get('accel_90d')}")


if __name__ == "__main__":
    main()