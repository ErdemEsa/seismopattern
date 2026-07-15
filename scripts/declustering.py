#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 1.2: Declustering (Artçı Deprem Filtreleme)
===================================================================
Gardner & Knopoff (1974) yöntemi ile artçı depremleri filtreler.

Artçı deprem neden sorun?
  Mw 7+ deprem → yüzlerce artçı → bir sonraki büyük depremin
  öncü penceresinde gürültü oluşturur → model yanılır

Bu script:
  1. all_earthquakes.csv üzerinde declustering uygular
  2. Temizlenmiş katalog oluşturur
  3. Önceki feature'ları temiz katalogla yeniden hesaplar
  4. Model performansını karşılaştırır
"""

import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime


# ═══════════════════════════════════════════════════
# GARDNER & KNOPOFF (1974) PARAMETRELERİ
# ═══════════════════════════════════════════════════

def gk_window(magnitude):
    """
    Gardner & Knopoff (1974) artçı pencere parametreleri.
    
    Bir ana depremden sonra bu süre ve mesafe içindeki
    daha küçük depremler artçı kabul edilir.
    
    Returns: (days, km)
    """
    if magnitude < 2.5:
        return 42, 19.5
    elif magnitude < 3.0:
        return 55, 22.5
    elif magnitude < 3.5:
        return 75, 26.0
    elif magnitude < 4.0:
        return 100, 30.0
    elif magnitude < 4.5:
        return 130, 35.0
    elif magnitude < 5.0:
        return 170, 40.0
    elif magnitude < 5.5:
        return 220, 47.0
    elif magnitude < 6.0:
        return 290, 54.0
    elif magnitude < 6.5:
        return 380, 61.0
    elif magnitude < 7.0:
        return 510, 70.0
    elif magnitude < 7.5:
        return 660, 81.0
    elif magnitude < 8.0:
        return 870, 94.0
    else:
        return 1100, 110.0


def uhrhammer_window(magnitude):
    """
    Uhrhammer (1986) — daha kısa pencereler.
    Daha agresif filtreleme.
    """
    t = np.exp(-3.024 + 0.804 * magnitude)   # gün
    d = np.exp(-1.024 + 0.804 * magnitude)    # km
    return max(1, t), max(1, d)


def haversine_km_vec(lat1, lon1, lat2, lon2):
    """Vektörize haversine"""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ═══════════════════════════════════════════════════
# DECLUSTERİNG ALGORİTMASI
# ═══════════════════════════════════════════════════

def decluster_gk(df, method="gk", verbose=True):
    """
    Gardner & Knopoff declustering.
    
    Algoritma:
    1. Olayları büyüklüğe göre azalan sırala
    2. En büyük olaydan başla
    3. Bu olayın GK penceresindeki tüm küçük olayları artçı say
    4. Artçıları işaretle, bir sonraki büyük olaya geç
    5. Zaten artçı işaretlenmiş olayları atla
    
    Parameters:
        df: DataFrame (datetime_utc, eff_lat, eff_lon, mw)
        method: "gk" veya "uhrhammer"
        
    Returns:
        df_main: sadece ana olaylar (artçılar çıkarılmış)
        df_after: sadece artçılar
    """
    df = df.copy()
    
    # Gerekli sütunları kontrol et
    time_col = None
    for c in ["datetime_utc", "time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("Zaman sütunu bulunamadı")
    
    lat_col = None
    for c in ["eff_lat", "centroid_lat", "lat", "hypo_lat"]:
        if c in df.columns:
            lat_col = c
            break
    
    lon_col = None
    for c in ["eff_lon", "centroid_lon", "lon", "hypo_lon"]:
        if c in df.columns:
            lon_col = c
            break
    
    mag_col = None
    for c in ["mw", "magnitude", "mag"]:
        if c in df.columns:
            mag_col = c
            break
    
    if any(x is None for x in [lat_col, lon_col, mag_col]):
        raise ValueError(f"Gerekli sütunlar bulunamadı. "
                        f"Mevcut: {list(df.columns[:15])}")
    
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, lat_col, lon_col, mag_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    n = len(df)
    is_aftershock = np.zeros(n, dtype=bool)
    parent_idx = np.full(n, -1, dtype=int)
    
    window_func = gk_window if method == "gk" else uhrhammer_window
    
    # Büyükten küçüğe sırala (indeks bazında)
    mag_order = df[mag_col].values.argsort()[::-1]
    
    if verbose:
        print(f"  Declustering başlıyor: {n:,} olay, yöntem={method}")
    
    processed = 0
    for rank, i in enumerate(mag_order):
        if is_aftershock[i]:
            continue
        
        mag_i = df.loc[i, mag_col]
        if pd.isna(mag_i):
            continue
        
        t_window, d_window = window_func(mag_i)
        time_i = df.loc[i, time_col]
        lat_i = df.loc[i, lat_col]
        lon_i = df.loc[i, lon_col]
        
        # Zaman penceresi içindeki olayları bul
        time_diff = (df[time_col] - time_i).dt.total_seconds() / 86400.0
        time_mask = (time_diff > 0) & (time_diff <= t_window)
        
        candidates = df.index[time_mask & ~is_aftershock]
        
        if len(candidates) == 0:
            continue
        
        # Mesafe kontrolü
        dists = haversine_km_vec(
            lat_i, lon_i,
            df.loc[candidates, lat_col].values,
            df.loc[candidates, lon_col].values
        )
        
        # Pencere içinde VE daha küçük olan → artçı
        for j, dist_j in zip(candidates, dists):
            if dist_j <= d_window:
                mag_j = df.loc[j, mag_col]
                if not pd.isna(mag_j) and mag_j <= mag_i:
                    is_aftershock[j] = True
                    parent_idx[j] = i
        
        processed += 1
        if verbose and processed % 5000 == 0:
            n_after = is_aftershock.sum()
            print(f"    {processed:,} ana olay işlendi, "
                  f"{n_after:,} artçı bulundu...")
    
    df["is_aftershock"] = is_aftershock
    df["parent_event_idx"] = parent_idx
    
    df_main = df[~is_aftershock].copy()
    df_after = df[is_aftershock].copy()
    
    n_after = is_aftershock.sum()
    pct = n_after / n * 100
    
    if verbose:
        print(f"\n  Sonuç:")
        print(f"    Toplam    : {n:,}")
        print(f"    Ana olay  : {len(df_main):,} ({100-pct:.1f}%)")
        print(f"    Artçı     : {n_after:,} ({pct:.1f}%)")
        
        # Büyüklük bazlı dağılım
        print(f"\n  Büyüklük bazlı artçı oranı:")
        for low, high in [(4.0,5.0),(5.0,5.5),(5.5,6.0),(6.0,6.5),
                          (6.5,7.0),(7.0,7.5),(7.5,10)]:
            mask = (df[mag_col] >= low) & (df[mag_col] < high)
            tot = mask.sum()
            aft = (mask & is_aftershock).sum()
            if tot > 0:
                print(f"    Mw {low:.1f}-{high:.1f}: "
                      f"{aft:>5}/{tot:>5} artçı ({aft/tot*100:.1f}%)")
    
    return df_main, df_after


# ═══════════════════════════════════════════════════
# ANA İŞLEM
# ═══════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Deprem kataloğu declustering"
    )
    ap.add_argument("--input", default="output/all_earthquakes.csv",
                    help="Girdi CSV dosyası")
    ap.add_argument("--output-main",
                    default="output/all_earthquakes_declustered.csv",
                    help="Ana olaylar çıktısı")
    ap.add_argument("--output-after",
                    default="output/aftershocks_removed.csv",
                    help="Çıkarılan artçılar")
    ap.add_argument("--method", default="gk",
                    choices=["gk", "uhrhammer"],
                    help="Declustering yöntemi")
    args = ap.parse_args()
    
    print("=" * 60)
    print("SeismoPattern - Declustering (Artçı Filtreleme)")
    print("=" * 60)
    
    # Veri yükle
    print(f"\nVeri yükleniyor: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    
    # Efektif koordinatlar
    if "eff_lat" not in df.columns:
        df["eff_lat"] = df.get("centroid_lat", 
                               df.get("hypo_lat", pd.Series()))
    if "eff_lon" not in df.columns:
        df["eff_lon"] = df.get("centroid_lon", 
                               df.get("hypo_lon", pd.Series()))
    
    # datetime
    time_col = None
    for c in ["datetime_utc", "time"]:
        if c in df.columns:
            time_col = c
            break
    
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    
    print(f"Toplam kayıt: {len(df):,}")
    
    # Büyüklük sütununu bul
    mag_col = "mw" if "mw" in df.columns else "magnitude"
    if mag_col in df.columns:
        valid = df[mag_col].notna()
        print(f"Mw mevcut: {valid.sum():,}")
        print(f"Mw aralığı: {df[mag_col].min():.2f} - {df[mag_col].max():.2f}")
    
    # Declustering uygula
    print(f"\nYöntem: {args.method.upper()}")
    print("─" * 50)
    
    df_main, df_after = decluster_gk(df, method=args.method)
    
    # Büyük deprem kontrolü
    if mag_col in df.columns:
        n_major_before = (df[mag_col] >= 7.0).sum()
        n_major_after = (df_main[mag_col] >= 7.0).sum()
        print(f"\n  Mw 7.0+ deprem sayısı:")
        print(f"    Öncesi : {n_major_before}")
        print(f"    Sonrası: {n_major_after}")
        print(f"    Fark   : {n_major_before - n_major_after} "
              f"(bu kadar Mw7+ olay artçı olarak işaretlendi)")
    
    # Kaydet
    print(f"\nKaydediliyor...")
    
    # is_aftershock ve parent_event_idx sütunlarını kaldır
    save_cols = [c for c in df_main.columns 
                 if c not in ["is_aftershock", "parent_event_idx"]]
    
    df_main[save_cols].to_csv(args.output_main, 
                               index=False, encoding="utf-8-sig")
    print(f"  Ana olaylar: {args.output_main} ({len(df_main):,} kayıt)")
    
    df_after.to_csv(args.output_after, 
                     index=False, encoding="utf-8-sig")
    print(f"  Artçılar   : {args.output_after} ({len(df_after):,} kayıt)")
    
    # Karşılaştırma raporu
    print(f"\n{'='*60}")
    print("KARŞILAŞTIRMA RAPORU")
    print(f"{'='*60}")
    
    print(f"\n  Orijinal katalog        : {len(df):,} olay")
    print(f"  Temizlenmiş katalog     : {len(df_main):,} olay")
    print(f"  Çıkarılan artçı        : {len(df_after):,} olay")
    print(f"  Azalma oranı           : %{len(df_after)/len(df)*100:.1f}")
    
    if mag_col in df.columns:
        # Yıllık oran karşılaştırması
        if time_col:
            years = (df[time_col].max() - df[time_col].min()).days / 365.25
            rate_before = len(df) / years
            rate_after = len(df_main) / years
            print(f"\n  Yıllık olay oranı:")
            print(f"    Öncesi : {rate_before:.1f}/yıl")
            print(f"    Sonrası: {rate_after:.1f}/yıl")
    
    print(f"\n  Sonraki adım:")
    print(f"    1. Temiz katalogla feature'ları yeniden hesapla")
    print(f"    2. Model performansını karşılaştır")
    print(f"    3. phase2_gcmt_features.py'yi temiz CSV ile çalıştır")
    
    print(f"\n  Komut:")
    print(f"    python scripts\\phase2_gcmt_features.py \\")
    print(f"      --input {args.output_main} \\")
    print(f"      --output output\\gcmt_precursor_features_declustered.csv")


if __name__ == "__main__":
    main()