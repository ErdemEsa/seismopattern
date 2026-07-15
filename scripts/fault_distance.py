#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 1.4: Fay Bazli Mesafe
=============================================
GEM Global Active Faults veritabanindan:
1. En yakin fayi bul
2. Fay hattina dik mesafe hesapla
3. Fay tipi ve kayma hizi bilgisi cek
4. Feature olarak modele ekle

Kullanim:
  python scripts/fault_distance.py --test
  python scripts/fault_distance.py --enrich
"""

import json
import math
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime


FAULT_FILE = Path("data/faults/gem_active_faults.geojson")
EARTH_R = 6371.0


# =========================================================
# GEOMETRI FONKSIYONLARI
# =========================================================

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(a))


def point_to_segment_distance_km(plat, plon, lat1, lon1, lat2, lon2):
    """
    Bir noktanin bir cizgi segmentine olan en kisa mesafesi (km).
    
    Yontem:
    1. Noktayi ve segmenti yaklasik kartezyen koordinata cevir
    2. Nokta projeksiyon formulu uygula
    3. Mesafeyi km'ye cevir
    """
    # Yaklasik kartezyen projeksiyon (orta enlem bazli)
    mid_lat = math.radians((plat + lat1 + lat2) / 3.0)
    cos_lat = math.cos(mid_lat)
    
    # Derece → km donusumu
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * cos_lat
    
    # Kartezyen koordinatlar
    px = (plon - lon1) * km_per_deg_lon
    py = (plat - lat1) * km_per_deg_lat
    
    ax = 0.0
    ay = 0.0
    bx = (lon2 - lon1) * km_per_deg_lon
    by = (lat2 - lat1) * km_per_deg_lat
    
    # Segment vektoru
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    
    if seg_len_sq < 1e-10:
        # Segment cok kisa, nokta mesafesi
        return math.sqrt(px * px + py * py)
    
    # Projeksiyon parametresi t
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    
    # En yakin nokta
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    
    dist = math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
    return dist


def point_to_polyline_distance_km(plat, plon, coords):
    """
    Bir noktanin bir polyline'a (fay hatti) olan en kisa mesafesi.
    coords: [[lon1,lat1], [lon2,lat2], ...] formatinda
    """
    min_dist = float('inf')
    
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i+1][0], coords[i+1][1]
        
        d = point_to_segment_distance_km(plat, plon, lat1, lon1, lat2, lon2)
        if d < min_dist:
            min_dist = d
    
    return min_dist


def fault_strike_at_nearest(plat, plon, coords):
    """
    En yakin segment icin fay dogrultusu (strike) hesapla.
    Derece cinsinden (0-360).
    """
    min_dist = float('inf')
    best_strike = np.nan
    
    mid_lat = math.radians(plat)
    cos_lat = math.cos(mid_lat)
    
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i][0], coords[i][1]
        lon2, lat2 = coords[i+1][0], coords[i+1][1]
        
        d = point_to_segment_distance_km(plat, plon, lat1, lon1, lat2, lon2)
        
        if d < min_dist:
            min_dist = d
            # Strike hesapla
            dx = (lon2 - lon1) * 111.320 * cos_lat
            dy = (lat2 - lat1) * 110.574
            strike = math.degrees(math.atan2(dx, dy)) % 360
            best_strike = round(strike, 1)
    
    return best_strike


# =========================================================
# GEM FAY VERITABANI YUKLEME
# =========================================================

class FaultDatabase:
    """GEM Global Active Faults veritabani yoneticisi."""
    
    def __init__(self, geojson_path=None):
        self.faults = []
        self.spatial_index = {}  # basit grid indeks
        self.grid_size = 2.0  # derece
        
        if geojson_path is None:
            geojson_path = FAULT_FILE
        
        if Path(geojson_path).exists():
            self.load(geojson_path)
    
    def load(self, path):
        """GeoJSON dosyasini yukle ve indeksle."""
        print(f"  Fay veritabani yukleniyor: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get("features", [])
        print(f"  Toplam fay segmenti: {len(features)}")
        
        for feat in features:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            
            geom_type = geom.get("type", "")
            coords = geom.get("coordinates", [])
            
            if not coords:
                continue
            
            # MultiLineString → LineString listesi
            if geom_type == "MultiLineString":
                lines = coords
            elif geom_type == "LineString":
                lines = [coords]
            else:
                continue
            
            for line in lines:
                if len(line) < 2:
                    continue
                
                # Fay bilgileri
                fault_info = {
                    "coords": line,
                    "name": props.get("fault_name", props.get("name", "Unknown")),
                    "slip_type": self._normalize_slip_type(
                        props.get("slip_type", props.get("rake", "Unknown"))
                    ),
                    "slip_rate": self._parse_slip_rate(props),
                    "dip": props.get("dip", props.get("average_dip")),
                    "rake": props.get("rake", props.get("average_rake")),
                    "length_km": self._calc_length(line),
                }
                
                self.faults.append(fault_info)
                
                # Spatial index: her segment hangi grid hucrelerinde?
                for coord in line:
                    gx = int(coord[0] / self.grid_size)
                    gy = int(coord[1] / self.grid_size)
                    key = (gx, gy)
                    if key not in self.spatial_index:
                        self.spatial_index[key] = []
                    if len(self.spatial_index[key]) == 0 or \
                       self.spatial_index[key][-1] is not fault_info:
                        self.spatial_index[key].append(fault_info)
        
        print(f"  Yuklenen fay: {len(self.faults)}")
        print(f"  Grid hucre: {len(self.spatial_index)}")
    
    def _normalize_slip_type(self, raw):
        """Fay tipini standartlastir."""
        if raw is None:
            return "Unknown"
        raw = str(raw).lower().strip()
        
        if any(k in raw for k in ["strike", "lateral", "transform"]):
            return "STRIKE_SLIP"
        elif any(k in raw for k in ["reverse", "thrust", "compression"]):
            return "REVERSE"
        elif any(k in raw for k in ["normal", "extension"]):
            return "NORMAL"
        elif "oblique" in raw:
            return "OBLIQUE"
        else:
            return "Unknown"
    
    def _parse_slip_rate(self, props):
        """Kayma hizini mm/yil olarak parse et."""
        for key in ["slip_rate", "net_slip_rate", "average_slip_rate",
                     "sr", "slip_rate_min"]:
            val = props.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None
    
    def _calc_length(self, coords):
        """Fay uzunlugu (km)."""
        total = 0
        for i in range(len(coords) - 1):
            total += haversine_km(
                coords[i][1], coords[i][0],
                coords[i+1][1], coords[i+1][0]
            )
        return round(total, 1)
    
    def find_nearest(self, lat, lon, max_dist_km=500, max_results=5):
        """
        Bir noktaya en yakin faylari bul.
        Spatial index kullanarak hizli arama.
        """
        if not self.faults:
            return []
        
        # Aday grid hucreleri (yakin hucreler)
        search_radius = int(max_dist_km / (self.grid_size * 111)) + 1
        gx = int(lon / self.grid_size)
        gy = int(lat / self.grid_size)
        
        candidates = set()
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                key = (gx + dx, gy + dy)
                if key in self.spatial_index:
                    for fault in self.spatial_index[key]:
                        candidates.add(id(fault))
        
        # Mesafe hesapla
        results = []
        seen_ids = set()
        
        for fault in self.faults:
            fid = id(fault)
            if fid not in candidates:
                continue
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            
            dist = point_to_polyline_distance_km(lat, lon, fault["coords"])
            
            if dist <= max_dist_km:
                strike = fault_strike_at_nearest(lat, lon, fault["coords"])
                results.append({
                    "distance_km": round(dist, 2),
                    "name": fault["name"],
                    "slip_type": fault["slip_type"],
                    "slip_rate": fault["slip_rate"],
                    "length_km": fault["length_km"],
                    "strike_deg": strike,
                    "dip": fault.get("dip"),
                    "rake": fault.get("rake"),
                })
        
        results.sort(key=lambda x: x["distance_km"])
        return results[:max_results]
    
    def compute_fault_features(self, lat, lon, max_dist_km=300):
        """
        Bir koordinat icin fay bazli feature'lar hesapla.
        Model icin kullanilacak.
        """
        nearest = self.find_nearest(lat, lon, max_dist_km=max_dist_km, max_results=10)
        
        if not nearest:
            return {
                "nearest_fault_dist_km": None,
                "nearest_fault_name": None,
                "nearest_fault_type": None,
                "nearest_fault_slip_rate": None,
                "nearest_fault_length_km": None,
                "nearest_fault_strike": None,
                "n_faults_within_100km": 0,
                "n_faults_within_200km": 0,
                "n_faults_within_300km": 0,
                "total_fault_length_300km": 0,
                "dominant_fault_type": None,
                "max_slip_rate_300km": None,
                "fault_complexity": 0,
            }
        
        closest = nearest[0]
        
        n_100 = sum(1 for f in nearest if f["distance_km"] <= 100)
        n_200 = sum(1 for f in nearest if f["distance_km"] <= 200)
        n_300 = len(nearest)
        
        total_length = sum(f["length_km"] for f in nearest)
        
        slip_rates = [f["slip_rate"] for f in nearest 
                      if f["slip_rate"] is not None]
        max_slip = max(slip_rates) if slip_rates else None
        
        # Baskin fay tipi
        type_counts = {}
        for f in nearest:
            ft = f["slip_type"]
            type_counts[ft] = type_counts.get(ft, 0) + 1
        dominant = max(type_counts, key=type_counts.get) if type_counts else None
        
        # Fay karmasikligi: farkli tip sayisi * fay sayisi
        n_types = len(set(f["slip_type"] for f in nearest if f["slip_type"] != "Unknown"))
        complexity = n_types * n_300
        
        return {
            "nearest_fault_dist_km": closest["distance_km"],
            "nearest_fault_name": closest["name"],
            "nearest_fault_type": closest["slip_type"],
            "nearest_fault_slip_rate": closest.get("slip_rate"),
            "nearest_fault_length_km": closest["length_km"],
            "nearest_fault_strike": closest.get("strike_deg"),
            "n_faults_within_100km": n_100,
            "n_faults_within_200km": n_200,
            "n_faults_within_300km": n_300,
            "total_fault_length_300km": round(total_length, 1),
            "dominant_fault_type": dominant,
            "max_slip_rate_300km": max_slip,
            "fault_complexity": complexity,
        }


# =========================================================
# GLOBAL FAY VERITABANI INSTANCE
# =========================================================

_FAULT_DB = None

def get_fault_db():
    """Singleton fault database."""
    global _FAULT_DB
    if _FAULT_DB is None:
        if FAULT_FILE.exists():
            _FAULT_DB = FaultDatabase(FAULT_FILE)
        else:
            print(f"  UYARI: Fay veritabani bulunamadi: {FAULT_FILE}")
            _FAULT_DB = FaultDatabase.__new__(FaultDatabase)
            _FAULT_DB.faults = []
            _FAULT_DB.spatial_index = {}
            _FAULT_DB.grid_size = 2.0
    return _FAULT_DB


# =========================================================
# TEST
# =========================================================

def test_locations():
    """Bilinen lokasyonlari test et."""
    print("=" * 65)
    print("FAY MESAFE TESTI")
    print("=" * 65)
    
    db = get_fault_db()
    
    if not db.faults:
        print("HATA: Fay veritabani bos!")
        return
    
    locations = [
        ("Istanbul (KAF yakini)",      41.01,  28.98),
        ("Kahramanmaras (DAF)",        37.22,  37.02),
        ("Izmit (KAF)",                40.75,  29.86),
        ("San Andreas (LA)",           34.05, -118.25),
        ("Tohoku (Japan Trench)",      38.30,  142.37),
        ("Cascadia (Seattle)",         47.61, -122.33),
        ("Nankai (Osaka)",             34.69,  135.50),
        ("Lima (Peru Trench)",        -12.05,  -77.04),
        ("Nepal (MBT/MCT)",           28.23,   84.73),
    ]
    
    for name, lat, lon in locations:
        print(f"\n{'--'*30}")
        print(f"  {name} ({lat}, {lon})")
        
        feats = db.compute_fault_features(lat, lon)
        
        nearest = db.find_nearest(lat, lon, max_dist_km=300, max_results=3)
        
        print(f"  En yakin fay   : {feats['nearest_fault_name']}")
        print(f"  Mesafe         : {feats['nearest_fault_dist_km']} km")
        print(f"  Fay tipi       : {feats['nearest_fault_type']}")
        print(f"  Kayma hizi     : {feats['nearest_fault_slip_rate']} mm/yil")
        print(f"  Fay uzunlugu   : {feats['nearest_fault_length_km']} km")
        print(f"  Strike         : {feats['nearest_fault_strike']} derece")
        print(f"  100km icinde   : {feats['n_faults_within_100km']} fay")
        print(f"  300km icinde   : {feats['n_faults_within_300km']} fay")
        print(f"  Toplam uzunluk : {feats['total_fault_length_300km']} km")
        print(f"  Baskin tip     : {feats['dominant_fault_type']}")
        print(f"  Karmasiklik    : {feats['fault_complexity']}")
        
        if nearest:
            print(f"\n  En yakin 3 fay:")
            for i, f in enumerate(nearest):
                print(f"    {i+1}. {f['name'][:40]:<40} "
                      f"{f['distance_km']:>6.1f} km  "
                      f"{f['slip_type']:<12}  "
                      f"{f['slip_rate'] or '?':>5} mm/y  "
                      f"{f['length_km']:>6.1f} km")


def enrich_catalog():
    """
    Mevcut buyuk deprem kataloguna fay bilgilerini ekle.
    Her Mw7+ deprem icin en yakin fay bilgisini hesapla.
    """
    print("=" * 65)
    print("KATALOG ZENGINLESTIRME (Fay bilgileri)")
    print("=" * 65)
    
    db = get_fault_db()
    if not db.faults:
        print("HATA: Fay veritabani bos!")
        return
    
    # Real scored dosyasini yukle
    scored_path = Path("output/real_risk_scored.csv")
    if not scored_path.exists():
        scored_path = Path("output/gcmt_precursor_features.csv")
    
    if not scored_path.exists():
        print(f"HATA: {scored_path} bulunamadi")
        return
    
    df = pd.read_csv(scored_path, low_memory=False)
    
    lat_col = next((c for c in ["main_lat","lat","centroid_lat","hypo_lat"]
                    if c in df.columns), None)
    lon_col = next((c for c in ["main_lon","lon","centroid_lon","hypo_lon"]
                    if c in df.columns), None)
    
    if not lat_col or not lon_col:
        print(f"HATA: Koordinat sutunlari bulunamadi")
        print(f"  Mevcut: {list(df.columns[:15])}")
        return
    
    print(f"  Kaynak : {scored_path}")
    print(f"  Kayit  : {len(df)}")
    print(f"  Lat    : {lat_col}")
    print(f"  Lon    : {lon_col}")
    
    # Her deprem icin fay bilgisi
    fault_data = []
    
    for i, row in df.iterrows():
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        
        if pd.isna(lat) or pd.isna(lon):
            fault_data.append({})
            continue
        
        feats = db.compute_fault_features(float(lat), float(lon))
        fault_data.append(feats)
        
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(df)} islendi...")
    
    fault_df = pd.DataFrame(fault_data)
    
    # Birlestir
    enriched = pd.concat([df.reset_index(drop=True),
                          fault_df.reset_index(drop=True)], axis=1)
    
    out_path = scored_path.parent / (scored_path.stem + "_with_faults.csv")
    enriched.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    print(f"\n  Kaydedildi: {out_path}")
    print(f"  Yeni sutunlar: {list(fault_df.columns)}")
    
    # Ozet istatistikler
    valid = fault_df.dropna(subset=["nearest_fault_dist_km"])
    print(f"\n  Fay bilgisi bulunan: {len(valid)}/{len(df)}")
    
    if len(valid) > 0:
        print(f"  En yakin fay mesafesi:")
        print(f"    Min    : {valid['nearest_fault_dist_km'].min():.1f} km")
        print(f"    Median : {valid['nearest_fault_dist_km'].median():.1f} km")
        print(f"    Max    : {valid['nearest_fault_dist_km'].max():.1f} km")
        
        if "nearest_fault_type" in valid.columns:
            print(f"\n  Fay tipi dagilimi:")
            for ft, cnt in valid["nearest_fault_type"].value_counts().items():
                print(f"    {ft:<15}: {cnt}")


# =========================================================
# APP.PY ENTEGRASYONU
# =========================================================

def get_fault_info_for_app(lat, lon):
    """
    app.py'den cagirilacak basit fonksiyon.
    Bir koordinat icin fay bilgisi dondurur.
    """
    db = get_fault_db()
    if not db.faults:
        return None
    
    feats = db.compute_fault_features(lat, lon)
    nearest = db.find_nearest(lat, lon, max_dist_km=200, max_results=3)
    
    return {
        "features": feats,
        "nearest_faults": nearest,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="Bilinen lokasyonlari test et")
    ap.add_argument("--enrich", action="store_true",
                    help="Katalogu fay bilgileriyle zenginlestir")
    ap.add_argument("--lat", type=float, help="Enlem")
    ap.add_argument("--lon", type=float, help="Boylam")
    args = ap.parse_args()
    
    if args.test:
        test_locations()
    elif args.enrich:
        enrich_catalog()
    elif args.lat and args.lon:
        db = get_fault_db()
        if not db.faults:
            print("HATA: Fay veritabani bos!")
            return
        
        feats = db.compute_fault_features(args.lat, args.lon)
        nearest = db.find_nearest(args.lat, args.lon, max_dist_km=300)
        
        print(f"\nKonum: {args.lat}, {args.lon}")
        for k, v in feats.items():
            print(f"  {k}: {v}")
        
        if nearest:
            print(f"\nEn yakin faylar:")
            for i, f in enumerate(nearest):
                print(f"  {i+1}. {f['name'][:45]:<45} "
                      f"{f['distance_km']:>6.1f} km  {f['slip_type']}")
    else:
        print("Kullanim:")
        print("  python scripts/fault_distance.py --test")
        print("  python scripts/fault_distance.py --enrich")
        print("  python scripts/fault_distance.py --lat 41.01 --lon 28.98")


if __name__ == "__main__":
    main()