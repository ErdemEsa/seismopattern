#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 3.1: NLP Deprem Raporu Tarama
=====================================================
Bilimsel raporlar ve deprem ozetlerinden oncu sinyal cikarir.

Katmanlar:
1. USGS Event Page metin analizi
2. Anahtar kelime tabanlı oncu sinyal tespiti
3. Tarihsel deprem raporlari veritabani
4. Yapilandirilmis cikti

Kullanim:
  python scripts/nlp_scanner.py --test
  python scripts/nlp_scanner.py --scan-usgs --lat 37.22 --lon 37.02
  python scripts/nlp_scanner.py --build-kb
"""

import re
import json
import math
import requests
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter


OUTPUT_DIR = Path("output/nlp")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# ONCU SINYAL SOZLUGU
# =========================================================

PRECURSOR_KEYWORDS = {
    "foreshock": {
        "terms": [
            "foreshock", "fore-shock", "foreshock sequence",
            "precursory earthquake", "precursory seismicity",
            "oncu deprem", "oncu sarsintilar",
        ],
        "weight": 0.9,
        "category": "seismic",
        "description": "Buyuk deprem oncesi kucuk depremler",
    },
    "quiescence": {
        "terms": [
            "seismic quiescence", "seismic gap", "seismic silence",
            "quiescence period", "quiet period", "lull in seismicity",
            "sismik sessizlik", "sismik bosluk",
        ],
        "weight": 0.8,
        "category": "seismic",
        "description": "Buyuk deprem oncesi sismik aktivite azalmasi",
    },
    "b_value": {
        "terms": [
            "b-value", "b value", "gutenberg-richter",
            "magnitude-frequency", "b-degeri",
            "low b-value", "decreasing b",
            "b value decrease", "b value anomaly",
        ],
        "weight": 0.85,
        "category": "seismic",
        "description": "b-degeri anomalisi (gerilim birikimi gostergesi)",
    },
    "migration": {
        "terms": [
            "seismicity migration", "migrating seismicity",
            "earthquake migration", "stress migration",
            "progressive failure", "rupture propagation",
            "sismik goc", "deprem gocu",
        ],
        "weight": 0.7,
        "category": "seismic",
        "description": "Depremlerin fay boyunca sistematik gocu",
    },
    "radon": {
        "terms": [
            "radon", "radon anomaly", "radon emission",
            "radon concentration", "groundwater radon",
            "radon gazi", "radon anomalisi",
        ],
        "weight": 0.5,
        "category": "geochemical",
        "description": "Yeraltindan radon gazi cikmasi (tartismali)",
    },
    "groundwater": {
        "terms": [
            "groundwater", "water level", "well water",
            "water table", "spring flow", "thermal spring",
            "geothermal", "hot spring anomaly",
            "yeralti suyu", "kaynak suyu", "sicak su",
        ],
        "weight": 0.5,
        "category": "geochemical",
        "description": "Yeralti suyu seviye/sicaklik degisimi",
    },
    "gps_anomaly": {
        "terms": [
            "gps anomaly", "geodetic anomaly", "crustal deformation",
            "strain anomaly", "transient deformation",
            "slow slip", "slow-slip event", "sse",
            "episodic tremor", "ets",
            "gps anomalisi", "kabuk deformasyonu",
        ],
        "weight": 0.75,
        "category": "geodetic",
        "description": "GPS ile tespit edilen anormal kabuk hareketi",
    },
    "ionosphere": {
        "terms": [
            "ionosphere", "ionospheric", "tec anomaly",
            "total electron content", "vtec",
            "electromagnetic", "em anomaly",
            "iyonosfer", "elektromanyetik",
        ],
        "weight": 0.4,
        "category": "atmospheric",
        "description": "Iyonosferik elektron yogunlugu anomalisi",
    },
    "stress_transfer": {
        "terms": [
            "stress transfer", "coulomb stress",
            "static stress", "stress triggering",
            "stress loading", "stress shadow",
            "gerilim transferi", "coulomb gerilimi",
        ],
        "weight": 0.7,
        "category": "mechanical",
        "description": "Komsu faylara gerilim aktarimi",
    },
    "accelerating_moment": {
        "terms": [
            "accelerating moment", "accelerating seismicity",
            "amr", "moment release", "benioff strain",
            "increasing rate", "seismicity rate increase",
            "hizlanan sismik aktivite",
        ],
        "weight": 0.75,
        "category": "seismic",
        "description": "Hizlanan sismik enerji salimi",
    },
    "animal_behavior": {
        "terms": [
            "animal behavior", "animal behaviour",
            "unusual animal", "animal anomaly",
            "hayvan davranisi",
        ],
        "weight": 0.2,
        "category": "anecdotal",
        "description": "Hayvan davranis degisikligi (bilimsel degeri dusuk)",
    },
}

# Negatif anahtar kelimeler (yanlis pozitif onleme)
NEGATIVE_TERMS = [
    "no foreshock", "no precursor", "without precursor",
    "lack of precursor", "absence of",
    "not observed", "no evidence",
    "aftershock", "post-seismic", "coseismic",
]


# =========================================================
# METIN ANALIZI
# =========================================================

def analyze_text(text, source="unknown"):
    """
    Bir metin icindeki oncu sinyal referanslarini bul.

    Returns:
        dict: bulunan sinyaller, skorlar, detaylar
    """
    if not text or len(text) < 10:
        return {"signals": [], "score": 0, "n_signals": 0}

    text_lower = text.lower()

    # Negatif terim kontrolu
    has_negative = any(neg in text_lower for neg in NEGATIVE_TERMS)

    signals = []
    categories = set()
    total_weight = 0

    for signal_id, info in PRECURSOR_KEYWORDS.items():
        for term in info["terms"]:
            if term.lower() in text_lower:
                # Terimin gecme sayisi
                count = text_lower.count(term.lower())

                # Baglamsal kontrol: terim yaninda negatif var mi?
                is_negated = False
                for neg in NEGATIVE_TERMS:
                    # Terimden 50 karakter oncesinde negatif var mi?
                    idx = text_lower.find(term.lower())
                    if idx > 0:
                        context = text_lower[max(0, idx-60):idx]
                        if any(n in context for n in ["no ", "not ", "without ", "lack of "]):
                            is_negated = True
                            break

                if is_negated:
                    continue

                weight = info["weight"]
                if has_negative:
                    weight *= 0.5

                signals.append({
                    "signal_id": signal_id,
                    "term": term,
                    "count": count,
                    "weight": weight,
                    "category": info["category"],
                    "description": info["description"],
                })

                categories.add(info["category"])
                total_weight += weight
                break  # Ayni sinyal icin birden fazla terim sayma

    # Skor hesapla (0-1 arasi)
    max_possible = sum(v["weight"] for v in PRECURSOR_KEYWORDS.values())
    score = min(1.0, total_weight / max_possible) if max_possible > 0 else 0

    return {
        "signals": signals,
        "score": round(score, 4),
        "n_signals": len(signals),
        "n_categories": len(categories),
        "categories": list(categories),
        "total_weight": round(total_weight, 3),
        "source": source,
        "has_negative_context": has_negative,
    }


# =========================================================
# USGS OLAY SAYFASI TARAMA
# =========================================================

def fetch_usgs_event_text(event_id):
    """USGS olay sayfasindan metin cek."""
    url = f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}/executive"
    try:
        resp = requests.get(url, timeout=30,
                           headers={"User-Agent": "SeismoPattern/1.0"})
        if resp.status_code == 200:
            # HTML'den metin cikar (basit)
            text = resp.text
            # Script ve style tag'lerini kaldir
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return None
    except Exception:
        return None


def fetch_usgs_nearby_events(lat, lon, radius_km=300,
                              min_mag=5.0, days_back=1095):
    """USGS'den yakin buyuk olaylari cek."""
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": lat, "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "orderby": "magnitude",
        "limit": 20,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        events = []
        for f in data.get("features", []):
            p = f["properties"]
            events.append({
                "id": f["id"],
                "title": p.get("title", ""),
                "mag": p.get("mag"),
                "time": datetime.utcfromtimestamp(p["time"]/1000),
                "place": p.get("place", ""),
                "url": p.get("url", ""),
            })
        return events
    except Exception as e:
        print(f"  USGS olay listesi hatasi: {e}")
        return []


def scan_usgs_region(lat, lon, radius_km=300, min_mag=5.0):
    """
    Bir bolgedeki USGS olay sayfalarini tara,
    oncu sinyal referanslarini bul.
    """
    print(f"  USGS bolge taramasi: {lat}, {lon} | {radius_km} km")

    events = fetch_usgs_nearby_events(lat, lon, radius_km, min_mag)
    print(f"  {len(events)} olay bulundu")

    results = []
    for ev in events[:10]:  # En buyuk 10
        print(f"    {ev['title']} (Mw {ev['mag']})...", end=" ")

        text = fetch_usgs_event_text(ev["id"])
        if text:
            analysis = analyze_text(text, source=f"USGS:{ev['id']}")
            analysis["event_id"] = ev["id"]
            analysis["event_title"] = ev["title"]
            analysis["event_mag"] = ev["mag"]
            analysis["event_time"] = str(ev["time"])
            results.append(analysis)

            if analysis["n_signals"] > 0:
                print(f"{analysis['n_signals']} sinyal bulundu!")
            else:
                print("sinyal yok")
        else:
            print("metin alinamadi")

        import time
        time.sleep(1)

    return results


# =========================================================
# TARIHSEL BILGI TABANI
# =========================================================

HISTORICAL_PRECURSORS = [
    {
        "event": "1999 Izmit (Mw 7.6)",
        "date": "1999-08-17",
        "lat": 40.75, "lon": 29.86,
        "precursors": [
            {"type": "foreshock", "detail": "Mw 5.1 foreshock 24 saat once (Yalova)"},
            {"type": "quiescence", "detail": "1997-1999 arasi bolgesel sismik sessizlik"},
            {"type": "gps_anomaly", "detail": "KAF uzerinde artan GPS hizlari"},
            {"type": "radon", "detail": "Kocaeli bolgesinde radon artisi raporlandi"},
        ],
        "source": "Baris et al. (2002), Parsons et al. (2000)",
    },
    {
        "event": "2023 Kahramanmaras (Mw 7.8)",
        "date": "2023-02-06",
        "lat": 37.22, "lon": 37.02,
        "precursors": [
            {"type": "quiescence", "detail": "DAF uzerinde uzun sureli sessizlik"},
            {"type": "stress_transfer", "detail": "1999 Izmit ve 2020 Elazig stres transferi"},
            {"type": "gps_anomaly", "detail": "Hatay-Maras segmentinde strain birikimi"},
        ],
        "source": "Melgar et al. (2023), Gallovič et al. (2023)",
    },
    {
        "event": "2011 Tohoku (Mw 9.1)",
        "date": "2011-03-11",
        "lat": 38.30, "lon": 142.37,
        "precursors": [
            {"type": "foreshock", "detail": "Mw 7.3 foreshock 2 gun once (9 Mart)"},
            {"type": "gps_anomaly", "detail": "Miyagi segmentinde slow slip (2008-2011)"},
            {"type": "b_value", "detail": "b-degeri dususu (Nanjo et al. 2012)"},
            {"type": "ionosphere", "detail": "TEC anomalisi 1-5 gun once (Heki 2011)"},
            {"type": "stress_transfer", "detail": "2008 Iwate-Miyagi (Mw 6.9) stres transferi"},
        ],
        "source": "Kato et al. (2012), Nanjo et al. (2012), Heki (2011)",
    },
    {
        "event": "2004 Sumatra (Mw 9.1)",
        "date": "2004-12-26",
        "lat": 3.30, "lon": 95.98,
        "precursors": [
            {"type": "quiescence", "detail": "2002-2004 arasi bolgesel sessizlik"},
            {"type": "b_value", "detail": "b-degeri anomalisi (Nuannin et al. 2005)"},
        ],
        "source": "Nuannin et al. (2005)",
    },
    {
        "event": "2010 Maule (Mw 8.8)",
        "date": "2010-02-27",
        "lat": -35.85, "lon": -72.72,
        "precursors": [
            {"type": "quiescence", "detail": "Uzun sureli sismik bosluk (Ruegg et al.)"},
            {"type": "gps_anomaly", "detail": "GPS coupling orani > 0.8"},
            {"type": "b_value", "detail": "b-degeri belirgin dusus"},
        ],
        "source": "Ruegg et al. (2009), Moreno et al. (2010)",
    },
    {
        "event": "1992 Landers (Mw 7.3)",
        "date": "1992-06-28",
        "lat": 34.20, "lon": -116.44,
        "precursors": [
            {"type": "foreshock", "detail": "Joshua Tree Mw 6.1 foreshock (Nisan 1992)"},
            {"type": "accelerating_moment", "detail": "Hizlanan moment salimi (Bowman et al.)"},
            {"type": "stress_transfer", "detail": "1989 Loma Prieta stres etkisi"},
        ],
        "source": "Hauksson et al. (1993), Bowman et al. (1998)",
    },
    {
        "event": "2015 Nepal (Mw 7.8)",
        "date": "2015-04-25",
        "lat": 28.23, "lon": 84.73,
        "precursors": [
            {"type": "gps_anomaly", "detail": "MHT uzerinde guclu coupling (Ader et al.)"},
            {"type": "quiescence", "detail": "1934'ten beri buyuk deprem olmamis"},
            {"type": "stress_transfer", "detail": "1505 ve 1934 kirilma bosluklari"},
        ],
        "source": "Ader et al. (2012), Avouac et al. (2015)",
    },
]


def search_historical_precursors(lat, lon, max_dist_km=500):
    """Tarihsel bilgi tabaninda yakin olaylari ara."""
    results = []
    for entry in HISTORICAL_PRECURSORS:
        d = haversine_km(lat, lon, entry["lat"], entry["lon"])
        if d <= max_dist_km:
            results.append({
                "event": entry["event"],
                "date": entry["date"],
                "distance_km": round(d, 1),
                "precursors": entry["precursors"],
                "source": entry["source"],
                "n_precursors": len(entry["precursors"]),
                "precursor_types": list(set(
                    p["type"] for p in entry["precursors"]
                )),
            })
    results.sort(key=lambda x: x["distance_km"])
    return results


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# =========================================================
# BIRLESIK TARAMA
# =========================================================

def full_scan(lat, lon, radius_km=300):
    """
    Bir bolge icin tam NLP taramasi:
    1. Tarihsel bilgi tabani
    2. USGS olay sayfalari (opsiyonel)
    """
    result = {
        "lat": lat, "lon": lon,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # 1. Tarihsel bilgi tabani
    print(f"\n  Tarihsel bilgi tabani taramasi...")
    hist = search_historical_precursors(lat, lon, max_dist_km=500)
    result["historical"] = hist
    print(f"    {len(hist)} tarihsel kayit bulundu")

    for h in hist:
        print(f"    - {h['event']} ({h['distance_km']} km)")
        for p in h["precursors"]:
            print(f"      [{p['type']}] {p['detail']}")

    # 2. USGS taramasi
    print(f"\n  USGS olay sayfasi taramasi...")
    usgs = scan_usgs_region(lat, lon, radius_km, min_mag=5.0)
    result["usgs_scan"] = usgs

    usgs_with_signals = [u for u in usgs if u["n_signals"] > 0]
    print(f"    {len(usgs_with_signals)}/{len(usgs)} olayda sinyal bulundu")

    # 3. Ozet
    all_types = set()
    for h in hist:
        for p in h["precursors"]:
            all_types.add(p["type"])
    for u in usgs_with_signals:
        for s in u["signals"]:
            all_types.add(s["signal_id"])

    result["summary"] = {
        "n_historical": len(hist),
        "n_usgs_scanned": len(usgs),
        "n_usgs_with_signals": len(usgs_with_signals),
        "all_precursor_types": list(all_types),
        "n_unique_types": len(all_types),
    }

    return result


def get_nlp_info_for_app(lat, lon):
    """app.py'den cagirilacak fonksiyon."""
    result = {
        "historical": search_historical_precursors(lat, lon, 500),
    }

    n_hist = len(result["historical"])
    all_types = set()
    for h in result["historical"]:
        for p in h["precursors"]:
            all_types.add(p["type"])

    result["summary"] = {
        "n_historical": n_hist,
        "precursor_types": list(all_types),
        "n_types": len(all_types),
    }

    return result


# =========================================================
# BILGI TABANI OLUSTURUCU
# =========================================================

def build_knowledge_base():
    """Tum tarihsel oncu sinyal verisini CSV olarak kaydet."""
    rows = []
    for entry in HISTORICAL_PRECURSORS:
        for p in entry["precursors"]:
            rows.append({
                "event": entry["event"],
                "date": entry["date"],
                "lat": entry["lat"],
                "lon": entry["lon"],
                "precursor_type": p["type"],
                "detail": p["detail"],
                "source": entry["source"],
            })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "precursor_knowledge_base.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Bilgi tabani kaydedildi: {out_path}")
    print(f"  {len(df)} kayit, {len(HISTORICAL_PRECURSORS)} olay")

    # Oncu tip dagilimi
    print("\nOncu sinyal tipi dagilimi:")
    for t, n in df["precursor_type"].value_counts().items():
        info = PRECURSOR_KEYWORDS.get(t, {})
        w = info.get("weight", "?")
        print(f"  {t:<25} {n:>3} kayit  (agirlik: {w})")

    return df


# =========================================================
# TEST
# =========================================================

def test():
    print("=" * 60)
    print("NLP ONCU SINYAL TARAMA TESTI")
    print("=" * 60)

    # 1. Metin analizi testi
    print("\n--- Metin Analizi Testi ---")
    test_texts = [
        (
            "A sequence of foreshocks was observed before the mainshock. "
            "The b-value showed a significant decrease in the months "
            "leading to the earthquake. GPS measurements indicated "
            "anomalous crustal deformation.",
            "Pozitif metin"
        ),
        (
            "No foreshock activity was detected prior to the event. "
            "The earthquake occurred without any precursory signals.",
            "Negatif metin"
        ),
        (
            "Radon gas emissions were measured at several stations. "
            "Ionospheric TEC anomalies were reported 3 days before "
            "the earthquake. Slow slip events preceded the mainshock.",
            "Karisik metin"
        ),
    ]

    for text, label in test_texts:
        result = analyze_text(text, source="test")
        print(f"\n  [{label}]")
        print(f"  Skor: {result['score']}")
        print(f"  Sinyal: {result['n_signals']}")
        for s in result["signals"]:
            print(f"    - {s['signal_id']} ({s['category']}, "
                  f"w={s['weight']})")

    # 2. Tarihsel bilgi tabani
    print("\n\n--- Tarihsel Bilgi Tabani ---")
    locations = [
        ("Istanbul",      41.01,  28.98),
        ("Kahramanmaras", 37.22,  37.02),
        ("Tohoku",        38.30, 142.37),
        ("Cascadia",      47.61,-122.33),
    ]

    for name, lat, lon in locations:
        hist = search_historical_precursors(lat, lon)
        print(f"\n  {name}: {len(hist)} tarihsel kayit")
        for h in hist:
            types = ", ".join(h["precursor_types"])
            print(f"    {h['event']} ({h['distance_km']} km) "
                  f"[{types}]")

    # 3. Bilgi tabani olustur
    print("\n\n--- Bilgi Tabani ---")
    build_knowledge_base()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--scan-usgs", action="store_true")
    ap.add_argument("--build-kb", action="store_true")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    args = ap.parse_args()

    if args.test:
        test()
    elif args.scan_usgs and args.lat and args.lon:
        result = full_scan(args.lat, args.lon)
        out = OUTPUT_DIR / "scan_result.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nKaydedildi: {out}")
    elif args.build_kb:
        build_knowledge_base()
    else:
        print("Kullanim:")
        print("  python scripts/nlp_scanner.py --test")
        print("  python scripts/nlp_scanner.py --scan-usgs --lat 41.01 --lon 28.98")
        print("  python scripts/nlp_scanner.py --build-kb")


if __name__ == "__main__":
    main()