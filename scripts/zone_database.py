#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Genisletilmis Bolge Veritabani
================================================
Mw 6.0+ deprem riski olan tum bolgeleri icerir.
Fay segment analizi + kirilma bosluklari.
"""

import json
import math
from pathlib import Path

ZONES_FILE = Path("data/zones_extended.json")

# JSON varsa oradan yukle, yoksa asagidaki hardcoded sozlugu kullan
def _load_zones_from_json():
    if ZONES_FILE.exists():
        try:
            data = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
            if len(data) >= 25:
                return data
        except Exception:
            pass
    return None

_json_zones = _load_zones_from_json()
if _json_zones is not None:
    EXTENDED_ZONES = _json_zones
else:
    # Hardcoded fallback (asagidaki orijinal sozluk)
    EXTENDED_ZONES = {
        # === TURKIYE ===
        "marmara": {
        "name": "Marmara (Istanbul Segmenti)",
        "lat": 40.77, "lon": 29.00,
        "last_major": "1999-08-17", "last_major_mw": 7.6,
        "recurrence_years": 250,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Kuzey Anadolu Fayi (KAF)",
        "expected_mw": "7.0-7.4",
        "population_risk": "~16 milyon",
        "segment_info": {
            "total_length_km": 150,
            "ruptured_km": 0,
            "last_rupture_year": 1766,
            "coupling_ratio": 0.85,
            "slip_rate_mm_yr": 24,
            "slip_deficit_m": 6.0,
            "notes": "1766'dan beri kirilmamis, 6m kayma acigi birikti"
        },
        "priority": 1, "region": "Turkiye",
    },
    "kahramanmaras": {
        "name": "Kahramanmaras (DAF)",
        "lat": 37.22, "lon": 37.02,
        "last_major": "2023-02-06", "last_major_mw": 7.8,
        "recurrence_years": 500,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Dogu Anadolu Fayi (DAF)",
        "expected_mw": "7.0-7.8",
        "population_risk": "~3 milyon",
        "segment_info": {
            "total_length_km": 300,
            "ruptured_km": 300,
            "last_rupture_year": 2023,
            "coupling_ratio": 0.1,
            "slip_rate_mm_yr": 10,
            "slip_deficit_m": 0.3,
            "notes": "2023'te tamamen kirildi, gerilim bosaldi"
        },
        "priority": 3, "region": "Turkiye",
    },
    "izmit": {
        "name": "Izmit (KAF)",
        "lat": 40.75, "lon": 29.86,
        "last_major": "1999-08-17", "last_major_mw": 7.6,
        "recurrence_years": 250,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "KAF - Izmit segmenti",
        "expected_mw": "7.0-7.6",
        "population_risk": "~5 milyon",
        "segment_info": {
            "total_length_km": 150,
            "ruptured_km": 150,
            "last_rupture_year": 1999,
            "coupling_ratio": 0.3,
            "slip_rate_mm_yr": 24,
            "slip_deficit_m": 0.6,
            "notes": "1999'da kirildi, henuz yeterli gerilim birikmedi"
        },
        "priority": 2, "region": "Turkiye",
    },
    "bolu_duzce": {
        "name": "Bolu-Duzce (KAF)",
        "lat": 40.75, "lon": 31.50,
        "last_major": "1999-11-12", "last_major_mw": 7.2,
        "recurrence_years": 300,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "KAF - Duzce segmenti",
        "expected_mw": "6.5-7.2",
        "population_risk": "~1 milyon",
        "segment_info": {
            "total_length_km": 80,
            "ruptured_km": 40,
            "last_rupture_year": 1999,
            "coupling_ratio": 0.4,
            "slip_rate_mm_yr": 18,
            "slip_deficit_m": 0.5,
            "notes": "Kismi kirilma, dogu segmenti hala riskli"
        },
        "priority": 2, "region": "Turkiye",
    },
    "erzincan": {
        "name": "Erzincan (KAF)",
        "lat": 39.75, "lon": 39.50,
        "last_major": "1992-03-13", "last_major_mw": 6.8,
        "recurrence_years": 200,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "KAF - Erzincan segmenti",
        "expected_mw": "6.5-7.5",
        "population_risk": "~500 bin",
        "segment_info": {
            "total_length_km": 120,
            "ruptured_km": 50,
            "last_rupture_year": 1992,
            "coupling_ratio": 0.6,
            "slip_rate_mm_yr": 20,
            "slip_deficit_m": 0.7,
            "notes": "1939 ve 1992 kirilmalarinin arasinda bosluk"
        },
        "priority": 2, "region": "Turkiye",
    },
    "fethiye_burdur": {
        "name": "Fethiye-Burdur Fayi",
        "lat": 36.80, "lon": 29.50,
        "last_major": "1957-04-25", "last_major_mw": 7.1,
        "recurrence_years": 200,
        "fault_type": "NORMAL",
        "fault_name": "Fethiye-Burdur Fay Zonu",
        "expected_mw": "6.5-7.2",
        "population_risk": "~2 milyon",
        "segment_info": {
            "total_length_km": 250,
            "ruptured_km": 80,
            "last_rupture_year": 1957,
            "coupling_ratio": 0.7,
            "slip_rate_mm_yr": 8,
            "slip_deficit_m": 0.5,
            "notes": "Turizm bolgesi, yuksek hasar potansiyeli"
        },
        "priority": 2, "region": "Turkiye",
    },

    # === JAPONYA ===
    "tohoku": {
        "name": "Tohoku (Japan Trench)",
        "lat": 38.30, "lon": 142.37,
        "last_major": "2011-03-11", "last_major_mw": 9.1,
        "recurrence_years": 600,
        "fault_type": "REVERSE",
        "fault_name": "Japan Trench Megathrust",
        "expected_mw": "8.0-9.0+",
        "population_risk": "~6 milyon",
        "segment_info": {
            "total_length_km": 500,
            "ruptured_km": 450,
            "last_rupture_year": 2011,
            "coupling_ratio": 0.2,
            "slip_rate_mm_yr": 80,
            "slip_deficit_m": 1.2,
            "notes": "2011'de kirildi, kuzeyde kalan segment izlenmeli"
        },
        "priority": 2, "region": "Japonya",
    },
    "nankai": {
        "name": "Nankai Trough",
        "lat": 33.00, "lon": 135.00,
        "last_major": "1946-12-21", "last_major_mw": 8.1,
        "recurrence_years": 140,
        "fault_type": "REVERSE",
        "fault_name": "Nankai Trough Megathrust",
        "expected_mw": "8.0-9.1",
        "population_risk": "~30 milyon",
        "segment_info": {
            "total_length_km": 700,
            "ruptured_km": 0,
            "last_rupture_year": 1946,
            "coupling_ratio": 0.85,
            "slip_rate_mm_yr": 45,
            "slip_deficit_m": 3.6,
            "notes": "80 yildir kirilmamis, 3.6m kayma acigi. Japon hukumeti %70-80 olasilik veriyor (30 yil)"
        },
        "priority": 1, "region": "Japonya",
    },
    "sagami": {
        "name": "Sagami Trough (Tokyo)",
        "lat": 35.20, "lon": 139.50,
        "last_major": "1923-09-01", "last_major_mw": 7.9,
        "recurrence_years": 200,
        "fault_type": "REVERSE",
        "fault_name": "Sagami Trough",
        "expected_mw": "7.5-8.2",
        "population_risk": "~38 milyon (Tokyo metro)",
        "segment_info": {
            "total_length_km": 200,
            "ruptured_km": 0,
            "last_rupture_year": 1923,
            "coupling_ratio": 0.75,
            "slip_rate_mm_yr": 30,
            "slip_deficit_m": 3.1,
            "notes": "1923 Kanto depreminden beri kirilmamis"
        },
        "priority": 1, "region": "Japonya",
    },

    # === ABD ===
    "cascadia": {
        "name": "Cascadia Subduction",
        "lat": 45.50, "lon": -125.00,
        "last_major": "1700-01-26", "last_major_mw": 9.0,
        "recurrence_years": 300,
        "fault_type": "REVERSE",
        "fault_name": "Cascadia Megathrust",
        "expected_mw": "8.5-9.2",
        "population_risk": "~10 milyon",
        "segment_info": {
            "total_length_km": 1100,
            "ruptured_km": 0,
            "last_rupture_year": 1700,
            "coupling_ratio": 0.95,
            "slip_rate_mm_yr": 40,
            "slip_deficit_m": 13.0,
            "notes": "326 yildir kirilmamis, 13m kayma acigi!"
        },
        "priority": 1, "region": "ABD",
    },
    "san_andreas_south": {
        "name": "San Andreas (Guney)",
        "lat": 33.70, "lon": -116.10,
        "last_major": "1857-01-09", "last_major_mw": 7.9,
        "recurrence_years": 150,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "San Andreas Fayi - Guney",
        "expected_mw": "7.2-8.0",
        "population_risk": "~20 milyon (LA metro)",
        "segment_info": {
            "total_length_km": 300,
            "ruptured_km": 0,
            "last_rupture_year": 1857,
            "coupling_ratio": 0.90,
            "slip_rate_mm_yr": 35,
            "slip_deficit_m": 5.9,
            "notes": "168 yildir kirilmamis, Big One beklentisi"
        },
        "priority": 1, "region": "ABD",
    },
    "hayward": {
        "name": "Hayward Fayi (SF Bay)",
        "lat": 37.70, "lon": -122.10,
        "last_major": "1868-10-21", "last_major_mw": 6.8,
        "recurrence_years": 140,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Hayward Fayi",
        "expected_mw": "6.8-7.2",
        "population_risk": "~7 milyon",
        "segment_info": {
            "total_length_km": 74,
            "ruptured_km": 0,
            "last_rupture_year": 1868,
            "coupling_ratio": 0.85,
            "slip_rate_mm_yr": 9,
            "slip_deficit_m": 1.4,
            "notes": "USGS 30 yil icinde %33 olasilik veriyor"
        },
        "priority": 1, "region": "ABD",
    },
    "new_madrid": {
        "name": "New Madrid (ABD ic)",
        "lat": 36.50, "lon": -89.60,
        "last_major": "1812-02-07", "last_major_mw": 7.7,
        "recurrence_years": 500,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "New Madrid Sismik Zonu",
        "expected_mw": "7.0-7.7",
        "population_risk": "~4 milyon",
        "segment_info": {
            "total_length_km": 250,
            "ruptured_km": 0,
            "last_rupture_year": 1812,
            "coupling_ratio": 0.5,
            "slip_rate_mm_yr": 2,
            "slip_deficit_m": 0.4,
            "notes": "Intraplate deprem, dusuk hiz ama yuksek hasar"
        },
        "priority": 2, "region": "ABD",
    },

    # === GUNEY AMERIKA ===
    "lima": {
        "name": "Lima (Peru Trench)",
        "lat": -12.00, "lon": -77.00,
        "last_major": "1746-10-28", "last_major_mw": 8.8,
        "recurrence_years": 300,
        "fault_type": "REVERSE",
        "fault_name": "Nazca - South America",
        "expected_mw": "8.5-9.0",
        "population_risk": "~12 milyon",
        "segment_info": {
            "total_length_km": 500,
            "ruptured_km": 0,
            "last_rupture_year": 1746,
            "coupling_ratio": 0.90,
            "slip_rate_mm_yr": 65,
            "slip_deficit_m": 18.2,
            "notes": "278 yildir kirilmamis, dev kayma acigi"
        },
        "priority": 1, "region": "Guney Amerika",
    },
    "valparaiso": {
        "name": "Valparaiso (Sili)",
        "lat": -33.00, "lon": -71.60,
        "last_major": "1906-08-17", "last_major_mw": 8.2,
        "recurrence_years": 80,
        "fault_type": "REVERSE",
        "fault_name": "Nazca - South America",
        "expected_mw": "8.0-8.5",
        "population_risk": "~5 milyon",
        "segment_info": {
            "total_length_km": 200,
            "ruptured_km": 0,
            "last_rupture_year": 1906,
            "coupling_ratio": 0.85,
            "slip_rate_mm_yr": 66,
            "slip_deficit_m": 7.9,
            "notes": "120 yildir kirilmamis, tekrarlama periyodu asildi"
        },
        "priority": 1, "region": "Guney Amerika",
    },
    "ecuador": {
        "name": "Ekvador-Kolombiya",
        "lat": 1.00, "lon": -79.50,
        "last_major": "1906-01-31", "last_major_mw": 8.8,
        "recurrence_years": 200,
        "fault_type": "REVERSE",
        "fault_name": "Nazca - South America (Kuzey)",
        "expected_mw": "7.5-8.5",
        "population_risk": "~8 milyon",
        "segment_info": {
            "total_length_km": 400,
            "ruptured_km": 150,
            "last_rupture_year": 2016,
            "coupling_ratio": 0.7,
            "slip_rate_mm_yr": 55,
            "slip_deficit_m": 3.5,
            "notes": "1906 tam kirilma, 2016 kismi kirilma, bosluk var"
        },
        "priority": 2, "region": "Guney Amerika",
    },

    # === AVRUPA ===
    "sicily": {
        "name": "Sicilya-Calabria",
        "lat": 38.20, "lon": 15.60,
        "last_major": "1908-12-28", "last_major_mw": 7.1,
        "recurrence_years": 200,
        "fault_type": "NORMAL",
        "fault_name": "Messina Bogazi Fayi",
        "expected_mw": "6.5-7.2",
        "population_risk": "~3 milyon",
        "segment_info": {
            "total_length_km": 100,
            "ruptured_km": 0,
            "last_rupture_year": 1908,
            "coupling_ratio": 0.6,
            "slip_rate_mm_yr": 3,
            "slip_deficit_m": 0.35,
            "notes": "1908 Messina tsunami, 80K+ olum"
        },
        "priority": 2, "region": "Avrupa",
    },
    "greece_hellenic": {
        "name": "Yunanistan (Hellenic Arc)",
        "lat": 35.50, "lon": 25.00,
        "last_major": "1903-08-11", "last_major_mw": 7.7,
        "recurrence_years": 150,
        "fault_type": "REVERSE",
        "fault_name": "Hellenic Subduction",
        "expected_mw": "7.5-8.5",
        "population_risk": "~5 milyon",
        "segment_info": {
            "total_length_km": 600,
            "ruptured_km": 200,
            "last_rupture_year": 1903,
            "coupling_ratio": 0.6,
            "slip_rate_mm_yr": 35,
            "slip_deficit_m": 4.3,
            "notes": "Girit onundeki megathrust, tsunami riski"
        },
        "priority": 1, "region": "Avrupa",
    },

    # === ASYA ===
    "nepal_mht": {
        "name": "Nepal (MHT)",
        "lat": 28.23, "lon": 84.73,
        "last_major": "2015-04-25", "last_major_mw": 7.8,
        "recurrence_years": 300,
        "fault_type": "REVERSE",
        "fault_name": "Main Himalayan Thrust",
        "expected_mw": "7.5-8.5",
        "population_risk": "~30 milyon",
        "segment_info": {
            "total_length_km": 600,
            "ruptured_km": 150,
            "last_rupture_year": 2015,
            "coupling_ratio": 0.75,
            "slip_rate_mm_yr": 18,
            "slip_deficit_m": 2.0,
            "notes": "2015 kismi kirilma, bati Nepal hala riskli"
        },
        "priority": 1, "region": "Asya",
    },
    "manila_trench": {
        "name": "Manila Trench (Filipinler)",
        "lat": 16.00, "lon": 119.50,
        "last_major": "1645-01-01", "last_major_mw": 8.0,
        "recurrence_years": 400,
        "fault_type": "REVERSE",
        "fault_name": "Manila Trench",
        "expected_mw": "8.0-8.8",
        "population_risk": "~25 milyon (Manila)",
        "segment_info": {
            "total_length_km": 400,
            "ruptured_km": 0,
            "last_rupture_year": 1645,
            "coupling_ratio": 0.8,
            "slip_rate_mm_yr": 60,
            "slip_deficit_m": 22.8,
            "notes": "380 yildir kirilmamis, dev kayma acigi, tsunami riski"
        },
        "priority": 1, "region": "Asya",
    },
    "makran": {
        "name": "Makran (Pakistan-Iran)",
        "lat": 25.50, "lon": 62.00,
        "last_major": "1945-11-27", "last_major_mw": 8.1,
        "recurrence_years": 200,
        "fault_type": "REVERSE",
        "fault_name": "Makran Subduction",
        "expected_mw": "8.0-8.8",
        "population_risk": "~5 milyon",
        "segment_info": {
            "total_length_km": 800,
            "ruptured_km": 200,
            "last_rupture_year": 1945,
            "coupling_ratio": 0.7,
            "slip_rate_mm_yr": 30,
            "slip_deficit_m": 2.4,
            "notes": "Bati Makran 250+ yildir kirilmamis"
        },
        "priority": 1, "region": "Asya",
    },

    # === OKYANUSYA ===
    "wellington_nz": {
        "name": "Wellington (Yeni Zelanda)",
        "lat": -41.30, "lon": 174.78,
        "last_major": "1855-01-23", "last_major_mw": 8.2,
        "recurrence_years": 500,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Wellington Fayi",
        "expected_mw": "7.0-8.0",
        "population_risk": "~500 bin",
        "segment_info": {
            "total_length_km": 75,
            "ruptured_km": 0,
            "last_rupture_year": 1855,
            "coupling_ratio": 0.7,
            "slip_rate_mm_yr": 7,
            "slip_deficit_m": 1.2,
            "notes": "Baskent, yuksek altyapi riski"
        },
        "priority": 2, "region": "Okyanusya",
    },
    "hikurangi": {
        "name": "Hikurangi (Yeni Zelanda)",
        "lat": -40.00, "lon": 178.00,
        "last_major": "1947-03-26", "last_major_mw": 7.1,
        "recurrence_years": 200,
        "fault_type": "REVERSE",
        "fault_name": "Hikurangi Subduction",
        "expected_mw": "8.0-8.8",
        "population_risk": "~2 milyon",
        "segment_info": {
            "total_length_km": 500,
            "ruptured_km": 50,
            "last_rupture_year": 1947,
            "coupling_ratio": 0.65,
            "slip_rate_mm_yr": 45,
            "slip_deficit_m": 3.6,
            "notes": "Slow slip olaylari gozleniyor, kuzey segment riskli"
        },
        "priority": 1, "region": "Okyanusya",
    },

    # === AFRIKA / ORTA DOGU ===
    "dead_sea": {
        "name": "Olü Deniz Fayi",
        "lat": 31.50, "lon": 35.50,
        "last_major": "1927-07-11", "last_major_mw": 6.3,
        "recurrence_years": 150,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Dead Sea Transform",
        "expected_mw": "6.5-7.5",
        "population_risk": "~15 milyon",
        "segment_info": {
            "total_length_km": 1000,
            "ruptured_km": 100,
            "last_rupture_year": 1927,
            "coupling_ratio": 0.6,
            "slip_rate_mm_yr": 5,
            "slip_deficit_m": 0.5,
            "notes": "1202 ve 1837 buyuk depremleri, uzun sessizlik"
        },
        "priority": 2, "region": "Orta Dogu",
    },
    "east_african_rift": {
        "name": "Dogu Afrika Rift",
        "lat": -2.00, "lon": 36.00,
        "last_major": "2006-07-17", "last_major_mw": 7.0,
        "recurrence_years": 300,
        "fault_type": "NORMAL",
        "fault_name": "East African Rift System",
        "expected_mw": "6.5-7.5",
        "population_risk": "~10 milyon",
        "segment_info": {
            "total_length_km": 3000,
            "ruptured_km": 100,
            "last_rupture_year": 2006,
            "coupling_ratio": 0.4,
            "slip_rate_mm_yr": 6,
            "slip_deficit_m": 0.5,
            "notes": "Cok uzun fay sistemi, parcali kirilma"
        },
        "priority": 2, "region": "Afrika",
    },
}


def compute_segment_risk(zone):
    """
    Fay segment analizi ile ek risk skoru hesapla.
    
    Faktorler:
    1. Kayma acigi (slip deficit) - ne kadar gerilim birikti
    2. Coupling orani - fay ne kadar kilitli
    3. Tekrarlama oranini asma - periyodu gecti mi
    4. Kirilma boyu - ne kadar kirilmamis segment var
    """
    seg = zone.get("segment_info")
    if not seg:
        return {"segment_risk_score": None}

    score = 0
    factors = []

    # 1. Kayma acigi (en guclu faktor)
    deficit = seg.get("slip_deficit_m", 0)
    if deficit >= 10:
        score += 0.35
        factors.append(f"Cok yuksek kayma acigi ({deficit:.1f}m)")
    elif deficit >= 5:
        score += 0.25
        factors.append(f"Yuksek kayma acigi ({deficit:.1f}m)")
    elif deficit >= 2:
        score += 0.15
        factors.append(f"Orta kayma acigi ({deficit:.1f}m)")
    elif deficit >= 0.5:
        score += 0.05
        factors.append(f"Dusuk kayma acigi ({deficit:.1f}m)")

    # 2. Coupling orani
    coupling = seg.get("coupling_ratio", 0)
    if coupling >= 0.8:
        score += 0.25
        factors.append(f"Yuksek coupling ({coupling:.2f})")
    elif coupling >= 0.5:
        score += 0.15
        factors.append(f"Orta coupling ({coupling:.2f})")

    # 3. Tekrarlama periyodunu asma
    last_year = seg.get("last_rupture_year")
    rec = zone.get("recurrence_years", 9999)
    if last_year:
        years_since = 2025 - last_year
        ratio = years_since / rec
        if ratio >= 1.0:
            score += 0.25
            factors.append(f"Periyot asildi ({years_since}/{rec} yil = {ratio:.2f})")
        elif ratio >= 0.7:
            score += 0.15
            factors.append(f"Gec faz ({years_since}/{rec} yil)")
        elif ratio >= 0.4:
            score += 0.05
            factors.append(f"Orta faz ({years_since}/{rec} yil)")

    # 4. Kirilmamis segment orani
    total = seg.get("total_length_km", 0)
    ruptured = seg.get("ruptured_km", 0)
    if total > 0:
        unruptured_ratio = 1 - (ruptured / total)
        if unruptured_ratio >= 0.8:
            score += 0.15
            factors.append(f"Buyuk kirilmamis segment ({(unruptured_ratio*100):.0f}%)")
        elif unruptured_ratio >= 0.5:
            score += 0.08
            factors.append(f"Orta kirilmamis segment ({(unruptured_ratio*100):.0f}%)")

    # Normalize (0-1)
    score = min(1.0, score)

    if score >= 0.75: level = "KRITIK"
    elif score >= 0.50: level = "YUKSEK"
    elif score >= 0.30: level = "ORTA"
    elif score >= 0.15: level = "DIKKAT"
    else: level = "DUSUK"

    return {
        "segment_risk_score": round(score, 3),
        "segment_risk_level": level,
        "segment_factors": factors,
        "slip_deficit_m": deficit,
        "coupling_ratio": coupling,
    }


def get_all_zones():
    """Tum bolgeleri segment analizi ile dondur."""
    result = {}
    for key, zone in EXTENDED_ZONES.items():
        z = dict(zone)
        seg_risk = compute_segment_risk(zone)
        z.update(seg_risk)
        result[key] = z
    return result


def get_zones_sorted():
    """Bolgeleri segment riskine gore sirali dondur."""
    zones = get_all_zones()
    sorted_zones = sorted(
        zones.items(),
        key=lambda x: (x[1].get("segment_risk_score") or 0),
        reverse=True
    )
    return sorted_zones


def save_zones():
    """Genisletilmis bolge veritabanini kaydet."""
    zones = get_all_zones()
    ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2, default=str)
    print(f"Kaydedildi: {ZONES_FILE} ({len(zones)} bolge)")


def print_report():
    """Tum bolgelerin segment riskini yazdir."""
    print("=" * 80)
    print("GENISLETILMIS BOLGE RISK RAPORU")
    print("=" * 80)

    sorted_z = get_zones_sorted()

    print(f"\n{'Bolge':<35} {'Seg.Risk':>8} {'Seviye':<8} "
          f"{'Deficit':>8} {'Coupling':>8} {'Nufus':<15}")
    print("-" * 95)

    for key, z in sorted_z:
        seg_score = z.get("segment_risk_score")
        seg_level = z.get("segment_risk_level", "?")
        deficit = z.get("slip_deficit_m", 0)
        coupling = z.get("coupling_ratio", 0)
        pop = z.get("population_risk", "-")

        if seg_score is not None:
            print(f"  {z['name']:<33} {seg_score*100:>7.1f}% {seg_level:<8} "
                  f"{deficit:>7.1f}m {coupling:>7.2f} {pop:<15}")
            for factor in z.get("segment_factors", []):
                print(f"    - {factor}")
        else:
            print(f"  {z['name']:<33} {'N/A':>8}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.report:
        print_report()
    elif args.save:
        save_zones()
        print_report()
    else:
        print_report()
        save_zones()


if __name__ == "__main__":
    main()