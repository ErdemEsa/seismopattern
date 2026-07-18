#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - 25 yeni bolge ekleyici
Mevcut 25 bolgeye 25 yeni bolge ekler (toplam 50).
"""

import json
from pathlib import Path

ZONES_FILE = Path("data/zones_extended.json")

NEW_ZONES = {
    # === SUBDUCTION ===
    "aleutian": {
        "name": "Aleutian Islands (Alaska)",
        "lat": 51.5, "lon": -176.0,
        "last_major": "1996-06-10", "last_major_mw": 7.9,
        "recurrence_years": 50,
        "fault_type": "REVERSE",
        "fault_name": "Aleutian Megathrust",
        "expected_mw": "7.5-9.0",
        "population_risk": "Dusuk (kiyisal tsunami riski)",
        "segment_info": {
            "total_length_km": 3400,
            "ruptured_km": 300,
            "last_rupture_year": 1996,
            "coupling_ratio": 0.70,
            "slip_rate_mm_yr": 65,
            "slip_deficit_m": 1.9,
            "notes": "Cok segmentli subduction, sik Mw7+ uretir"
        },
        "priority": 2, "region": "Kuzey Amerika",
    },
    "kuril": {
        "name": "Kuril Trench",
        "lat": 45.0, "lon": 150.0,
        "last_major": "2006-11-15", "last_major_mw": 8.3,
        "recurrence_years": 80,
        "fault_type": "REVERSE",
        "fault_name": "Kuril-Kamchatka Trench",
        "expected_mw": "7.5-8.5",
        "population_risk": "Orta (tsunami riski)",
        "segment_info": {
            "total_length_km": 2200,
            "ruptured_km": 500,
            "last_rupture_year": 2006,
            "coupling_ratio": 0.65,
            "slip_rate_mm_yr": 80,
            "slip_deficit_m": 1.6,
            "notes": "Kuzeybati Pasifik subduction"
        },
        "priority": 2, "region": "Japonya",
    },
    "kamchatka": {
        "name": "Kamchatka",
        "lat": 53.0, "lon": 160.0,
        "last_major": "1997-12-05", "last_major_mw": 7.8,
        "recurrence_years": 60,
        "fault_type": "REVERSE",
        "fault_name": "Kamchatka Subduction Zone",
        "expected_mw": "7.5-9.0",
        "population_risk": "Dusuk",
        "segment_info": {
            "total_length_km": 1500,
            "ruptured_km": 200,
            "last_rupture_year": 1997,
            "coupling_ratio": 0.75,
            "slip_rate_mm_yr": 78,
            "slip_deficit_m": 2.1,
            "notes": "1952 Mw9.0 uretmis, cok aktif zon"
        },
        "priority": 2, "region": "Rusya",
    },
    "sumatra_north": {
        "name": "Kuzey Sumatra (Mentawai Gap)",
        "lat": 1.0, "lon": 97.0,
        "last_major": "2005-03-28", "last_major_mw": 8.6,
        "recurrence_years": 200,
        "fault_type": "REVERSE",
        "fault_name": "Sunda Megathrust",
        "expected_mw": "7.5-8.8",
        "population_risk": "Yuksek (~5 milyon kiyisal)",
        "segment_info": {
            "total_length_km": 1600,
            "ruptured_km": 300,
            "last_rupture_year": 2005,
            "coupling_ratio": 0.60,
            "slip_rate_mm_yr": 50,
            "slip_deficit_m": 1.0,
            "notes": "Mentawai gap hala kirilmadi"
        },
        "priority": 1, "region": "Guneydogu Asya",
    },
    "tonga_kermadec": {
        "name": "Tonga-Kermadec Trench",
        "lat": -22.0, "lon": -175.0,
        "last_major": "2009-09-29", "last_major_mw": 8.1,
        "recurrence_years": 60,
        "fault_type": "REVERSE",
        "fault_name": "Tonga-Kermadec Subduction",
        "expected_mw": "7.5-8.5",
        "population_risk": "Dusuk (ada toplulugu)",
        "segment_info": {
            "total_length_km": 2500,
            "ruptured_km": 400,
            "last_rupture_year": 2009,
            "coupling_ratio": 0.50,
            "slip_rate_mm_yr": 80,
            "slip_deficit_m": 1.2,
            "notes": "Dunyanin en hizli subduksiyon zonu"
        },
        "priority": 3, "region": "Okyanusya",
    },
    "puerto_rico": {
        "name": "Puerto Rico Trench",
        "lat": 19.5, "lon": -66.5,
        "last_major": "1946-08-04", "last_major_mw": 8.1,
        "recurrence_years": 150,
        "fault_type": "REVERSE",
        "fault_name": "Puerto Rico Trench",
        "expected_mw": "7.5-8.5",
        "population_risk": "Yuksek (~4 milyon, tsunami riski)",
        "segment_info": {
            "total_length_km": 800,
            "ruptured_km": 200,
            "last_rupture_year": 1946,
            "coupling_ratio": 0.55,
            "slip_rate_mm_yr": 20,
            "slip_deficit_m": 1.6,
            "notes": "Atlantikte en derin hendek"
        },
        "priority": 2, "region": "Karayipler",
    },
    "solomon": {
        "name": "Solomon Islands",
        "lat": -8.5, "lon": 157.0,
        "last_major": "2007-04-01", "last_major_mw": 8.1,
        "recurrence_years": 50,
        "fault_type": "REVERSE",
        "fault_name": "Solomon Islands Subduction",
        "expected_mw": "7.0-8.0",
        "population_risk": "Dusuk",
        "segment_info": {
            "total_length_km": 900,
            "ruptured_km": 300,
            "last_rupture_year": 2007,
            "coupling_ratio": 0.45,
            "slip_rate_mm_yr": 95,
            "slip_deficit_m": 0.9,
            "notes": "Cok hizli yakinsama, sik buyuk deprem"
        },
        "priority": 3, "region": "Okyanusya",
    },

    # === TRANSFORM ===
    "altai": {
        "name": "Altai (Rusya-Mogolistan)",
        "lat": 50.0, "lon": 88.0,
        "last_major": "2003-09-27", "last_major_mw": 7.3,
        "recurrence_years": 150,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Altai Fault System",
        "expected_mw": "7.0-7.5",
        "population_risk": "Dusuk",
        "segment_info": {
            "total_length_km": 500,
            "ruptured_km": 100,
            "last_rupture_year": 2003,
            "coupling_ratio": 0.40,
            "slip_rate_mm_yr": 3,
            "slip_deficit_m": 0.3,
            "notes": "Intrakontinental transform"
        },
        "priority": 3, "region": "Orta Asya",
    },
    "chaman": {
        "name": "Chaman Fayi (Pakistan)",
        "lat": 30.5, "lon": 67.0,
        "last_major": "2013-09-24", "last_major_mw": 7.7,
        "recurrence_years": 100,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Chaman Fault",
        "expected_mw": "7.0-7.8",
        "population_risk": "Orta (~2 milyon)",
        "segment_info": {
            "total_length_km": 860,
            "ruptured_km": 200,
            "last_rupture_year": 2013,
            "coupling_ratio": 0.50,
            "slip_rate_mm_yr": 25,
            "slip_deficit_m": 1.3,
            "notes": "Hindistan-Avrasya siniri, sol yanal"
        },
        "priority": 2, "region": "Guney Asya",
    },
    "enriquillo": {
        "name": "Enriquillo-Plantain (Haiti)",
        "lat": 18.5, "lon": -72.5,
        "last_major": "2010-01-12", "last_major_mw": 7.0,
        "recurrence_years": 200,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Enriquillo-Plantain Garden Fault",
        "expected_mw": "7.0-7.5",
        "population_risk": "Cok Yuksek (~3 milyon Port-au-Prince)",
        "segment_info": {
            "total_length_km": 300,
            "ruptured_km": 65,
            "last_rupture_year": 2010,
            "coupling_ratio": 0.60,
            "slip_rate_mm_yr": 7,
            "slip_deficit_m": 0.9,
            "notes": "2010 felaketi, bati segment hala kilitli"
        },
        "priority": 2, "region": "Karayipler",
    },
    "north_anatolian_east": {
        "name": "KAF Dogu (Erzurum-Karliova)",
        "lat": 39.9, "lon": 41.5,
        "last_major": "1939-12-26", "last_major_mw": 7.8,
        "recurrence_years": 200,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "Kuzey Anadolu Fayi (Dogu)",
        "expected_mw": "7.0-7.5",
        "population_risk": "Orta (~1 milyon)",
        "segment_info": {
            "total_length_km": 350,
            "ruptured_km": 100,
            "last_rupture_year": 1939,
            "coupling_ratio": 0.65,
            "slip_rate_mm_yr": 20,
            "slip_deficit_m": 1.7,
            "notes": "1939 Erzincan sonrasi bosalan segment"
        },
        "priority": 2, "region": "Turkiye",
    },

    # === CONTINENTAL COLLISION ===
    "sichuan": {
        "name": "Sichuan (Longmenshan)",
        "lat": 31.0, "lon": 103.4,
        "last_major": "2008-05-12", "last_major_mw": 7.9,
        "recurrence_years": 300,
        "fault_type": "REVERSE",
        "fault_name": "Longmenshan Fault",
        "expected_mw": "7.0-8.0",
        "population_risk": "Cok Yuksek (~80 milyon Sichuan Basin)",
        "segment_info": {
            "total_length_km": 500,
            "ruptured_km": 240,
            "last_rupture_year": 2008,
            "coupling_ratio": 0.45,
            "slip_rate_mm_yr": 3,
            "slip_deficit_m": 0.5,
            "notes": "2008 Wenchuan felaketi"
        },
        "priority": 2, "region": "Cin",
    },
    "hindu_kush": {
        "name": "Hindu Kush (Afganistan)",
        "lat": 36.5, "lon": 71.0,
        "last_major": "2015-10-26", "last_major_mw": 7.5,
        "recurrence_years": 40,
        "fault_type": "REVERSE",
        "fault_name": "Hindu Kush Subduction Zone",
        "expected_mw": "7.0-7.8",
        "population_risk": "Yuksek (~5 milyon)",
        "segment_info": {
            "total_length_km": 600,
            "ruptured_km": 100,
            "last_rupture_year": 2015,
            "coupling_ratio": 0.50,
            "slip_rate_mm_yr": 30,
            "slip_deficit_m": 0.3,
            "notes": "Derin deprem ureten zon (200km+)"
        },
        "priority": 2, "region": "Orta Asya",
    },
    "zagros": {
        "name": "Zagros (Iran)",
        "lat": 32.5, "lon": 50.0,
        "last_major": "2017-11-12", "last_major_mw": 7.3,
        "recurrence_years": 80,
        "fault_type": "REVERSE",
        "fault_name": "Zagros Fold-Thrust Belt",
        "expected_mw": "6.5-7.5",
        "population_risk": "Yuksek (~10 milyon)",
        "segment_info": {
            "total_length_km": 1500,
            "ruptured_km": 100,
            "last_rupture_year": 2017,
            "coupling_ratio": 0.35,
            "slip_rate_mm_yr": 10,
            "slip_deficit_m": 0.4,
            "notes": "Arap-Avrasya carpismasi, yaygik deformasyon"
        },
        "priority": 2, "region": "Iran",
    },
    "tabriz": {
        "name": "Tabriz (Kuzey Iran)",
        "lat": 38.0, "lon": 46.5,
        "last_major": "1780-01-08", "last_major_mw": 7.4,
        "recurrence_years": 250,
        "fault_type": "STRIKE_SLIP",
        "fault_name": "North Tabriz Fault",
        "expected_mw": "7.0-7.5",
        "population_risk": "Cok Yuksek (~2 milyon Tabriz)",
        "segment_info": {
            "total_length_km": 200,
            "ruptured_km": 0,
            "last_rupture_year": 1780,
            "coupling_ratio": 0.75,
            "slip_rate_mm_yr": 7,
            "slip_deficit_m": 1.7,
            "notes": "246 yildir kirilmamis, onemli kayma acigi"
        },
        "priority": 1, "region": "Iran",
    },
    "apennine": {
        "name": "Orta Apennine (Italya)",
        "lat": 42.5, "lon": 13.3,
        "last_major": "2016-10-30", "last_major_mw": 6.6,
        "recurrence_years": 100,
        "fault_type": "NORMAL",
        "fault_name": "Central Apennine Fault System",
        "expected_mw": "6.5-7.0",
        "population_risk": "Orta",
        "segment_info": {
            "total_length_km": 400,
            "ruptured_km": 50,
            "last_rupture_year": 2016,
            "coupling_ratio": 0.40,
            "slip_rate_mm_yr": 2,
            "slip_deficit_m": 0.2,
            "notes": "2009 L'Aquila, 2016 Amatrice dizisi"
        },
        "priority": 3, "region": "Avrupa",
    },

    # === INTRAPLATE ===
    "bhuj": {
        "name": "Bhuj (Gujarat, Hindistan)",
        "lat": 23.4, "lon": 70.2,
        "last_major": "2001-01-26", "last_major_mw": 7.7,
        "recurrence_years": 500,
        "fault_type": "REVERSE",
        "fault_name": "Kachchh Rift Basin",
        "expected_mw": "6.5-7.5",
        "population_risk": "Yuksek (~5 milyon)",
        "segment_info": {
            "total_length_km": 300,
            "ruptured_km": 50,
            "last_rupture_year": 2001,
            "coupling_ratio": 0.20,
            "slip_rate_mm_yr": 1,
            "slip_deficit_m": 0.2,
            "notes": "Intraplate, 2001 felaketi"
        },
        "priority": 3, "region": "Guney Asya",
    },
    "shanxi": {
        "name": "Shanxi Graben (Cin)",
        "lat": 37.5, "lon": 112.5,
        "last_major": "1556-01-23", "last_major_mw": 8.0,
        "recurrence_years": 500,
        "fault_type": "NORMAL",
        "fault_name": "Shanxi Rift System",
        "expected_mw": "7.0-8.0",
        "population_risk": "Cok Yuksek (~30 milyon)",
        "segment_info": {
            "total_length_km": 600,
            "ruptured_km": 0,
            "last_rupture_year": 1556,
            "coupling_ratio": 0.30,
            "slip_rate_mm_yr": 2,
            "slip_deficit_m": 0.9,
            "notes": "1556 tarihin en olumcul depremi (830K+)"
        },
        "priority": 2, "region": "Cin",
    },

    # === RIFT / EXTENSION ===
    "baikal": {
        "name": "Baikal Rift",
        "lat": 52.0, "lon": 107.0,
        "last_major": "1957-06-27", "last_major_mw": 7.6,
        "recurrence_years": 150,
        "fault_type": "NORMAL",
        "fault_name": "Baikal Rift Zone",
        "expected_mw": "7.0-7.5",
        "population_risk": "Dusuk",
        "segment_info": {
            "total_length_km": 1500,
            "ruptured_km": 200,
            "last_rupture_year": 1957,
            "coupling_ratio": 0.25,
            "slip_rate_mm_yr": 4,
            "slip_deficit_m": 0.3,
            "notes": "Derin gol boyunca rift sistemi"
        },
        "priority": 3, "region": "Rusya",
    },
    "rhine_graben": {
        "name": "Rhine Graben (Almanya-Fransa)",
        "lat": 48.5, "lon": 7.7,
        "last_major": "1356-10-18", "last_major_mw": 6.5,
        "recurrence_years": 500,
        "fault_type": "NORMAL",
        "fault_name": "Upper Rhine Graben",
        "expected_mw": "6.0-6.5",
        "population_risk": "Yuksek (~5 milyon, Basel-Strasbourg)",
        "segment_info": {
            "total_length_km": 300,
            "ruptured_km": 0,
            "last_rupture_year": 1356,
            "coupling_ratio": 0.20,
            "slip_rate_mm_yr": 1,
            "slip_deficit_m": 0.7,
            "notes": "1356 Basel depremi, dusuk sismisiteli zon"
        },
        "priority": 3, "region": "Avrupa",
    },
    "corinth": {
        "name": "Korint Korfezi (Yunanistan)",
        "lat": 38.3, "lon": 22.0,
        "last_major": "1995-06-15", "last_major_mw": 6.5,
        "recurrence_years": 60,
        "fault_type": "NORMAL",
        "fault_name": "Gulf of Corinth Rift",
        "expected_mw": "6.5-7.0",
        "population_risk": "Orta (~500K, Patras dahil)",
        "segment_info": {
            "total_length_km": 120,
            "ruptured_km": 30,
            "last_rupture_year": 1995,
            "coupling_ratio": 0.55,
            "slip_rate_mm_yr": 15,
            "slip_deficit_m": 0.5,
            "notes": "Avrupanin en hizli acilan rifti"
        },
        "priority": 2, "region": "Avrupa",
    },

    # === ADA YAYI ===
    "caribbean_north": {
        "name": "Kuzey Karayip Yitim Zonu",
        "lat": 17.0, "lon": -62.0,
        "last_major": "2004-11-21", "last_major_mw": 6.3,
        "recurrence_years": 200,
        "fault_type": "REVERSE",
        "fault_name": "Lesser Antilles Subduction",
        "expected_mw": "7.0-8.0",
        "population_risk": "Yuksek (ada devletleri)",
        "segment_info": {
            "total_length_km": 850,
            "ruptured_km": 100,
            "last_rupture_year": 2004,
            "coupling_ratio": 0.40,
            "slip_rate_mm_yr": 20,
            "slip_deficit_m": 1.2,
            "notes": "1843 Mw8.0 uretmis, uzun sessiz donem"
        },
        "priority": 2, "region": "Karayipler",
    },
    "vanuatu": {
        "name": "Vanuatu",
        "lat": -16.0, "lon": 167.0,
        "last_major": "2018-08-22", "last_major_mw": 7.0,
        "recurrence_years": 30,
        "fault_type": "REVERSE",
        "fault_name": "Vanuatu Subduction Zone",
        "expected_mw": "7.0-7.8",
        "population_risk": "Dusuk",
        "segment_info": {
            "total_length_km": 1200,
            "ruptured_km": 200,
            "last_rupture_year": 2018,
            "coupling_ratio": 0.35,
            "slip_rate_mm_yr": 90,
            "slip_deficit_m": 0.3,
            "notes": "Cok hizli yakinsama, sik deprem"
        },
        "priority": 3, "region": "Okyanusya",
    },
    "banda_sea": {
        "name": "Banda Sea (Endonezya)",
        "lat": -6.5, "lon": 129.5,
        "last_major": "2019-09-26", "last_major_mw": 6.5,
        "recurrence_years": 80,
        "fault_type": "REVERSE",
        "fault_name": "Banda Arc",
        "expected_mw": "7.0-8.0",
        "population_risk": "Orta",
        "segment_info": {
            "total_length_km": 1000,
            "ruptured_km": 200,
            "last_rupture_year": 2019,
            "coupling_ratio": 0.40,
            "slip_rate_mm_yr": 60,
            "slip_deficit_m": 0.8,
            "notes": "Karmasik tektonik, cift subduction"
        },
        "priority": 3, "region": "Guneydogu Asya",
    },
    "java_trench": {
        "name": "Java Trench (Guney Endonezya)",
        "lat": -10.0, "lon": 112.0,
        "last_major": "1994-06-02", "last_major_mw": 7.8,
        "recurrence_years": 100,
        "fault_type": "REVERSE",
        "fault_name": "Java Subduction Zone",
        "expected_mw": "7.5-8.5",
        "population_risk": "Cok Yuksek (~50 milyon guney Java)",
        "segment_info": {
            "total_length_km": 3000,
            "ruptured_km": 200,
            "last_rupture_year": 1994,
            "coupling_ratio": 0.50,
            "slip_rate_mm_yr": 60,
            "slip_deficit_m": 1.9,
            "notes": "Dunya nufus yogunlugu en yuksek subduction"
        },
        "priority": 1, "region": "Guneydogu Asya",
    },
}


def main():
    # Mevcut zones'u oku
    if ZONES_FILE.exists():
        existing = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
    else:
        existing = {}

    print(f"Mevcut bolge: {len(existing)}")

    added = 0
    skipped = 0

    for key, zone in NEW_ZONES.items():
        if key in existing:
            print(f"  ATLANDI (zaten var): {key}")
            skipped += 1
        else:
            existing[key] = zone
            added += 1
            print(f"  EKLENDI: {key} -> {zone['name']}")

    # Kaydet
    ZONES_FILE.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nSonuc:")
    print(f"  Eklenen : {added}")
    print(f"  Atlanan : {skipped}")
    print(f"  Toplam  : {len(existing)}")

    # zone_database.py sync uyarisi
    print(f"\nUYARI: scripts/zone_database.py icerisindeki EXTENDED_ZONES'u da guncellemelisiniz.")
    print(f"       Ama app.py zones_extended.json dosyasini dogrudan okuyorsa")
    print(f"       bu adim yeterli olabilir.")


if __name__ == "__main__":
    main()