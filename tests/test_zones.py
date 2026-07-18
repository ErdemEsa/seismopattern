import json
from pathlib import Path


def test_zones_json_exists():
    assert Path("data/zones_extended.json").exists()


def test_zone_count_is_50_or_more():
    z = json.loads(Path("data/zones_extended.json").read_text(encoding="utf-8"))
    assert len(z) >= 50, f"Beklenen en az 50 bolge, bulundu: {len(z)}"


def test_new_zone_keys_exist():
    z = json.loads(Path("data/zones_extended.json").read_text(encoding="utf-8"))
    required = [
        "aleutian", "kuril", "kamchatka", "sumatra_north", "tonga_kermadec",
        "puerto_rico", "solomon", "altai", "chaman", "enriquillo",
        "north_anatolian_east", "sichuan", "hindu_kush", "zagros", "tabriz",
        "apennine", "bhuj", "shanxi", "baikal", "rhine_graben", "corinth",
        "caribbean_north", "vanuatu", "banda_sea", "java_trench",
    ]
    missing = [k for k in required if k not in z]
    assert not missing, f"Eksik yeni bolgeler: {missing}"


def test_zone_schema_minimum():
    z = json.loads(Path("data/zones_extended.json").read_text(encoding="utf-8"))
    sample_keys = ["marmara", "aleutian", "tabriz", "java_trench"]
    for key in sample_keys:
        assert key in z, f"{key} bulunamadi"
        item = z[key]
        for field in ["name", "lat", "lon", "fault_name", "fault_type", "region", "segment_info"]:
            assert field in item, f"{key} icin eksik alan: {field}"
        assert isinstance(item["segment_info"], dict)
