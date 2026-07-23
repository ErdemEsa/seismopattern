#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template_string, request

# =========================================================
# PERFORMANS: GLOBAL LAZY-LOAD CACHE
# =========================================================

import time as _time

_STARTUP = _time.time()

# Tüm modüller burada bir kez yüklenir
# Sonraki çağrılarda cache'ten döner

# Model cache (zaten var, dokunmuyoruz)
# MODELS, FL = load_models() — bu zaten dosyanın altında

# Debiased model cache
_DEB_MODEL_CACHE = None
_DEB_FEATS_CACHE = None

def _get_debiased():
    global _DEB_MODEL_CACHE, _DEB_FEATS_CACHE
    if _DEB_MODEL_CACHE is None:
        deb_path = Path("output/counterfactual/debiased_model.joblib")
        deb_feat_path = Path("output/counterfactual/debiased_features.json")
        if deb_path.exists() and deb_feat_path.exists():
            _DEB_MODEL_CACHE = joblib.load(deb_path)
            with open(deb_feat_path) as f:
                _DEB_FEATS_CACHE = json.load(f)
    return _DEB_MODEL_CACHE, _DEB_FEATS_CACHE


# Query cache (ISC+USGS sonuçları)
_QUERY_CACHE_TTL = 7200  # 2 saat

warnings.filterwarnings("ignore")

app = Flask(__name__)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/healthz")
def healthz():
    return {"status": "ok"}, 200
MODEL_DIR = Path("output/models")

sys.path.insert(0, "scripts")
try:
    from isc_fetch_v2 import fetch_and_analyze as isc_fetch_analyze
    HAS_ISC = True
except Exception:
    HAS_ISC = False


# =========================================================
# YARDIMCI
# =========================================================

def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def rule_based_type(r):
    qr = r.get("quiescence_ratio")
    acc = r.get("accel_90d")
    n3 = r.get("w3_n_events", 0) or 0

    if qr is None or pd.isna(qr) or n3 < 3:
        return "TIP_C"
    if qr < 0.5:
        return "TIP_B"
    if qr >= 1.0:
        return "TIP_A"
    if 0.5 <= qr < 0.8:
        if acc is not None and not pd.isna(acc) and acc >= 1.5:
            return "TIP_A"
        return "TIP_B"
    return "TIP_A"


def add_derived(r):
    r = r.copy()

    c0 = float(r.get("count_0_1y", 0) or 0)
    c1 = float(r.get("count_1_2y", 0) or 0)
    c2 = float(r.get("count_2_3y", 0) or 0)

    r["count_linear_trend"] = c0 - c2
    r["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)

    def safe_diff(a, b):
        try:
            if a is None or b is None:
                return np.nan
            return float(a) - float(b)
        except Exception:
            return np.nan

    r["b_drop_w3_w1"] = safe_diff(r.get("w3_b_value"), r.get("w1_b_value"))
    r["spatial_focus_change"] = safe_diff(r.get("w3_mean_dist_km"), r.get("w1_mean_dist_km"))
    r["depth_change_km"] = safe_diff(r.get("w1_mean_depth_km"), r.get("w3_mean_depth_km"))

    if not r.get("quiescence_ratio"):
        prev = (c1 + c2) / 2.0
        r["quiescence_ratio"] = c0 / prev if prev > 0 else np.nan

    if not r.get("w3_n_events"):
        r["w3_n_events"] = c0 + c1 + c2

    if not r.get("w1_n_events"):
        r["w1_n_events"] = c0

    return r


def predict_risk(features, models, feature_lists):
    row = add_derived(features)
    pt = rule_based_type(row)

    component_scores = {}

    for tip, (pipe, feats) in models.items():
        if pipe is None:
            continue
        try:
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
            p = float(pipe.predict_proba(x)[0, 1])
            component_scores[tip] = round(p, 4)
        except Exception:
            component_scores[tip] = None

    valid = {k: v for k, v in component_scores.items() if v is not None}
    if not valid:
        return {"error": "Model skoru hesaplanamadi"}

    weights = {"TIP_A": 1.0, "TIP_B": 1.2, "TIP_C": 0.5}
    primary = component_scores.get(pt)

    ws = sum(weights.get(t, 1.0) * s for t, s in valid.items())
    wt = sum(weights.get(t, 1.0) for t in valid)
    ensemble = ws / wt if wt > 0 else 0.5

    final = 0.7 * primary + 0.3 * ensemble if primary is not None else ensemble
    final = round(float(final), 4)

    # Kalibre edilmis model varsa onu kullan
    if "CALIBRATED" in models:
        cal_pipe, cal_feats = models["CALIBRATED"]
        try:
            x_cal = pd.DataFrame([{f: row.get(f, np.nan) for f in cal_feats}])
            cal_score = float(cal_pipe.predict_proba(x_cal)[0, 1])

            # %70 tip-bazli skor + %30 kalibre model
            final = round(0.70 * final + 0.30 * cal_score, 4)
            component_scores["CALIBRATED"] = round(cal_score, 4)

        except Exception as e:
            component_scores["CALIBRATED"] = f"HATA: {str(e)}"

    if final >= 0.75:
        risk_level = "KRITIK"
        risk_class = "K"
    elif final >= 0.60:
        risk_level = "YUKSEK"
        risk_class = "Y"
    elif final >= 0.45:
        risk_level = "ORTA"
        risk_class = "O"
    elif final >= 0.30:
        risk_level = "DIKKAT"
        risk_class = "D"
    else:
        risk_level = "DUSUK"
        risk_class = "L"

    confidence_issues = []
    n_total = row.get("w3_n_events", 0) or 0

    if n_total < 20:
        confidence_issues.append("Az veri")
    if row.get("w1_b_value") is None:
        confidence_issues.append("Son 1 yil b-degeri yok")

    z_all_zero = all(
        (row.get(k, 0) == 0 or row.get(k, 0) is None)
        for k in ["z_rate_1y", "z_b_value_1y", "z_max_mw_1y", "z_dist_1y"]
    )
    if z_all_zero:
        confidence_issues.append("Normalize z-score eksik")

    if n_total < 50 and row.get("w1_b_value") is None:
        confidence = "DUSUK"
    elif len(confidence_issues) >= 2:
        confidence = "DUSUK"
    elif len(confidence_issues) == 1:
        confidence = "ORTA"
    else:
        confidence = "YUKSEK"

    parts = []
    if pt == "TIP_A":
        parts.append("Aktivasyon sablonu tespit edildi")
    elif pt == "TIP_B":
        parts.append("Sismik sessizlik sablonu tespit edildi")
    else:
        parts.append("Belirsiz sablon")

    qr = row.get("quiescence_ratio")
    if qr is not None and not pd.isna(qr):
        if qr < 0.5:
            parts.append(f"Aktivite onceki doneme gore %{(1-qr)*100:.0f} azaldi")
        elif qr > 1.5:
            parts.append(f"Aktivite onceki doneme gore %{(qr-1)*100:.0f} artti")

    b1 = row.get("w1_b_value")
    b3 = row.get("w3_b_value")
    if b1 is not None and b3 is not None and not pd.isna(b1) and not pd.isna(b3):
        b1f = float(b1)
        b3f = float(b3)
        if b3f > b1f:
            parts.append(f"b-degeri dustu ({b3f:.2f}->{b1f:.2f})")
        else:
            parts.append(f"b-degeri artti ({b3f:.2f}->{b1f:.2f})")

    return {
        "pattern_type": pt,
        "risk_score": final,
        "risk_level": risk_level,
        "risk_class": risk_class,
        "component_scores": component_scores,
        "interpretation": " | ".join(parts),
        "confidence": confidence,
        "confidence_issues": confidence_issues,
        "quiescence_ratio": round(float(qr), 3) if qr is not None and not pd.isna(qr) else None,
    }

# =========================================================
# GENISLETILMIS BOLGE VERITABANI
# =========================================================

try:
    from zone_database import get_all_zones, compute_segment_risk
    KNOWN_ZONES = get_all_zones()
    print(f"  Bolge DB: {len(KNOWN_ZONES)} bolge yuklendi")
except Exception:
    KNOWN_ZONES = {}


def get_zone_context(lat, lon):
    best = None
    best_dist = 999999
    for key, z in KNOWN_ZONES.items():
        d = haversine_km(lat, lon, z["lat"], z["lon"])
        if d < best_dist and d < 500:
            best = dict(z)
            best["key"] = key
            best_dist = d
    return best


def compute_poisson_probability(recurrence_years, years_since_last):
    lam = 1.0 / recurrence_years
    probs = {}
    for window in [1, 5, 10, 30, 50]:
        probs[f"{window}y"] = round((1 - math.exp(-lam * window)) * 100, 1)

    ratio = years_since_last / recurrence_years
    if ratio > 0.5:
        phase = "GEC"
    elif ratio > 0.3:
        phase = "ORTA"
    else:
        phase = "ERKEN"

    return probs, phase, ratio


def build_explanation(result, feats, meta, zone):
    score = result["risk_score"]
    level = result["risk_level"]
    pt = result["pattern_type"]
    conf = result["confidence"]

    lines = []
    lines.append(f"SONUC: {level} ({score*100:.1f}%)")
    lines.append("")

    if level == "KRITIK":
        lines.append("Model cok guclu oncu benzerligi goruyor.")
    elif level == "YUKSEK":
        lines.append("Belirgin anormallik var, dikkatli izleme onerilir.")
    elif level == "ORTA":
        lines.append("Bazi oncu sinyaller mevcut, ama cok guclu degil.")
    elif level == "DIKKAT":
        lines.append("Sinirda bir durum. Anormallikler var ama tam alarm degil.")
    else:
        lines.append("Belirgin guclu oncu oruntu yok. Bu guvenli demek degildir.")

    lines.append("")
    lines.append(f"Guven skoru: {conf}")
    if result.get("confidence_issues"):
        lines.append("Guven sorunlari: " + ", ".join(result["confidence_issues"]))
    lines.append("")

    if pt == "TIP_A":
        lines.append("Sablon: AKTIVASYON")
        lines.append("Son donemde aktivite artis egiliminde.")
    elif pt == "TIP_B":
        lines.append("Sablon: SISMİK SESSIZLIK")
        lines.append("Son donemde aktivite azalmis, fay kilitleniyor olabilir.")
    else:
        lines.append("Sablon: BELIRSIZ")
        lines.append("Yeterli veya net oruntu yok.")

    lines.append("")

    qr = feats.get("quiescence_ratio")
    if qr is not None:
        if qr < 0.5:
            lines.append(f"Quiescence: son 1 yilda aktivite % {round((1-qr)*100)} azalma")
        elif qr > 1.5:
            lines.append(f"Quiescence: son 1 yilda aktivite % {round((qr-1)*100)} artis")

    b1 = feats.get("w1_b_value")
    b3 = feats.get("w3_b_value")
    if b1 is not None and b3 is not None:
        if b3 > b1:
            lines.append(f"b-degeri dususu: {b3:.2f} -> {b1:.2f}")
        else:
            lines.append(f"b-degeri yukselisi: {b3:.2f} -> {b1:.2f}")

    if zone:
        lines.append("")
        lines.append(f"Bolge: {zone['name']}")
        lines.append(f"Fay: {zone['fault_name']} ({zone['fault_type']})")
        lines.append(f"Beklenen buyukluk: Mw {zone['expected_mw']}")
        lines.append(f"Son buyuk deprem: {zone['last_major']} (Mw {zone['last_major_mw']})")

        try:
            last_dt = datetime.strptime(zone["last_major"], "%Y-%m-%d")
            years_since = (datetime.utcnow() - last_dt).days / 365.25
            probs, phase, ratio = compute_poisson_probability(
                zone["recurrence_years"], years_since
            )
            lines.append(f"Gecen sure: {years_since:.1f} yil")
            lines.append(f"Tarihsel tekrar: ~{zone['recurrence_years']} yil")
            lines.append(f"Faz: {phase} (oran={ratio:.2f})")
            lines.append("")
            lines.append("Istatistiksel olasiliklar (Poisson):")
            lines.append(f"  1 yil  : %{probs['1y']}")
            lines.append(f"  5 yil  : %{probs['5y']}")
            lines.append(f"  10 yil : %{probs['10y']}")
            lines.append(f"  30 yil : %{probs['30y']}")
        except Exception:
            pass

    lines.append("")
    lines.append("NOT: Bu bir deprem tahmini degildir. Bu sistem sadece anormal sismik oruntuleri tespit eder.")

    return "\n".join(lines)


# =========================================================
# VERI CEKME
# =========================================================

# =========================================================
# ONBELLEK
# =========================================================

# =========================================================
# QUERY CACHE
# =========================================================

_QUERY_CACHE = {}
_QUERY_CACHE_TTL = 6 * 3600  # 6 saat


def _query_cache_key(lat, lon, radius_km, min_mag, ref_date):
    return f"{lat:.3f}_{lon:.3f}_{radius_km}_{min_mag}_{ref_date or 'now'}"


def fetch_best(lat, lon, radius_km=300, min_mag=2.5, ref_date=None):
    """
    ISC+USGS hibrit, cache destekli.
    Aynı sorgu tekrar gelirse direkt RAM cache'ten dön.
    """
    import time

    key = _query_cache_key(lat, lon, radius_km, min_mag, ref_date)
    now_ts = time.time()

    # 1) RAM cache kontrol
    if key in _QUERY_CACHE:
        ts, feats, meta = _QUERY_CACHE[key]
        if ref_date or (now_ts - ts) < _QUERY_CACHE_TTL:
            meta2 = dict(meta)
            meta2["cache"] = True
            meta2["cache_age_sec"] = int(now_ts - ts)
            return feats, meta2

    # 2) ISC+USGS dene
    if HAS_ISC:
        try:
            # isc_fetch_v2 cache kullanabilsin diye sentetik event_id ver
            synthetic_id = (
                f"live_{lat:.3f}_{lon:.3f}_{radius_km}_{min_mag}_{ref_date or 'now'}"
                .replace("-", "m")
                .replace(".", "p")
            )

            feats, meta, err = isc_fetch_analyze(
                lat, lon,
                radius_km=radius_km,
                min_mag=min_mag,
                ref_date=ref_date,
                event_id=synthetic_id,
                use_cache=True
            )

            if feats is not None:
                meta = dict(meta)
                meta["cache"] = False
                _QUERY_CACHE[key] = (now_ts, feats, meta)
                return feats, meta

        except Exception as e:
            print(f"ISC hata -> fallback USGS: {e}")

    # 3) USGS fallback
    end = datetime.strptime(ref_date, "%Y-%m-%d") if ref_date else datetime.utcnow()
    start = end - timedelta(days=1095)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "orderby": "time-asc",
        "limit": 10000,
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    fs = data.get("features", [])
    if not fs:
        return None, "Veri bulunamadi"

    recs = []
    for f in fs:
        p, g = f["properties"], f["geometry"]["coordinates"]
        recs.append({
            "time": datetime.utcfromtimestamp(p["time"] / 1000),
            "magnitude": p.get("mag"),
            "depth_km": g[2],
            "lat": g[1],
            "lon": g[0],
            "source": "USGS"
        })

    df = pd.DataFrame(recs).dropna(subset=["time", "magnitude"])
    if len(df) == 0:
        return None, "Veri bulunamadi"

    df["days"] = (end - df["time"]).dt.total_seconds() / 86400.0
    w1 = df[df["days"] <= 365]
    w2 = df[(df["days"] > 365) & (df["days"] <= 730)]
    w3 = df[df["days"] > 730]
    c0, c1, c2 = len(w1), len(w2), len(w3)

    def bv(m):
        m = np.array(m)
        m = m[np.isfinite(m)]
        if len(m) < 15:
            return None
        mc = np.min(m)
        d = np.mean(m) - (mc - 0.025)
        return round(np.log10(np.e) / d, 4) if d > 0 else None

    n90 = len(w1[w1["days"] <= 90])
    n275 = len(w1[(w1["days"] > 90) & (w1["days"] <= 365)])
    accel = round((n90 / 90.0) / (n275 / 275.0), 3) if n275 > 0 else None

    feats = {
        "count_0_1y": c0,
        "count_1_2y": c1,
        "count_2_3y": c2,
        "accel_90d": accel,
        "w1_b_value": bv(w1["magnitude"].values),
        "w3_b_value": bv(df["magnitude"].values),
        "w1_max_mw": float(w1["magnitude"].max()) if len(w1) > 0 else None,
        "w1_mean_mw": float(w1["magnitude"].mean()) if len(w1) > 0 else None,
        "w3_max_mw": float(df["magnitude"].max()),
        "w3_mean_mw": float(df["magnitude"].mean()),
        "w1_mean_dist_km": 100,
        "w3_mean_dist_km": 130,
        "w1_mean_depth_km": float(w1["depth_km"].mean()) if len(w1) > 0 else None,
        "w3_mean_depth_km": float(df["depth_km"].mean()),
        "w1_n_events": c0,
        "w3_n_events": len(df),
        "monthly_slope_36m": 0,
        "w1_std_mw": float(w1["magnitude"].std()) if len(w1) > 1 else 0.5,
        "w1_std_dist_km": 50,
        "w1_std_depth_km": 10,
        "w1_migration_slope_km_day": 0,
        "w3_migration_slope_km_day": 0,
        "z_rate_1y": 0,
        "z_rate_3y": 0,
        "z_b_value_1y": 0,
        "z_b_value_3y": 0,
        "z_max_mw_1y": 0,
        "z_depth_1y": 0,
        "z_dist_1y": 0,
    }

    meta = {
        "n_total": len(df),
        "mc": round(float(df["magnitude"].min()), 1),
        "n_above_mc": len(df),
        "source": "USGS",
        "cache": False,
    }

    _QUERY_CACHE[key] = (now_ts, feats, meta)
    return feats, meta


# =========================================================
# MODEL YUKLE
# =========================================================

def load_models():
    models = {}
    feature_lists = {}

    fl_path = MODEL_DIR / "feature_lists.json"
    if not fl_path.exists():
        print("  HATA: feature_lists.json bulunamadi")
        return models, feature_lists

    with open(fl_path, "r", encoding="utf-8") as f:
        feature_lists = json.load(f)

    # Tip bazli modeller
    for pt in ["TIP_A", "TIP_B", "TIP_C"]:
        p = MODEL_DIR / f"model_{pt}.joblib"
        if p.exists():
            models[pt] = (joblib.load(p), feature_lists.get(pt, []))

    # Kalibre edilmis model
    cal_path = Path("output/calibrated_models/calibrated_model.joblib")
    cal_feat_path = Path("output/calibrated_models/calibrated_features.json")

    if cal_path.exists():
        try:
            cal_model = joblib.load(cal_path)

            if cal_feat_path.exists():
                with open(cal_feat_path, "r", encoding="utf-8") as f:
                    cal_feats = json.load(f)
            else:
                cal_feats = feature_lists.get("TIP_A", [])

            models["CALIBRATED"] = (cal_model, cal_feats)
            print(f"  Kalibre model yuklendi: ECE=0.018 | features={len(cal_feats)}")
        except Exception as e:
            print(f"  Kalibre model yuklenemedi: {e}")

    return models, feature_lists


MODELS, FL = load_models()

# =========================================================
# ROUTES
# =========================================================

from flask import send_from_directory

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        f = request.get_json(force=True)
        result = predict_risk(f, MODELS, FL)
        return jsonify(clean_nans(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/realtime", methods=["POST"])
def api_realtime():
    try:
        b = request.get_json(force=True)
        lat = float(b["lat"])
        lon = float(b["lon"])
        feats, meta = fetch_best(
            lat, lon,
            float(b.get("radius_km", 300)),
            float(b.get("min_mag", 2.5))
        )
        if feats is None:
            return jsonify({"error": meta}), 400

        result = predict_risk(feats, MODELS, FL)
        result["meta"] = meta
        result["features"] = feats
        zone = get_zone_context(lat, lon)
        result["explanation"] = build_explanation(result, feats, meta, zone)

        return jsonify(clean_nans(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/historical", methods=["POST"])
def api_historical():
    try:
        b = request.get_json(force=True)
        lat = float(b["lat"])
        lon = float(b["lon"])
        ref_date = b.get("ref_date")
        if not ref_date:
            return jsonify({"error": "ref_date gerekli"}), 400

        feats, meta = fetch_best(
            lat, lon,
            float(b.get("radius_km", 300)),
            float(b.get("min_mag", 2.5)),
            ref_date=ref_date
        )
        if feats is None:
            return jsonify({"error": meta}), 400

        result = predict_risk(feats, MODELS, FL)
        result["meta"] = meta
        result["features"] = feats
        zone = get_zone_context(lat, lon)
        result["explanation"] = build_explanation(result, feats, meta, zone)

        return jsonify(clean_nans(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/zones")
def api_zones():
    return jsonify(clean_nans(KNOWN_ZONES))


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": "ok",
        "version": "4.0",
        "models": list(MODELS.keys()),
        "auc": 0.9087,
        "isc_enabled": HAS_ISC,
        "calibrated": Path("output/calibrated_models/calibrated_model.joblib").exists(),
    })

# =========================================================
# FAY BILGISI
# =========================================================

try:
    from fault_distance import get_fault_info_for_app, get_fault_db
    _fdb = get_fault_db()
    HAS_FAULTS = len(_fdb.faults) > 0
except Exception:
    HAS_FAULTS = False
    _fdb = None


@app.route("/api/faults")
def api_faults():
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400
        if not HAS_FAULTS:
            return jsonify({"error": "Fay veritabani yuklu degil"}), 500
        info = get_fault_info_for_app(lat, lon)
        return jsonify(clean_nans(info))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# =========================================================
# COULOMB STRES
# =========================================================

try:
    from coulomb_simple import compute_cff_for_app
    HAS_CFF = True
except Exception:
    HAS_CFF = False


@app.route("/api/cff")
def api_cff():
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        ref = request.args.get("ref_date")
        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400
        if not HAS_CFF:
            return jsonify({"error": "CFF modulu yuklu degil"}), 500
        result = compute_cff_for_app(lat, lon, ref)
        return jsonify(clean_nans(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# =========================================================
# GPS VELOCITY
# =========================================================

try:
    from gps_velocity import get_gps_info_for_app, get_gps_db
    _gps_db = get_gps_db()
    HAS_GPS = len(_gps_db.stations) > 0
except Exception:
    HAS_GPS = False
    _gps_db = None


@app.route("/api/gps")
def api_gps():
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400
        if not HAS_GPS:
            return jsonify({"error": "GPS veritabani yuklu degil"}), 500
        info = get_gps_info_for_app(lat, lon)
        return jsonify(clean_nans(info))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# BIRLESIK JEODINAMIK OZET
# =========================================================

@app.route("/api/geodynamic")
def api_geodynamic():
    """
    Birlesik jeodinamik + dual risk + hazard ozeti.
    Tek sorguda tum katmanlari toplar.
    Hedef: ISC cache'li sorgularda <3 saniye.
    """
    t0 = _time.time()

    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        ref = request.args.get("ref_date")
        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400

        result = {"lat": lat, "lon": lon, "ref_date": ref}

        # 1. Sismik risk (ISC+USGS, cache'li)
        feats, meta = fetch_best(lat, lon, 300, 2.5, ref_date=ref)
        if feats:
            risk = predict_risk(feats, MODELS, FL)
            result["seismic"] = risk
            result["seismic"]["meta"] = meta
        else:
            result["seismic"] = {"error": "Veri alinamadi"}

        # 2. Quality-adjusted short-term
        if HAS_QAR and feats:
            try:
                from quality_adjusted_risk import compute_quality_adjusted_from_data
                qar = compute_quality_adjusted_from_data(
                    lat=lat, lon=lon, feats=feats, meta=meta, ref_date=ref
                )
                result["quality_adjusted"] = {
                    "st_raw": qar.get("st_raw"),
                    "st_debiased": qar.get("st_debiased"),
                    "st_adjusted": qar.get("st_adjusted"),
                    "st_adjusted_level": qar.get("st_adjusted_level"),
                    "confidence": qar.get("confidence"),
                    "data_quality": qar.get("data_quality"),
                    "tectonic_class": qar.get("tectonic_class"),
                    "quality_penalty": qar.get("quality_penalty"),
                    "tectonic_penalty": qar.get("tectonic_penalty"),
                }
            except Exception:
                pass

        # 3. Fay bilgisi (cache'li, hızlı)
        if HAS_FAULTS:
            try:
                result["faults"] = get_fault_info_for_app(lat, lon)
            except Exception:
                pass

        # 4. CFF (hesaplama, orta hız)
        if HAS_CFF:
            try:
                cff = compute_cff_for_app(lat, lon, ref)
                result["cff"] = {
                    "cff_total": cff.get("cff_total"),
                    "cff_n_sources": cff.get("cff_n_sources"),
                    "cff_nearest_source_km": cff.get("cff_nearest_source_km"),
                }
            except Exception:
                pass

        # 5. GPS (cache'li, hızlı)
        if HAS_GPS:
            try:
                gps = get_gps_info_for_app(lat, lon)
                result["gps"] = {
                    "gps_mean_vh_mm_yr": gps.get("gps_mean_vh_mm_yr"),
                    "gps_strain_rate": gps.get("gps_strain_rate"),
                    "gps_nearest_site": gps.get("gps_nearest_site"),
                    "gps_nearest_dist_km": gps.get("gps_nearest_dist_km"),
                }
            except Exception:
                pass

        # 6. NLP (cache'li, hızlı)
        if HAS_NLP:
            try:
                nlp = get_nlp_info_for_app(lat, lon)
                result["nlp"] = nlp.get("summary")
            except Exception:
                pass

        # 7. Bölge bağlamı
        zone = get_zone_context(lat, lon)
        if zone:
            result["zone"] = {
                "name": zone.get("name"),
                "fault_name": zone.get("fault_name"),
                "expected_mw": zone.get("expected_mw"),
                "last_major": zone.get("last_major"),
                "last_major_mw": zone.get("last_major_mw"),
                "population_risk": zone.get("population_risk"),
            }

            try:
                last_dt = datetime.strptime(zone["last_major"], "%Y-%m-%d")
                yrs = (datetime.utcnow() - last_dt).days / 365.25
                probs, phase, ratio = compute_poisson_probability(
                    zone["recurrence_years"], yrs
                )
                result["zone"]["years_since"] = round(yrs, 1)
                result["zone"]["phase"] = phase
                result["zone"]["poisson"] = probs
            except Exception:
                pass

        # 8. Hazard skorları (hesaplama, hızlı)
        if HAS_HAZARD_API and zone:
            try:
                from survival_hazard_model import compute_long_term_score, compute_hazard_scores
                lt = compute_long_term_score(zone, lat=lat, lon=lon)
                result["long_term"] = {
                    "score": lt.get("score"),
                    "level": lt.get("level"),
                    "factors": lt.get("factors", [])[:4],
                }

                st_for_hz = result.get("quality_adjusted", {"st_adjusted": 0, "confidence": "ORTA"})
                hz = compute_hazard_scores(
                    zone=zone,
                    long_term=lt,
                    short_term=st_for_hz,
                    cff_info=result.get("cff"),
                    nlp_info=result.get("nlp"),
                )
                result["hazards"] = {
                    "30d": hz.get("30d"),
                    "90d": hz.get("90d"),
                    "1y": hz.get("1y"),
                    "5y": hz.get("5y"),
                    "multiplier": hz.get("total_multiplier"),
                }
            except Exception:
                pass

        # 9. Açıklama
        result["explanation"] = build_explanation(
            result.get("seismic", {}),
            feats or {},
            meta or {},
            zone
        )

        # Süre
        elapsed = round(_time.time() - t0, 2)
        result["elapsed_seconds"] = elapsed

        return jsonify(clean_nans(result))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def build_geodynamic_summary(data):
    """Tum katmanlardan birlesik yorum olustur."""
    lines = []
    risk_factors = 0
    total_factors = 0

    # Sismik
    seismic = data.get("seismic", {})
    if not seismic.get("error"):
        score = seismic.get("risk_score", 0)
        level = seismic.get("risk_level", "?")
        pt = seismic.get("pattern_type", "?")
        lines.append(f"SISMIK ANALIZ: {level} ({score*100:.1f}%)")
        lines.append(f"  Sablon: {pt}")
        if seismic.get("interpretation"):
            lines.append(f"  {seismic['interpretation']}")
        total_factors += 1
        if score >= 0.30:
            risk_factors += 1

    # Fay
    faults = data.get("faults", {})
    ff = faults.get("features", {})
    if ff and not faults.get("error"):
        dist = ff.get("nearest_fault_dist_km")
        name = ff.get("nearest_fault_name", "?")
        ftype = ff.get("nearest_fault_type", "?")
        n300 = ff.get("n_faults_within_300km", 0)
        lines.append("")
        lines.append(f"FAY BILGISI:")
        lines.append(f"  En yakin fay: {name} ({dist} km)")
        lines.append(f"  Fay tipi: {ftype}")
        lines.append(f"  300 km icinde {n300} fay segmenti")
        total_factors += 1
        if dist is not None and dist < 30:
            risk_factors += 1
            lines.append(f"  UYARI: Fay cok yakin ({dist} km)")

    # CFF
    cff = data.get("coulomb", {})
    if not cff.get("error"):
        cff_total = cff.get("cff_total", 0)
        n_src = cff.get("cff_n_sources", 0)
        lines.append("")
        lines.append(f"COULOMB STRES:")
        lines.append(f"  Toplam CFF: {cff_total:.6f} bar ({n_src} kaynak)")
        total_factors += 1
        if cff_total > 0.1:
            risk_factors += 1
            lines.append(f"  UYARI: Yuksek stres transferi ({cff_total:.4f} bar)")
        elif cff_total > 0.01:
            lines.append(f"  Orta seviye stres transferi")
        else:
            lines.append(f"  Dusuk stres transferi")

    # GPS
    gps = data.get("gps", {})
    if not gps.get("error") and gps:
        vh = gps.get("gps_mean_vh_mm_yr")
        strain = gps.get("gps_strain_rate")
        n_st = gps.get("gps_n_stations", 0)
        nearest = gps.get("gps_nearest_site", "?")
        nearest_d = gps.get("gps_nearest_dist_km")
        lines.append("")
        lines.append(f"GPS DEFORMASYON:")
        lines.append(f"  {n_st} istasyon, en yakin: {nearest} ({nearest_d} km)")
        if vh is not None:
            lines.append(f"  Ort. yatay hiz: {vh} mm/yil")
        if strain is not None:
            lines.append(f"  Strain rate: {strain}")
        total_factors += 1
        if vh is not None and vh > 25:
            risk_factors += 1
            lines.append(f"  UYARI: Yuksek deformasyon hizi ({vh} mm/yil)")

    # Bolge
    zone = data.get("zone", {})
    if zone:
        lines.append("")
        lines.append(f"BOLGE BAGLAMI:")
        lines.append(f"  {zone.get('name', '?')}")
        lines.append(f"  Son buyuk deprem: {zone.get('last_major','?')} "
                     f"(Mw {zone.get('last_major_mw','?')})")
        ys = zone.get("years_since")
        if ys:
            lines.append(f"  Gecen sure: {ys} yil")
            lines.append(f"  Faz: {zone.get('phase','?')} "
                         f"(oran: {zone.get('cycle_ratio','?')})")
        poisson = zone.get("poisson", {})
        if poisson:
            lines.append(f"  Poisson olasiliklari:")
            for k, v in poisson.items():
                lines.append(f"    {k}: %{v}")

    # NLP tarihsel
    if HAS_NLP:
        try:
            nlp = get_nlp_info_for_app(data.get("lat", 0), data.get("lon", 0))
            hist = nlp.get("historical", [])
            if hist:
                lines.append("")
                lines.append("TARIHSEL ONCU KAYITLARI:")
                for h in hist[:3]:
                    lines.append(f"  {h['event']} ({h['distance_km']} km)")
                    for p in h["precursors"]:
                        lines.append(f"    [{p['type']}] {p['detail']}")
                    lines.append(f"    Kaynak: {h['source']}")
        except Exception:
            pass

    # Genel degerlendirme
    lines.append("")
    lines.append("=" * 40)
    lines.append("GENEL DEGERLENDIRME:")

    if total_factors == 0:
        lines.append("  Yetersiz veri")
    else:
        risk_pct = risk_factors / total_factors * 100
        if risk_pct >= 75:
            lines.append(f"  YUKSEK ONCELIK: {risk_factors}/{total_factors} "
                        f"katmanda risk sinyali")
        elif risk_pct >= 50:
            lines.append(f"  ORTA ONCELIK: {risk_factors}/{total_factors} "
                        f"katmanda risk sinyali")
        elif risk_pct >= 25:
            lines.append(f"  DIKKAT: {risk_factors}/{total_factors} "
                        f"katmanda risk sinyali")
        else:
            lines.append(f"  DUSUK: {risk_factors}/{total_factors} "
                        f"katmanda risk sinyali")

    lines.append("")
    lines.append("NOT: Bu bir deprem tahmini degildir.")

    return {
        "text": "\n".join(lines),
        "risk_factors": risk_factors,
        "total_factors": total_factors,
        "risk_ratio": round(risk_factors / max(total_factors, 1), 2),
    }

# =========================================================
# NLP TARAMA
# =========================================================

try:
    from nlp_scanner import get_nlp_info_for_app
    HAS_NLP = True
except Exception:
    HAS_NLP = False


@app.route("/api/nlp")
def api_nlp():
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400
        if not HAS_NLP:
            return jsonify({"error": "NLP modulu yuklu degil"}), 500
        info = get_nlp_info_for_app(lat, lon)
        return jsonify(clean_nans(info))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# =========================================================
# HARITA
# =========================================================

@app.route("/map")
def serve_map():
    map_path = Path("output/seismo_map.html")
    if not map_path.exists():
        # Haritayi olustur
        try:
            import subprocess
            subprocess.run([sys.executable, "scripts/build_map.py"],
                          capture_output=True)
        except Exception:
            pass

    if map_path.exists():
        return map_path.read_text(encoding="utf-8")
    return "Harita olusturulamadi. Once auto_monitor.py --once calistirin.", 404


@app.route("/api/build_map", methods=["POST"])
def api_build_map():
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/build_map.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return jsonify({"success": True, "output": result.stdout})
        return jsonify({"error": result.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/monitor_data")
def api_monitor_data():
    """Harita icin monitor verisi."""
    try:
        db_path = Path("output/monitor.db")
        if not db_path.exists():
            return jsonify([])
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("""
            SELECT zone_key, zone_name, lat, lon,
                   risk_score, risk_level, pattern_type,
                   n_events, b_value, quiescence, cff_total, gps_vh,
                   source, timestamp
            FROM scan_results
            WHERE id IN (SELECT MAX(id) FROM scan_results GROUP BY zone_key)
            ORDER BY risk_score DESC
        """).fetchall()
        conn.close()
        return jsonify([{
            "key":r[0],"name":r[1],"lat":r[2],"lon":r[3],
            "score":r[4],"level":r[5],"pattern":r[6],
            "n_events":r[7],"b_value":r[8],"quiescence":r[9],
            "cff":r[10],"gps_vh":r[11],"source":r[12],"timestamp":r[13]
        } for r in rows])
    except Exception as e:
        return jsonify([])


@app.route("/api/scored_events")
def api_scored_events():
    """Harita icin tarihsel skorlu olaylar."""
    try:
        path = Path("output/real_risk_scored.csv")
        if not path.exists():
            return jsonify([])
        df = pd.read_csv(path, low_memory=False)
        if "risk_score" not in df.columns:
            return jsonify([])
        lat_col = next((c for c in ["lat","main_lat"] if c in df.columns), None)
        lon_col = next((c for c in ["lon","main_lon"] if c in df.columns), None)
        mw_col = next((c for c in ["mw","main_mw"] if c in df.columns), None)
        dt_col = next((c for c in ["datetime_utc","main_datetime_utc"] if c in df.columns), None)
        if not lat_col or not lon_col:
            return jsonify([])
        result = []
        for _, row in df.nlargest(200, "risk_score").iterrows():
            lat = row.get(lat_col)
            lon = row.get(lon_col)
            if pd.isna(lat) or pd.isna(lon): continue
            result.append({
                "lat":float(lat),"lon":float(lon),
                "score":float(row.get("risk_score",0)),
                "mw":float(row.get(mw_col,0)) if mw_col else 0,
                "date":str(row.get(dt_col,""))[:10] if dt_col else "",
            })
        return jsonify(result)
    except Exception:
        return jsonify([])
    
# =========================================================
# UYARI SISTEMI
# =========================================================

try:
    from alert_system import get_alerts_for_app, check_alerts as run_alert_check
    HAS_ALERTS = True
except Exception:
    HAS_ALERTS = False


@app.route("/api/alerts")
def api_alerts():
    try:
        if not HAS_ALERTS:
            return jsonify({"error": "Uyari modulu yuklu degil"}), 500
        return jsonify(clean_nans(get_alerts_for_app()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts/check", methods=["POST"])
def api_alerts_check():
    try:
        if not HAS_ALERTS:
            return jsonify({"error": "Uyari modulu yuklu degil"}), 500
        alerts = run_alert_check()
        return jsonify(clean_nans({
            "n_alerts": len(alerts),
            "alerts": alerts
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# =========================================================
# PDF RAPOR
# =========================================================

try:
    from pdf_report import generate_pdf_for_app
    HAS_PDF = True
except Exception:
    HAS_PDF = False


@app.route("/api/pdf")
def api_pdf():
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        ref = request.args.get("ref_date")
        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400
        if not HAS_PDF:
            return jsonify({"error": "PDF modulu yuklu degil"}), 500

        # Cache'li veriyi kullan (tekrar ISC cekme)
        feats, meta = fetch_best(lat, lon, 300, 2.5, ref_date=ref)

        path = generate_pdf_for_app(lat, lon, ref)
        return send_from_directory(
            str(Path(path).parent),
            Path(path).name,
            as_attachment=True
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# =========================================================
# API DOKUMANTASYONU / SWAGGER
# =========================================================

def build_openapi_spec():
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "SeismoPattern API",
            "version": "4.5.0",
            "description": (
                "Deprem oncu sablon analiz sistemi API'si. "
                "ISC+USGS sismik veri, fay bilgisi, Coulomb stres, GPS deformasyon "
                "ve NLP tarihsel kayitlarini birlestirir."
            ),
        },
        "servers": [
            {"url": "http://127.0.0.1:5000", "description": "Local Flask server"}
        ],
        "paths": {
            "/api/status": {
                "get": {
                    "summary": "Sistem durumu",
                    "responses": {
                        "200": {"description": "API durumu"}
                    }
                }
            },
            "/api/predict": {
                "post": {
                    "summary": "Manuel risk tahmini",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "count_0_1y": {"type": "number"},
                                        "count_1_2y": {"type": "number"},
                                        "count_2_3y": {"type": "number"},
                                        "accel_90d": {"type": "number"},
                                        "w1_b_value": {"type": "number"},
                                        "w3_b_value": {"type": "number"},
                                        "w1_max_mw": {"type": "number"},
                                        "w1_mean_mw": {"type": "number"},
                                        "w1_mean_dist_km": {"type": "number"},
                                        "w3_mean_dist_km": {"type": "number"},
                                        "w1_mean_depth_km": {"type": "number"},
                                        "w3_mean_depth_km": {"type": "number"},
                                    }
                                },
                                "example": {
                                    "count_0_1y": 45,
                                    "count_1_2y": 20,
                                    "count_2_3y": 15,
                                    "accel_90d": 4.2,
                                    "w1_b_value": 0.72,
                                    "w3_b_value": 0.91,
                                    "w1_max_mw": 6.8,
                                    "w1_mean_mw": 5.8,
                                    "w1_mean_dist_km": 95,
                                    "w3_mean_dist_km": 140,
                                    "w1_mean_depth_km": 25,
                                    "w3_mean_depth_km": 32
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Risk tahmin sonucu"}
                    }
                }
            },
            "/api/realtime": {
                "post": {
                    "summary": "Canli ISC+USGS analiz",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "example": {
                                    "lat": 40.77,
                                    "lon": 29.00,
                                    "radius_km": 300,
                                    "min_mag": 2.5
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Canli analiz sonucu"}
                    }
                }
            },
            "/api/historical": {
                "post": {
                    "summary": "Tarihsel analiz",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "example": {
                                    "lat": 38.30,
                                    "lon": 142.37,
                                    "radius_km": 300,
                                    "min_mag": 2.5,
                                    "ref_date": "2011-02-11"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Tarihsel analiz sonucu"}
                    }
                }
            },
            "/api/geodynamic": {
                "get": {
                    "summary": "Birlesik jeodinamik ozet",
                    "parameters": [
                        {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "ref_date", "in": "query", "required": False, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Sismik + Fay + CFF + GPS + NLP birlesik ozet"}
                    }
                }
            },
            "/api/faults": {
                "get": {
                    "summary": "Fay bilgisi",
                    "parameters": [
                        {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}}
                    ],
                    "responses": {
                        "200": {"description": "En yakin faylar ve fay feature'lari"}
                    }
                }
            },
            "/api/cff": {
                "get": {
                    "summary": "Coulomb stres ozeti",
                    "parameters": [
                        {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "ref_date", "in": "query", "required": False, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "Basitlestirilmis CFF sonucu"}
                    }
                }
            },
            "/api/gps": {
                "get": {
                    "summary": "GPS velocity / strain bilgisi",
                    "parameters": [
                        {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}}
                    ],
                    "responses": {
                        "200": {"description": "Yakin GPS istasyonlari ve deformasyon ozeti"}
                    }
                }
            },
            "/api/nlp": {
                "get": {
                    "summary": "Tarihsel NLP precursor kayitlari",
                    "parameters": [
                        {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}}
                    ],
                    "responses": {
                        "200": {"description": "Yakin tarihsel oncu kayitlari"}
                    }
                }
            },
            "/api/zones": {
                "get": {
                    "summary": "Bilinen riskli bolgeler veritabani",
                    "responses": {
                        "200": {"description": "25 bolgelik genisletilmis veritabani"}
                    }
                }
            },
            "/api/alerts": {
                "get": {
                    "summary": "Son uyarilar",
                    "responses": {
                        "200": {"description": "Son 7 ve 30 gun uyarilari"}
                    }
                }
            },
            "/api/alerts/check": {
                "post": {
                    "summary": "Uyari kontrolu calistir",
                    "responses": {
                        "200": {"description": "Uretilen yeni uyarilar"}
                    }
                }
            },
            "/api/monitor_data": {
                "get": {
                    "summary": "Harita icin monitor verisi",
                    "responses": {
                        "200": {"description": "Son tarama skorlarinin listesi"}
                    }
                }
            },
            "/api/scored_events": {
                "get": {
                    "summary": "Harita icin tarihsel skorlu olaylar",
                    "responses": {
                        "200": {"description": "Tarihsel buyuk deprem skor listesi"}
                    }
                }
            },
            "/api/pdf": {
                "get": {
                    "summary": "PDF rapor indir",
                    "parameters": [
                        {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}},
                        {"name": "ref_date", "in": "query", "required": False, "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {"description": "PDF dosyasi"}
                    }
                }
            }
        }
    }


@app.route("/api/openapi.json")
def api_openapi():
    return jsonify(build_openapi_spec())


@app.route("/docs")
def api_docs():
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>SeismoPattern API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <style>body{margin:0;}</style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function() {
      SwaggerUIBundle({
        url: "/api/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>"""

# =========================================================
# UNCERTAINTY API
# =========================================================

# UNCERTAINTY API
# =========================================================

@app.route("/api/uncertainty")
def api_uncertainty():
    """
    Gercek bootstrap ensemble ile tahmin belirsizligi hesapla.
    """
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))

        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400

        feats, meta = fetch_best(lat, lon, 300, 2.5)
        if feats is None:
            return jsonify({"error": "Veri alinamadi"}), 400

        # Gercek bootstrap belirsizligi
        try:
            from bootstrap_uncertainty import predict_with_uncertainty
        except Exception as e:
            return jsonify({
                "error": f"bootstrap_uncertainty import hatasi: {e}"
            }), 500

        result = predict_with_uncertainty(feats)

        if "error" in result:
            return jsonify(clean_nans(result)), 500

        # Meta bilgi ekle
        result["lat"] = lat
        result["lon"] = lon
        result["n_events"] = meta.get("n_total")

        raw_source = meta.get("source")
        cache_flag = bool(meta.get("cache", False))

        # Tutarlilik: source="cache" ise cache de True olmali
        if isinstance(raw_source, str) and raw_source.strip().lower() == "cache":
            cache_flag = True

        result["source"] = raw_source
        result["cache"] = cache_flag

        return jsonify(clean_nans(result))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# SHAP API
# =========================================================

@app.route("/api/shap")
def api_shap():
    try:
        shap_path = Path("output/shap_analysis/shap_importance.csv")
        if not shap_path.exists():
            return jsonify({"error": "SHAP analizi bulunamadi"}), 404

        df = pd.read_csv(shap_path)

        # İlk 20 feature'ı döndür
        return jsonify(clean_nans(df.head(20).to_dict("records")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# DUAL RISK API
# =========================================================

try:
    from dual_risk_framework import analyze_zone as dr_analyze_zone
    HAS_DUAL_RISK = True
except Exception:
    HAS_DUAL_RISK = False


@app.route("/api/dual_risk")
def api_dual_risk():
    try:
        zone_key = request.args.get("zone")
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        ref = request.args.get("ref_date")

        if not HAS_DUAL_RISK:
            return jsonify({"error": "Dual risk modulu yuklu degil"}), 500

        if zone_key:
            result = dr_analyze_zone(zone_key=zone_key, ref_date=ref)
        elif lat and lon:
            result = dr_analyze_zone(lat=float(lat), lon=float(lon),
                                     ref_date=ref)
        else:
            return jsonify({"error": "zone veya lat/lon gerekli"}), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dual_risk_table")
def api_dual_risk_table():
    """Tüm bölgeler için dual risk tablosu."""
    try:
        csv_path = Path("output/dual_risk/dual_risk_table.csv")
        if not csv_path.exists():
            return jsonify({"error": "Tablo bulunamadi. --all calistirin."}), 404
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.sort_values("combined_score", ascending=False)
        return jsonify(clean_nans(df.head(25).to_dict("records")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# DEBIASED MODEL API
# =========================================================

_DEBIASED_MODEL = None
_DEBIASED_FEATS = None

deb_path = Path("output/counterfactual/debiased_model.joblib")
deb_feat_path = Path("output/counterfactual/debiased_features.json")

if deb_path.exists() and deb_feat_path.exists():
    try:
        _DEBIASED_MODEL = joblib.load(deb_path)
        with open(deb_feat_path) as f:
            _DEBIASED_FEATS = json.load(f)
        print(f"  Debiased model yuklendi: {len(_DEBIASED_FEATS)} feature")
    except Exception as e:
        print(f"  Debiased model yuklenemedi: {e}")


@app.route("/api/debiased_predict")
def api_debiased():
    """Magnitude-free tahmin (bolge imzasi azaltilmis)."""
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400
        if _DEBIASED_MODEL is None:
            return jsonify({"error": "Debiased model yuklu degil"}), 500

        feats, meta = fetch_best(lat, lon, 300, 2.5)
        if feats is None:
            return jsonify({"error": meta}), 400

        row = add_derived(feats)
        x = pd.DataFrame([{f: row.get(f, np.nan) for f in _DEBIASED_FEATS}])
        score = float(_DEBIASED_MODEL.predict_proba(x)[0, 1])

        return jsonify(clean_nans({
            "debiased_score": round(score, 4),
            "debiased_level": risk_level(score) if score >= 0.75 else
                             "YUKSEK" if score >= 0.60 else
                             "ORTA" if score >= 0.45 else
                             "DIKKAT" if score >= 0.30 else "DUSUK",
            "note": "Magnitude feature'lar cikarilmis, bolge imzasi azaltilmis model",
            "meta": meta,
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# QUALITY ADJUSTED RISK API
# =========================================================

try:
    from quality_adjusted_risk import get_adjusted_risk_for_app
    HAS_QAR = True
    print("  Quality-adjusted risk modulu yuklendi")
except Exception:
    HAS_QAR = False


@app.route("/api/quality_risk")
def api_quality_risk():
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        ref = request.args.get("ref_date")
        lt = request.args.get("lt_score")
        lt = float(lt) if lt else None

        if lat == 0 and lon == 0:
            return jsonify({"error": "lat ve lon gerekli"}), 400
        if not HAS_QAR:
            return jsonify({"error": "Quality risk modulu yuklu degil"}), 500

        result = get_adjusted_risk_for_app(lat, lon, ref, lt)
        return jsonify(clean_nans(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# SURVIVAL HAZARD API
# =========================================================

try:
    from survival_hazard_model import analyze_hazard
    HAS_HAZARD_API = True
    print("  Survival hazard modulu yuklendi")
except Exception:
    HAS_HAZARD_API = False


@app.route("/api/hazard")
def api_hazard():
    try:
        zone_key = request.args.get("zone")
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        ref = request.args.get("ref_date")

        if not HAS_HAZARD_API:
            return jsonify({"error": "Hazard modulu yuklu degil"}), 500

        if zone_key:
            result = analyze_hazard(zone_key=zone_key, ref_date=ref)
        elif lat and lon:
            result = analyze_hazard(lat=float(lat), lon=float(lon), ref_date=ref)
        else:
            return jsonify({"error": "zone veya lat/lon gerekli"}), 400

        # Sadece ozet dondur (tam JSON cok buyuk)
        hz = result.get("hazards", {})
        lt = result.get("long_term", {})
        st = result.get("short_term", {})
        z = result.get("zone", {})

        return jsonify(clean_nans({
            "zone": z.get("name"),
            "long_term": {
                "score": lt.get("score"),
                "level": lt.get("level"),
                "factors": lt.get("factors", [])[:4],
            },
            "short_term": {
                "st_raw": st.get("st_raw"),
                "st_debiased": st.get("st_debiased"),
                "st_adjusted": st.get("st_adjusted"),
                "st_adjusted_level": st.get("st_adjusted_level"),
                "confidence": st.get("confidence"),
                "tectonic_class": st.get("tectonic_class"),
            },
            "hazards": {
                "30d": hz.get("30d"),
                "90d": hz.get("90d"),
                "1y": hz.get("1y"),
                "5y": hz.get("5y"),
                "total_multiplier": hz.get("total_multiplier"),
            },
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hazard_table")
def api_hazard_table():
    """Tum bolgeler icin hazard tablosu (CSV'den)."""
    try:
        csv_path = Path("output/survival_hazard/hazard_table.csv")
        if not csv_path.exists():
            return jsonify({"error": "Tablo yok. --all calistirin."}), 404
        df = pd.read_csv(csv_path)
        df = df.sort_values("1y", ascending=False)
        return jsonify(clean_nans(df.head(25).to_dict("records")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 50)
    print("SeismoPattern v4")
    print(f"ISC aktif: {HAS_ISC}")
    print(f"Modeller : {list(MODELS.keys())}")
    print("http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)