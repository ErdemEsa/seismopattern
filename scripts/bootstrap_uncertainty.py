#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Bootstrap Uncertainty Estimator
================================================
Gercek bootstrap ensemble ile belirsizlik hesaplar.

Egitim:
    python scripts/bootstrap_uncertainty.py --train --n 30

Tahmin (tek row dict):
    from scripts.bootstrap_uncertainty import predict_with_uncertainty
    result = predict_with_uncertainty(features_dict)

Cikti:
    {
        "mean": 0.61,
        "std": 0.07,
        "ci_lower": 0.48,
        "ci_upper": 0.74,
        "n_models": 30,
        "method": "bootstrap_ensemble"
    }
"""

import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
ROOT = Path(".")
MODEL_DIR = ROOT / "output" / "uncertainty_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REAL_CSV  = ROOT / "output" / "real_normalized.csv"
CTRL_CSV  = ROOT / "output" / "ctrl_normalized.csv"
FEAT_JSON = ROOT / "output" / "models" / "feature_lists.json"
MANIFEST  = MODEL_DIR / "bootstrap_manifest.json"

TIPS = ["TIP_A", "TIP_B", "TIP_C"]


# -------------------------------------------------------
# Yardimci: phase9 ile ayni mantik
# -------------------------------------------------------
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c0 = df.get("count_0_1y", pd.Series(0, index=df.index)).fillna(0)
    c1 = df.get("count_1_2y", pd.Series(0, index=df.index)).fillna(0)
    c2 = df.get("count_2_3y", pd.Series(0, index=df.index)).fillna(0)
    df["count_linear_trend"]  = c0 - c2
    df["count_accel_ratio"]   = c0 / ((c1 + c2) / 2.0 + 1e-6)
    b1  = df.get("w1_b_value",       pd.Series(np.nan, index=df.index))
    b3  = df.get("w3_b_value",       pd.Series(np.nan, index=df.index))
    d1  = df.get("w1_mean_dist_km",  pd.Series(np.nan, index=df.index))
    d3  = df.get("w3_mean_dist_km",  pd.Series(np.nan, index=df.index))
    dep1= df.get("w1_mean_depth_km", pd.Series(np.nan, index=df.index))
    dep3= df.get("w3_mean_depth_km", pd.Series(np.nan, index=df.index))
    df["b_drop_w3_w1"]        = b3  - b1
    df["spatial_focus_change"]= d3  - d1
    df["depth_change_km"]     = dep1 - dep3
    return df


def rule_based_type(row: dict) -> str:
    qr  = row.get("quiescence_ratio", np.nan)
    acc = row.get("accel_90d", np.nan)
    n3  = row.get("w3_n_events", 0) or 0
    if n3 < 3 or (isinstance(qr, float) and np.isnan(qr)):
        return "TIP_C"
    if qr < 0.5:
        return "TIP_B"
    if qr >= 1.0:
        return "TIP_A"
    if 0.5 <= qr < 0.8:
        return "TIP_A" if (not np.isnan(float(acc or np.nan)) and float(acc) >= 1.5) else "TIP_B"
    return "TIP_A"


def make_pipe(pos_weight: float = 1.0) -> Pipeline:
    if HAS_XGB:
        mdl = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr", verbosity=0, random_state=None
        )
    elif True:
        mdl = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            class_weight={0: 1, 1: max(1, int(pos_weight))},
            max_features="sqrt", random_state=None
        )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", mdl),
    ])


# -------------------------------------------------------
# Egitim
# -------------------------------------------------------
def train_bootstrap(n_bootstrap: int = 30, random_seed: int = 0):
    """
    N bootstrap sample ile model ensemble egit.
    """
    if not REAL_CSV.exists() or not CTRL_CSV.exists():
        raise FileNotFoundError(f"Normalizasyon CSV bulunamadi: {REAL_CSV}, {CTRL_CSV}")

    if not FEAT_JSON.exists():
        raise FileNotFoundError(f"Feature listesi bulunamadi: {FEAT_JSON}")

    real = pd.read_csv(REAL_CSV)
    ctrl = pd.read_csv(CTRL_CSV)

    real = add_derived(real)
    ctrl = add_derived(ctrl)

    real["pattern_type"] = real.apply(
        lambda r: rule_based_type(r.to_dict()), axis=1
    )
    real["target"] = 1
    ctrl["target"] = 0

    with open(FEAT_JSON, encoding="utf-8") as f:
        feat_lists = json.load(f)

    avail_r = set(real.columns)
    avail_c = set(ctrl.columns)
    common_all = avail_r & avail_c

    tip_feats = {}
    for tip in TIPS:
        fl = feat_lists.get(tip, [])
        feats = [f for f in fl if f in common_all]
        tip_feats[tip] = feats

    manifest = {
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
        "tip_feats": tip_feats,
        "models": {}
    }

    rng = np.random.RandomState(random_seed)

    print(f"Bootstrap ensemble egitimi: {n_bootstrap} sample x {len(TIPS)} tip")
    print()

    for i in range(n_bootstrap):
        seed_i = int(rng.randint(0, 2**31))
        np.random.seed(seed_i)

        for tip in TIPS:
            feats = tip_feats[tip]
            if not feats:
                continue

            real_pt = real[real["pattern_type"] == tip].copy()

            if len(real_pt) < 5:
                continue

            # Bootstrap resample: geri koyarak ornekleme
            real_bs = real_pt.sample(n=len(real_pt), replace=True, random_state=seed_i)
            ctrl_bs = ctrl.sample(n=min(len(ctrl), len(real_bs) * 3), replace=True, random_state=seed_i)

            combined = pd.concat([
                real_bs[feats + ["target"]],
                ctrl_bs[feats + ["target"]],
            ], ignore_index=True)

            X = combined[feats]
            y = combined["target"]

            if y.sum() < 5:
                continue

            pw = float((y == 0).sum()) / max(float(y.sum()), 1.0)
            pipe = make_pipe(pw)

            try:
                pipe.fit(X, y)
            except Exception as e:
                print(f"  [{i:02d}/{tip}] HATA: {e}")
                continue

            if HAS_JOBLIB:
                fname = f"bootstrap_{tip}_{i:03d}.joblib"
                fpath = MODEL_DIR / fname
                joblib.dump(pipe, fpath)

                if tip not in manifest["models"]:
                    manifest["models"][tip] = []
                manifest["models"][tip].append({
                    "index": i,
                    "seed": seed_i,
                    "file": fname,
                    "n_real": int(y.sum()),
                    "n_ctrl": int((y == 0).sum()),
                    "n_feats": len(feats),
                })

        if (i + 1) % 5 == 0:
            pct = (i + 1) / n_bootstrap * 100
            print(f"  {i+1}/{n_bootstrap} ({pct:.0f}%) tamamlandi...")

    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print()
    n_saved = sum(len(v) for v in manifest["models"].values())
    print(f"Tamamlandi: {n_saved} model kaydedildi -> {MODEL_DIR}")
    print(f"Manifest: {MANIFEST}")

    for tip in TIPS:
        cnt = len(manifest["models"].get(tip, []))
        print(f"  {tip}: {cnt} model")

    return manifest


# -------------------------------------------------------
# Tahmin
# -------------------------------------------------------
_cache = {}


def _load_bootstrap_models():
    """
    Bootstrap modellerini yukle. Cache'lenir.
    """
    global _cache
    if _cache:
        return _cache

    if not MANIFEST.exists():
        return {}

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tip_feats = manifest.get("tip_feats", {})

    loaded = {}
    for tip, model_list in manifest.get("models", {}).items():
        loaded[tip] = {
            "feats": tip_feats.get(tip, []),
            "pipes": []
        }
        for m in model_list:
            fpath = MODEL_DIR / m["file"]
            if fpath.exists() and HAS_JOBLIB:
                try:
                    pipe = joblib.load(fpath)
                    loaded[tip]["pipes"].append(pipe)
                except Exception:
                    pass

    _cache = loaded
    total = sum(len(v["pipes"]) for v in loaded.values())
    return loaded


def predict_with_uncertainty(features: dict) -> dict:
    """
    Bootstrap ensemble ile belirsizlik tahmin et.

    Yeni mantik:
      - once birincil tip belirlenir
      - yalnizca o tipe ait bootstrap modeller kullanilir
      - boylece epistemik uncertainty daha temiz olculur
    """
    models = _load_bootstrap_models()

    if not models:
        return {
            "error": "Bootstrap modeller bulunamadi. Once --train calistirin.",
            "method": "bootstrap_ensemble"
        }

    # Derived features ekle
    row = features.copy()
    c0 = float(row.get("count_0_1y", 0) or 0)
    c1 = float(row.get("count_1_2y", 0) or 0)
    c2 = float(row.get("count_2_3y", 0) or 0)

    row["count_linear_trend"]   = c0 - c2
    row["count_accel_ratio"]    = c0 / ((c1 + c2) / 2.0 + 1e-6)
    row["b_drop_w3_w1"]         = (
        float(row.get("w3_b_value") or np.nan) -
        float(row.get("w1_b_value") or np.nan)
    )
    row["spatial_focus_change"] = (
        float(row.get("w3_mean_dist_km") or np.nan) -
        float(row.get("w1_mean_dist_km") or np.nan)
    )
    row["depth_change_km"]      = (
        float(row.get("w1_mean_depth_km") or np.nan) -
        float(row.get("w3_mean_depth_km") or np.nan)
    )

    pt = rule_based_type(row)

    if pt not in models or not models[pt]["pipes"]:
        return {
            "error": f"{pt} icin bootstrap model bulunamadi",
            "pattern_type": pt,
            "method": "bootstrap_ensemble"
        }

    feats = models[pt]["feats"]
    pipes = models[pt]["pipes"]

    scores = []
    for pipe in pipes:
        try:
            x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats}])
            s = float(pipe.predict_proba(x)[0, 1])
            scores.append(s)
        except Exception:
            pass

    if not scores:
        return {
            "error": "Skor hesaplanamadi",
            "pattern_type": pt,
            "method": "bootstrap_ensemble"
        }

    scores_arr = np.array(scores, dtype=float)
    mean_s = float(np.mean(scores_arr))
    std_s  = float(np.std(scores_arr))
    ci_lo  = float(np.percentile(scores_arr, 2.5))
    ci_hi  = float(np.percentile(scores_arr, 97.5))

    # Tipler arasi farki sadece diagnostik olarak ekleyelim
    cross_type_means = {}
    for tip, data in models.items():
        feats_t = data["feats"]
        pipes_t = data["pipes"]
        vals = []
        for pipe in pipes_t[:5]:   # hizli tanisal ozet
            try:
                x = pd.DataFrame([{f: row.get(f, np.nan) for f in feats_t}])
                vals.append(float(pipe.predict_proba(x)[0, 1]))
            except Exception:
                pass
        if vals:
            cross_type_means[tip] = round(float(np.mean(vals)), 4)

    return {
        "mean":         round(mean_s, 4),
        "std":          round(std_s, 4),
        "ci_lower":     round(ci_lo, 4),
        "ci_upper":     round(ci_hi, 4),
        "n_models":     len(scores),
        "pattern_type": pt,
        "method":       "bootstrap_ensemble",
        "cross_type_means": cross_type_means,
    }


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Bootstrap Uncertainty Estimator")
    ap.add_argument("--train", action="store_true",
                    help="Bootstrap ensemble egit")
    ap.add_argument("--n", type=int, default=30,
                    help="Bootstrap sample sayisi (varsayilan: 30)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    ap.add_argument("--test", action="store_true",
                    help="Egitim sonrasi ornek tahmin yap")
    args = ap.parse_args()

    if args.train:
        manifest = train_bootstrap(n_bootstrap=args.n, random_seed=args.seed)

        if args.test:
            print("\nOrnek tahmin testi:")
            test_features = {
                "count_0_1y": 15, "count_1_2y": 8, "count_2_3y": 6,
                "quiescence_ratio": 1.875, "accel_90d": 3.2,
                "w1_b_value": 0.72, "w3_b_value": 0.85,
                "w1_n_events": 15, "w3_n_events": 29,
                "monthly_slope_36m": 0.3,
                "w1_mean_mw": 5.2, "w1_std_mw": 0.5,
                "w3_mean_mw": 5.1, "w3_b_value": 0.85,
                "w1_mean_depth_km": 25, "w1_std_depth_km": 8,
                "w3_mean_depth_km": 30,
                "w1_mean_dist_km": 110, "w1_std_dist_km": 45,
                "w3_mean_dist_km": 140,
                "w1_migration_slope_km_day": 0.01,
                "w3_migration_slope_km_day": 0.02,
                "z_rate_1y": 1.5, "z_rate_3y": 0.8,
                "z_b_value_3y": -0.5, "z_max_mw_1y": 1.2,
                "z_depth_1y": 0.3, "z_dist_1y": -0.4,
                "temporal_entropy_12m": 2.1,
                "monthly_entropy_36m": 3.4,
                "interevent_cv_12m": 0.85,
            }
            result = predict_with_uncertainty(test_features)
            print()
            for k, v in result.items():
                print(f"  {k}: {v}")
    else:
        print("Kullanim:")
        print("  python scripts/bootstrap_uncertainty.py --train --n 30")
        print("  python scripts/bootstrap_uncertainty.py --train --n 30 --test")


if __name__ == "__main__":
    main()