#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Model Dogrulama Audit Paketi
==============================================
Modelin ezber mi oruntu mu ogrendigini test eder.

6 Test:
  1. Label Shuffle Test (leakage/sanity check)
  2. Time-Split Test (gelecege genelleme)
  3. Leave-One-Region-Out Test (bolge ezberi)
  4. Naive Baseline Karsilastirma
  5. Spatial Leakage Test (koordinat ezberi)
  6. Feature Ablation Test (hangi bilgi kritik?)

Kullanim:
  python scripts/validation_audit.py --all
  python scripts/validation_audit.py --test label_shuffle
  python scripts/validation_audit.py --test time_split
"""

import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.dummy import DummyClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier

OUTPUT_DIR = Path("output/audit")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "count_0_1y","count_1_2y","count_2_3y",
    "count_linear_trend","count_accel_ratio",
    "w1_n_events","w3_n_events",
    "quiescence_ratio","accel_90d","monthly_slope_36m",
    "w1_mean_mw","w1_std_mw","w1_max_mw",
    "w3_mean_mw","w3_max_mw",
    "w1_b_value","w3_b_value","b_drop_w3_w1",
    "w1_mean_depth_km","w1_std_depth_km",
    "w3_mean_depth_km","depth_change_km",
    "w1_mean_dist_km","w1_std_dist_km",
    "w3_mean_dist_km","spatial_focus_change",
    "w1_migration_slope_km_day","w3_migration_slope_km_day",
]

SPATIAL_FEATURES = [
    "z_rate_1y","z_rate_3y",
    "z_b_value_1y","z_b_value_3y",
    "z_max_mw_1y","z_depth_1y","z_dist_1y",
]


# =========================================================
# VERI HAZIRLAMA
# =========================================================

def load_data():
    real_path = Path("output/gcmt_precursor_features.csv")
    ctrl_path = Path("output/gcmt_control_features.csv")

    if not real_path.exists() or not ctrl_path.exists():
        print("HATA: Feature dosyalari bulunamadi")
        return None, None, None, None

    real = pd.read_csv(real_path, low_memory=False)
    ctrl = pd.read_csv(ctrl_path, low_memory=False)

    if "radius_km" in real.columns:
        real = real[real["radius_km"] == 200].copy()
    if "radius_km" in ctrl.columns:
        ctrl = ctrl[ctrl["radius_km"] == 200].copy()

    return real, ctrl, real_path, ctrl_path


def add_derived(df):
    df = df.copy()
    c0 = df.get("count_0_1y", pd.Series(0)).fillna(0)
    c1 = df.get("count_1_2y", pd.Series(0)).fillna(0)
    c2 = df.get("count_2_3y", pd.Series(0)).fillna(0)
    df["count_linear_trend"] = c0 - c2
    df["count_accel_ratio"] = c0 / ((c1 + c2) / 2.0 + 1e-6)
    df["b_drop_w3_w1"] = df.get("w3_b_value", pd.Series(dtype=float)) - \
                          df.get("w1_b_value", pd.Series(dtype=float))
    df["spatial_focus_change"] = df.get("w3_mean_dist_km", pd.Series(dtype=float)) - \
                                  df.get("w1_mean_dist_km", pd.Series(dtype=float))
    df["depth_change_km"] = df.get("w1_mean_depth_km", pd.Series(dtype=float)) - \
                             df.get("w3_mean_depth_km", pd.Series(dtype=float))
    return df


def make_dataset(real, ctrl, feature_list=None):
    if feature_list is None:
        feature_list = FEATURES + SPATIAL_FEATURES

    real = add_derived(real.copy())
    ctrl = add_derived(ctrl.copy())
    real["target"] = 1
    ctrl["target"] = 0

    avail = [f for f in feature_list if f in real.columns and f in ctrl.columns]

    combined = pd.concat([
        real[avail + ["target"]],
        ctrl[avail + ["target"]],
    ], ignore_index=True)

    X = combined[avail]
    y = combined["target"]
    return X, y, avail


def make_pipe(pos_weight=1.0):
    if HAS_XGB:
        mdl = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr", verbosity=0, random_state=42
        )
    else:
        mdl = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42
        )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", mdl),
    ])


def eval_cv(X, y, pipe=None, n_splits=5):
    if pipe is None:
        pw = (y == 0).sum() / max(y.sum(), 1)
        pipe = make_pipe(pw)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    aps = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
    return {
        "auc_mean": round(float(aucs.mean()), 4),
        "auc_std": round(float(aucs.std()), 4),
        "ap_mean": round(float(aps.mean()), 4),
        "fold_aucs": [round(float(a), 4) for a in aucs],
    }


# =========================================================
# TEST 1: LABEL SHUFFLE (Sanity Check)
# =========================================================

def test_label_shuffle(X, y, n_repeats=5):
    """
    Etiketleri karistir ve modeli egit.
    Saglikli modelde AUC ~0.50 olmali.
    Eger AUC >> 0.50 ise ciddi leakage var.
    """
    print("\n" + "=" * 60)
    print("TEST 1: LABEL SHUFFLE (Sanity Check)")
    print("=" * 60)
    print("  Amac: Etiketler rastgele olunca model hala ogrenebiliyor mu?")
    print("  Beklenen: AUC ~ 0.50 (rastgele)")

    shuffle_aucs = []

    for i in range(n_repeats):
        y_shuffled = y.sample(frac=1, random_state=i + 100).reset_index(drop=True)
        pw = (y_shuffled == 0).sum() / max(y_shuffled.sum(), 1)
        pipe = make_pipe(pw)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(pipe, X, y_shuffled, cv=cv, scoring="roc_auc")
        mean_auc = aucs.mean()
        shuffle_aucs.append(mean_auc)
        print(f"  Tekrar {i + 1}: AUC = {mean_auc:.4f}")

    avg_shuffle = np.mean(shuffle_aucs)
    print(f"\n  Ortalama shuffle AUC: {avg_shuffle:.4f}")

    if avg_shuffle < 0.55:
        verdict = "GECTI — Model rastgele etiketlerden ogrenmedi"
        passed = True
    else:
        verdict = f"BASARISIZ — Shuffle AUC cok yuksek ({avg_shuffle:.4f}), leakage olabilir"
        passed = False

    print(f"  SONUC: {verdict}")
    return {"test": "label_shuffle", "passed": passed,
            "shuffle_auc": round(avg_shuffle, 4),
            "detail": shuffle_aucs, "verdict": verdict}


# =========================================================
# TEST 2: TIME-SPLIT (Gelecege genelleme)
# =========================================================

def test_time_split(real, ctrl):
    """
    Gecmisle egit, gelecekle test et.
    Train: 1976-2005, Val: 2006-2015, Test: 2016-2025
    """
    print("\n" + "=" * 60)
    print("TEST 2: TIME-SPLIT (Gelecege Genelleme)")
    print("=" * 60)
    print("  Amac: Gecmisle egitten model gelecekte de calisiyor mu?")

    real = add_derived(real.copy())
    ctrl = add_derived(ctrl.copy())

    # Zaman sutunu bul
    time_col = next((c for c in ["main_datetime_utc", "datetime_utc",
                                  "ref_datetime_utc"]
                     if c in real.columns), None)
    time_col_c = next((c for c in ["ref_datetime_utc", "datetime_utc",
                                    "main_datetime_utc"]
                       if c in ctrl.columns), None)

    if not time_col:
        print("  UYARI: Zaman sutunu bulunamadi, test atlanıyor")
        return {"test": "time_split", "passed": None, "verdict": "ATLANDI"}

    real[time_col] = pd.to_datetime(real[time_col], errors="coerce")
    real["_year"] = real[time_col].dt.year

    if time_col_c:
        ctrl[time_col_c] = pd.to_datetime(ctrl[time_col_c], errors="coerce")
        ctrl["_year"] = ctrl[time_col_c].dt.year
    else:
        ctrl["_year"] = 2000  # fallback

    real["target"] = 1
    ctrl["target"] = 0

    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]

    splits = [
        ("1976-2005 -> 2006-2015", (1976, 2005), (2006, 2015)),
        ("1976-2010 -> 2011-2020", (1976, 2010), (2011, 2020)),
        ("1976-2015 -> 2016-2025", (1976, 2015), (2016, 2025)),
    ]

    results = []

    for name, (tr_s, tr_e), (te_s, te_e) in splits:
        r_train = real[(real["_year"] >= tr_s) & (real["_year"] <= tr_e)]
        r_test = real[(real["_year"] >= te_s) & (real["_year"] <= te_e)]
        c_train = ctrl[(ctrl["_year"] >= tr_s) & (ctrl["_year"] <= tr_e)]
        c_test = ctrl[(ctrl["_year"] >= te_s) & (ctrl["_year"] <= te_e)]

        train_df = pd.concat([r_train[avail + ["target"]],
                               c_train[avail + ["target"]]], ignore_index=True)
        test_df = pd.concat([r_test[avail + ["target"]],
                              c_test[avail + ["target"]]], ignore_index=True)

        if len(train_df) < 30 or test_df["target"].sum() < 5:
            print(f"  {name}: Yetersiz veri, atlandi")
            continue

        X_train = train_df[avail]
        y_train = train_df["target"]
        X_test = test_df[avail]
        y_test = test_df["target"]

        pw = (y_train == 0).sum() / max(y_train.sum(), 1)
        pipe = make_pipe(pw)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_pred)
            ap = average_precision_score(y_test, y_pred)
        except:
            auc, ap = 0, 0

        print(f"  {name}")
        print(f"    Train: {len(train_df)} ({int(y_train.sum())} real)")
        print(f"    Test:  {len(test_df)} ({int(y_test.sum())} real)")
        print(f"    AUC: {auc:.4f}, AP: {ap:.4f}")

        results.append({"split": name, "auc": round(auc, 4),
                        "ap": round(ap, 4),
                        "n_train": len(train_df),
                        "n_test": len(test_df)})

    if not results:
        return {"test": "time_split", "passed": None, "verdict": "YETERSIZ VERI"}

    avg_auc = np.mean([r["auc"] for r in results])
    passed = avg_auc >= 0.65

    verdict = (f"{'GECTI' if passed else 'BASARISIZ'} — "
               f"Ortalama time-split AUC: {avg_auc:.4f}")
    print(f"\n  SONUC: {verdict}")

    return {"test": "time_split", "passed": passed,
            "avg_auc": round(avg_auc, 4),
            "splits": results, "verdict": verdict}


# =========================================================
# TEST 3: LEAVE-ONE-REGION-OUT (Bolge ezberi)
# =========================================================

def test_leave_region_out(real, ctrl):
    """
    Bir bolgeyi tamamen test setine al, kalanla egit.
    Model o bolgeyi hic gormeden ne kadar basarili?
    """
    print("\n" + "=" * 60)
    print("TEST 3: LEAVE-ONE-REGION-OUT (Bolge Ezberi)")
    print("=" * 60)
    print("  Amac: Model bolgeyi mi ezberliyor, oruntyu mu ogreniyor?")

    real = add_derived(real.copy())
    ctrl = add_derived(ctrl.copy())
    real["target"] = 1
    ctrl["target"] = 0

    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]

    # Bolge tespiti: lat/lon bazli kaba gruplama
    lat_col = next((c for c in ["main_lat", "lat"] if c in real.columns), None)
    lon_col = next((c for c in ["main_lon", "lon"] if c in real.columns), None)

    if not lat_col or not lon_col:
        print("  UYARI: Koordinat sutunu yok, atlaniyor")
        return {"test": "leave_region_out", "passed": None, "verdict": "ATLANDI"}

    # Bolge tanimlari
    regions = {
        "Turkiye_KAF": {"lat": (39, 42), "lon": (26, 42)},
        "Japonya": {"lat": (30, 46), "lon": (128, 148)},
        "Sili_Peru": {"lat": (-45, 5), "lon": (-85, -65)},
        "Cascadia_SanAndreas": {"lat": (30, 50), "lon": (-130, -110)},
        "Guneydogu_Asya": {"lat": (-15, 20), "lon": (90, 145)},
        "Akdeniz": {"lat": (33, 42), "lon": (10, 35)},
    }

    results = []

    for region_name, bounds in regions.items():
        lat_mask = (real[lat_col] >= bounds["lat"][0]) & \
                   (real[lat_col] <= bounds["lat"][1])
        lon_mask = (real[lon_col] >= bounds["lon"][0]) & \
                   (real[lon_col] <= bounds["lon"][1])
        region_mask = lat_mask & lon_mask

        r_test = real[region_mask]
        r_train = real[~region_mask]

        if len(r_test) < 10 or len(r_train) < 50:
            continue

        train_df = pd.concat([r_train[avail + ["target"]],
                               ctrl[avail + ["target"]]], ignore_index=True)
        test_df = pd.concat([r_test[avail + ["target"]],
                              ctrl.sample(min(len(ctrl), len(r_test) * 2),
                                          random_state=42)[avail + ["target"]]],
                             ignore_index=True)

        X_train = train_df[avail]
        y_train = train_df["target"]
        X_test = test_df[avail]
        y_test = test_df["target"]

        pw = (y_train == 0).sum() / max(y_train.sum(), 1)
        pipe = make_pipe(pw)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_pred)
        except:
            auc = 0

        print(f"  {region_name}: AUC = {auc:.4f} "
              f"(test {len(r_test)} real, train {len(r_train)} real)")

        results.append({"region": region_name, "auc": round(auc, 4),
                        "n_test_real": len(r_test),
                        "n_train_real": len(r_train)})

    if not results:
        return {"test": "leave_region_out", "passed": None,
                "verdict": "YETERSIZ VERI"}

    avg_auc = np.mean([r["auc"] for r in results])
    min_auc = min(r["auc"] for r in results)

    passed = avg_auc >= 0.65 and min_auc >= 0.55

    verdict = (f"{'GECTI' if passed else 'BASARISIZ'} — "
               f"Ort AUC: {avg_auc:.4f}, Min: {min_auc:.4f}")
    print(f"\n  SONUC: {verdict}")

    return {"test": "leave_region_out", "passed": passed,
            "avg_auc": round(avg_auc, 4), "min_auc": round(min_auc, 4),
            "regions": results, "verdict": verdict}


# =========================================================
# TEST 4: NAIVE BASELINE KARSILASTIRMA
# =========================================================

def test_naive_baselines(X, y, model_auc):
    """
    Basit baseline'larla karsilastir.
    Model bunlardan anlamli olarak iyi mi?
    """
    print("\n" + "=" * 60)
    print("TEST 4: NAIVE BASELINE KARSILASTIRMA")
    print("=" * 60)
    print(f"  Model AUC: {model_auc:.4f}")
    print(f"  Amac: Model basit kurallarden gercekten daha iyi mi?")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    baselines = {}

    # 1. Random baseline
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    random_aucs = cross_val_score(dummy, X, y, cv=cv, scoring="roc_auc")
    baselines["random"] = round(float(random_aucs.mean()), 4)
    print(f"  Random          : {random_aucs.mean():.4f}")

    # 2. Tek feature baseline'lari
    print(f"\n  Tek feature baseline'lari:")
    single_results = []

    for feat in X.columns:
        vals = X[feat].fillna(0).values
        if np.std(vals) == 0:
            continue
        try:
            auc = roc_auc_score(y, vals)
            if auc < 0.5:
                auc = 1 - auc  # yonu duzelt
            single_results.append((feat, round(auc, 4)))
        except:
            pass

    single_results.sort(key=lambda x: -x[1])
    for feat, auc in single_results[:10]:
        print(f"    {feat:<35} AUC = {auc:.4f}")
    baselines["best_single_feature"] = single_results[0] if single_results else ("?", 0)

    # 3. Sadece olay sayisi (count_0_1y)
    if "count_0_1y" in X.columns:
        try:
            count_auc = roc_auc_score(y, X["count_0_1y"].fillna(0))
            if count_auc < 0.5:
                count_auc = 1 - count_auc
            baselines["count_only"] = round(count_auc, 4)
            print(f"\n  Sadece olay sayisi: {count_auc:.4f}")
        except:
            pass

    # 4. Sadece w3_max_mw (en guclu tek feature)
    if "w3_max_mw" in X.columns:
        try:
            max_mw_auc = roc_auc_score(y, X["w3_max_mw"].fillna(0))
            if max_mw_auc < 0.5:
                max_mw_auc = 1 - max_mw_auc
            baselines["max_mw_only"] = round(max_mw_auc, 4)
            print(f"  Sadece max Mw    : {max_mw_auc:.4f}")
        except:
            pass

    # 5. Top-3 feature ile basit model
    if len(single_results) >= 3:
        top3_feats = [f for f, _ in single_results[:3]]
        X_top3 = X[top3_feats]
        pipe_simple = make_pipe()
        top3_aucs = cross_val_score(pipe_simple, X_top3, y, cv=cv,
                                     scoring="roc_auc")
        baselines["top3_features"] = round(float(top3_aucs.mean()), 4)
        print(f"  Top-3 feature   : {top3_aucs.mean():.4f}")

    # Karsilastirma
    best_baseline = max(baselines.values(),
                        key=lambda x: x if isinstance(x, float) else x[1])
    if isinstance(best_baseline, tuple):
        best_baseline = best_baseline[1]

    improvement = model_auc - best_baseline
    passed = improvement >= 0.05

    verdict = (f"{'GECTI' if passed else 'BASARISIZ'} — "
               f"Model en iyi baseline'dan {improvement:+.4f} daha iyi")
    print(f"\n  En iyi baseline AUC: {best_baseline:.4f}")
    print(f"  Model AUC:           {model_auc:.4f}")
    print(f"  Fark:                {improvement:+.4f}")
    print(f"  SONUC: {verdict}")

    return {"test": "naive_baselines", "passed": passed,
            "model_auc": model_auc,
            "best_baseline": best_baseline,
            "improvement": round(improvement, 4),
            "baselines": {k: v if isinstance(v, float) else v[1]
                         for k, v in baselines.items()},
            "top_single_features": single_results[:10],
            "verdict": verdict}


# =========================================================
# TEST 5: SPATIAL LEAKAGE (Koordinat ezberi)
# =========================================================

def test_spatial_leakage(real, ctrl):
    """
    Z-score ve spatial feature'lari cikart.
    Sadece fiziksel ozet feature'larla calisiyor mu?
    """
    print("\n" + "=" * 60)
    print("TEST 5: SPATIAL LEAKAGE (Koordinat Ezberi)")
    print("=" * 60)
    print("  Amac: Z-score/spatial feature'lar olmadan model calisiyor mu?")

    # Tam feature seti
    X_full, y_full, avail_full = make_dataset(real, ctrl,
                                               FEATURES + SPATIAL_FEATURES)
    full_res = eval_cv(X_full, y_full)

    # Sadece fiziksel feature'lar (z-score yok)
    X_phys, y_phys, avail_phys = make_dataset(real, ctrl, FEATURES)
    phys_res = eval_cv(X_phys, y_phys)

    # Minimal feature seti (sadece sayim + b-degeri)
    minimal_feats = [
        "count_0_1y", "count_1_2y", "count_2_3y",
        "quiescence_ratio", "accel_90d",
        "w1_b_value", "w3_b_value",
        "w1_n_events", "w3_n_events",
    ]
    X_min, y_min, avail_min = make_dataset(real, ctrl, minimal_feats)
    min_res = eval_cv(X_min, y_min)

    print(f"\n  Tam set ({len(avail_full)} feature)    : AUC = {full_res['auc_mean']:.4f}")
    print(f"  Fiziksel ({len(avail_phys)} feature)   : AUC = {phys_res['auc_mean']:.4f}")
    print(f"  Minimal ({len(avail_min)} feature)     : AUC = {min_res['auc_mean']:.4f}")

    drop_full_to_phys = full_res["auc_mean"] - phys_res["auc_mean"]
    drop_full_to_min = full_res["auc_mean"] - min_res["auc_mean"]

    print(f"\n  Z-score cikarilinca AUC degisimi: {drop_full_to_phys:+.4f}")
    print(f"  Minimal'e dusunce AUC degisimi:   {drop_full_to_min:+.4f}")

    # Z-score olmadan da iyi calisiyorsa, spatial leakage dusuk
    passed = phys_res["auc_mean"] >= 0.65

    if drop_full_to_phys > 0.05:
        note = "Z-score kaldirmak performansi dusurdu, biraz spatial leakage olabilir"
    elif drop_full_to_phys < -0.01:
        note = "Z-score kaldirmak performansi artirdi (!), z-score gurultu ekliyor"
    else:
        note = "Z-score etkisi minimal, spatial leakage dusuk"

    verdict = f"{'GECTI' if passed else 'BASARISIZ'} — {note}"
    print(f"  SONUC: {verdict}")

    return {"test": "spatial_leakage", "passed": passed,
            "full_auc": full_res["auc_mean"],
            "physical_auc": phys_res["auc_mean"],
            "minimal_auc": min_res["auc_mean"],
            "z_score_impact": round(drop_full_to_phys, 4),
            "verdict": verdict}


# =========================================================
# TEST 6: FEATURE ABLATION (Hangi bilgi kritik?)
# =========================================================

def test_feature_ablation(X, y):
    """
    Her feature grubunu tek tek cikart.
    Hangi grup cikarilinca model en cok etkileniyor?
    """
    print("\n" + "=" * 60)
    print("TEST 6: FEATURE ABLATION (Hangi bilgi kritik?)")
    print("=" * 60)

    baseline = eval_cv(X, y)
    baseline_auc = baseline["auc_mean"]
    print(f"  Baseline (tum feature): AUC = {baseline_auc:.4f}")

    groups = {
        "Olay sayilari": ["count_0_1y", "count_1_2y", "count_2_3y",
                          "w1_n_events", "w3_n_events",
                          "count_linear_trend", "count_accel_ratio"],
        "b-degeri": ["w1_b_value", "w3_b_value", "b_drop_w3_w1"],
        "Buyukluk": ["w1_mean_mw", "w1_std_mw", "w1_max_mw",
                      "w3_mean_mw", "w3_max_mw"],
        "Quiescence/Accel": ["quiescence_ratio", "accel_90d",
                              "monthly_slope_36m"],
        "Mekansal": ["w1_mean_dist_km", "w1_std_dist_km",
                      "w3_mean_dist_km", "spatial_focus_change"],
        "Derinlik": ["w1_mean_depth_km", "w1_std_depth_km",
                      "w3_mean_depth_km", "depth_change_km"],
        "Goc trendi": ["w1_migration_slope_km_day",
                        "w3_migration_slope_km_day"],
    }

    results = []
    print(f"\n  {'Grup':<25} {'AUC':>8} {'Degisim':>10} {'Etki':<15}")
    print(f"  {'-'*60}")

    for group_name, group_feats in groups.items():
        remaining = [f for f in X.columns if f not in group_feats]
        if len(remaining) < 3:
            continue

        X_ablated = X[remaining]
        res = eval_cv(X_ablated, y)
        auc = res["auc_mean"]
        delta = auc - baseline_auc

        if delta < -0.02:
            impact = "KRITIK"
        elif delta < -0.005:
            impact = "ONEMLI"
        elif delta < 0.005:
            impact = "DUSUK"
        else:
            impact = "GEREKSIZ"

        print(f"  {group_name:<25} {auc:>8.4f} {delta:>+10.4f} {impact:<15}")
        results.append({"group": group_name, "auc": auc,
                        "delta": round(delta, 4), "impact": impact})

    results.sort(key=lambda x: x["delta"])
    most_critical = results[0]["group"] if results else "?"

    print(f"\n  En kritik grup: {most_critical}")
    return {"test": "feature_ablation",
            "baseline_auc": baseline_auc,
            "results": results,
            "most_critical": most_critical}


# =========================================================
# TUM TESTLER
# =========================================================

def run_all_tests():
    print("=" * 60)
    print("SEISMOPATTERN MODEL DOGRULAMA AUDIT")
    print(f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    real, ctrl, _, _ = load_data()
    if real is None:
        return

    X, y, avail = make_dataset(real, ctrl)
    print(f"\nVeri: {len(X)} ornek, {len(avail)} feature")
    print(f"  real: {int(y.sum())}, ctrl: {int((y == 0).sum())}")

    # Mevcut model performansi
    baseline = eval_cv(X, y)
    model_auc = baseline["auc_mean"]
    print(f"\nMevcut model AUC: {model_auc:.4f}")

    all_results = {}

    # Test 1
    all_results["label_shuffle"] = test_label_shuffle(X, y)

    # Test 2
    all_results["time_split"] = test_time_split(real, ctrl)

    # Test 3
    all_results["leave_region_out"] = test_leave_region_out(real, ctrl)

    # Test 4
    all_results["naive_baselines"] = test_naive_baselines(X, y, model_auc)

    # Test 5
    all_results["spatial_leakage"] = test_spatial_leakage(real, ctrl)

    # Test 6
    all_results["feature_ablation"] = test_feature_ablation(X, y)

    # =========================================================
    # GENEL RAPOR
    # =========================================================

    print(f"\n{'=' * 60}")
    print("AUDIT OZET RAPORU")
    print(f"{'=' * 60}")

    n_passed = 0
    n_failed = 0
    n_skipped = 0

    print(f"\n  {'Test':<30} {'Sonuc':<12} {'Detay'}")
    print(f"  {'-' * 70}")

    for name, res in all_results.items():
        if res.get("passed") is True:
            status = "GECTI"
            n_passed += 1
        elif res.get("passed") is False:
            status = "BASARISIZ"
            n_failed += 1
        else:
            status = "ATLANDI"
            n_skipped += 1

        detail = res.get("verdict", "")[:50]
        print(f"  {name:<30} {status:<12} {detail}")

    print(f"\n  Gecen  : {n_passed}")
    print(f"  Kalan  : {n_failed}")
    print(f"  Atlanan: {n_skipped}")

    # Genel hukum
    if n_failed == 0 and n_passed >= 4:
        overall = "MODEL GUVENILIR — Tum testler gecti"
    elif n_failed <= 1 and n_passed >= 3:
        overall = "MODEL KABUL EDILEBILIR — Kucuk sorunlar var"
    elif n_failed <= 2:
        overall = "MODEL DIKKATLE KULLANILMALI — Bazi testler basarisiz"
    else:
        overall = "MODEL GUVENILMEZ — Ciddi sorunlar var"

    print(f"\n  GENEL HUKUM: {overall}")

    # Kaydet
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_auc": model_auc,
        "n_samples": len(X),
        "n_features": len(avail),
        "tests": {k: {kk: vv for kk, vv in v.items()
                       if kk != "detail"}
                  for k, v in all_results.items()},
        "summary": {
            "passed": n_passed,
            "failed": n_failed,
            "skipped": n_skipped,
            "overall": overall,
        }
    }
    report_path = OUTPUT_DIR / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Rapor kaydedildi: {report_path}")
    return report


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Tum testleri calistir")
    ap.add_argument("--test", type=str, default=None,
                    help="Tek test calistir (label_shuffle, time_split, "
                         "leave_region_out, naive_baselines, spatial_leakage, "
                         "feature_ablation)")
    args = ap.parse_args()

    if args.all:
        run_all_tests()
    elif args.test:
        real, ctrl, _, _ = load_data()
        if real is None:
            return

        X, y, avail = make_dataset(real, ctrl)
        baseline = eval_cv(X, y)

        if args.test == "label_shuffle":
            test_label_shuffle(X, y)
        elif args.test == "time_split":
            test_time_split(real, ctrl)
        elif args.test == "leave_region_out":
            test_leave_region_out(real, ctrl)
        elif args.test == "naive_baselines":
            test_naive_baselines(X, y, baseline["auc_mean"])
        elif args.test == "spatial_leakage":
            test_spatial_leakage(real, ctrl)
        elif args.test == "feature_ablation":
            test_feature_ablation(X, y)
        else:
            print(f"Bilinmeyen test: {args.test}")
    else:
        print("Kullanim:")
        print("  python scripts/validation_audit.py --all")
        print("  python scripts/validation_audit.py --test label_shuffle")


if __name__ == "__main__":
    main()