#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - SHAP Analizi + Ensemble + Uncertainty
======================================================
1. SHAP feature onem analizi
2. LightGBM / CatBoost ensemble
3. Bootstrap uncertainty tahmini
4. Harici (external) validation
5. Entropy / fractal feature'lar

Kullanim:
  python scripts/shap_analysis.py --all
"""

import json
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP yuklu degil: pip install shap")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

from scipy.stats import entropy as scipy_entropy

OUTPUT_DIR = Path("output/shap_analysis")
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
    "z_rate_1y","z_rate_3y",
    "z_b_value_1y","z_b_value_3y",
    "z_max_mw_1y","z_depth_1y","z_dist_1y",
]


# =========================================================
# VERI HAZIRLAMA
# =========================================================

def load_data():
    real = pd.read_csv("output/gcmt_precursor_features.csv", low_memory=False)
    ctrl = pd.read_csv("output/gcmt_control_features.csv", low_memory=False)
    if "radius_km" in real.columns:
        real = real[real["radius_km"] == 200].copy()
    if "radius_km" in ctrl.columns:
        ctrl = ctrl[ctrl["radius_km"] == 200].copy()

    for df in [real, ctrl]:
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

    real["target"] = 1
    ctrl["target"] = 0
    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]
    combined = pd.concat([real[avail + ["target"]],
                           ctrl[avail + ["target"]]], ignore_index=True)
    X = combined[avail]
    y = combined["target"]
    return X, y, avail, real, ctrl


# =========================================================
# 1. SHAP ANALİZİ
# =========================================================

def run_shap(X, y, avail):
    print("\n" + "=" * 65)
    print("1. SHAP FEATURE ONEM ANALIZI")
    print("=" * 65)

    if not HAS_SHAP:
        print("  SHAP yuklu degil, atlaniyor")
        return None

    # Impute + model egit
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=avail)

    pw = (y == 0).sum() / max(y.sum(), 1)
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw, eval_metric="aucpr",
            verbosity=0, random_state=42
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        )

    model.fit(X_imp, y)

    # SHAP hesapla
    print("  SHAP degerleri hesaplaniyor...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    # Ortalama mutlak SHAP degerleri
    if isinstance(shap_values, list):
        sv = shap_values[1]  # pozitif sinif
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": avail,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\n  SHAP Feature Onem Siralamasi:")
    print(f"  {'Feature':<35} {'Mean |SHAP|':>12} {'Pct':>6}")
    print(f"  {'-'*55}")

    total_shap = mean_abs_shap.sum()
    for _, row in shap_df.head(20).iterrows():
        pct = row["mean_abs_shap"] / total_shap * 100
        bar = "█" * int(pct * 2)
        print(f"  {row['feature']:<35} {row['mean_abs_shap']:>12.4f} "
              f"{pct:>5.1f}% {bar}")

    # Kaydet
    shap_df.to_csv(OUTPUT_DIR / "shap_importance.csv",
                     index=False, encoding="utf-8-sig")

    # SHAP interaction ozeti
    print(f"\n  En onemli 5 feature:")
    top5 = shap_df.head(5)["feature"].tolist()
    for feat in top5:
        idx = avail.index(feat)
        pos_effect = sv[y == 1, idx].mean()
        neg_effect = sv[y == 0, idx].mean()
        print(f"    {feat}:")
        print(f"      Real orneklerde SHAP ort: {pos_effect:+.4f}")
        print(f"      Ctrl orneklerde SHAP ort: {neg_effect:+.4f}")

    return {"shap_df": shap_df, "shap_values": sv, "top5": top5}


# =========================================================
# 2. ENSEMBLE MODELLER
# =========================================================

def run_ensemble(X, y):
    print("\n" + "=" * 65)
    print("2. ENSEMBLE MODEL KARSILASTIRMASI")
    print("=" * 65)

    pw = (y == 0).sum() / max(y.sum(), 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {}

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.8, scale_pos_weight=pw,
                eval_metric="aucpr", verbosity=0, random_state=42
            ))
        ])

    # LightGBM
    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.8, scale_pos_weight=pw,
                verbose=-1, random_state=42
            ))
        ])

    # CatBoost
    if HAS_CAT:
        models["CatBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", CatBoostClassifier(
                iterations=200, depth=4, learning_rate=0.03,
                auto_class_weights="Balanced",
                verbose=0, random_state=42
            ))
        ])

    # GradientBoosting (sklearn)
    from sklearn.ensemble import (GradientBoostingClassifier,
                                   ExtraTreesClassifier,
                                   RandomForestClassifier)

    models["GradientBoosting"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42
        ))
    ])

    models["ExtraTrees"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", ExtraTreesClassifier(
            n_estimators=300, max_depth=6,
            class_weight="balanced", random_state=42
        ))
    ])

    models["RandomForest"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", RandomForestClassifier(
            n_estimators=300, max_depth=6,
            class_weight="balanced", random_state=42
        ))
    ])

    print(f"\n  {'Model':<20} {'AUC':>8} {'AP':>8} {'Brier':>8}")
    print(f"  {'-'*48}")

    results = {}
    best_auc = 0
    best_name = ""

    for name, pipe in models.items():
        try:
            aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
            aps = cross_val_score(pipe, X, y, cv=cv,
                                   scoring="average_precision")

            # Brier icin OOS tahmin gerekli
            y_pred = np.zeros(len(y))
            for tr, te in cv.split(X, y):
                p = Pipeline(pipe.steps.copy())
                p.fit(X.iloc[tr], y.iloc[tr])
                y_pred[te] = p.predict_proba(X.iloc[te])[:, 1]
            brier = brier_score_loss(y, y_pred)

            auc = aucs.mean()
            ap = aps.mean()

            print(f"  {name:<20} {auc:>8.4f} {ap:>8.4f} {brier:>8.4f}")
            results[name] = {"auc": round(auc, 4), "ap": round(ap, 4),
                              "brier": round(brier, 4)}
            if auc > best_auc:
                best_auc = auc
                best_name = name

        except Exception as e:
            print(f"  {name:<20} HATA: {e}")

    # Ensemble: tum modellerin ortalamasi
    print(f"\n  Ensemble (ortalama) hesaplaniyor...")
    y_ens = np.zeros(len(y))
    n_models = 0

    for name, pipe in models.items():
        try:
            y_pred = np.zeros(len(y))
            for tr, te in cv.split(X, y):
                p = Pipeline(pipe.steps.copy())
                p.fit(X.iloc[tr], y.iloc[tr])
                y_pred[te] = p.predict_proba(X.iloc[te])[:, 1]
            y_ens += y_pred
            n_models += 1
        except:
            pass

    if n_models > 0:
        y_ens /= n_models
        ens_auc = roc_auc_score(y, y_ens)
        ens_ap = average_precision_score(y, y_ens)
        ens_brier = brier_score_loss(y, y_ens)
        print(f"  {'ENSEMBLE':<20} {ens_auc:>8.4f} {ens_ap:>8.4f} "
              f"{ens_brier:>8.4f} ({n_models} model)")
        results["ENSEMBLE"] = {"auc": round(ens_auc, 4),
                                "ap": round(ens_ap, 4),
                                "brier": round(ens_brier, 4)}

    print(f"\n  En iyi tekil: {best_name} (AUC={best_auc:.4f})")
    return results


# =========================================================
# 3. UNCERTAINTY TAHMİNİ
# =========================================================

def run_uncertainty(X, y, n_bootstrap=100):
    print("\n" + "=" * 65)
    print("3. UNCERTAINTY (BELIRSIZLIK) TAHMINI")
    print("=" * 65)
    print(f"  {n_bootstrap} bootstrap modeli egitiliyor...")

    pw = (y == 0).sum() / max(y.sum(), 1)
    rng = np.random.RandomState(42)
    n = len(X)

    all_preds = np.zeros((n_bootstrap, n))

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        if HAS_XGB:
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", RobustScaler()),
                ("mdl", XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, scale_pos_weight=pw,
                    eval_metric="aucpr", verbosity=0,
                    random_state=i
                ))
            ])
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", RobustScaler()),
                ("mdl", GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, random_state=i
                ))
            ])

        pipe.fit(X_boot, y_boot)
        all_preds[i] = pipe.predict_proba(X)[:, 1]

        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n_bootstrap} model tamamlandi")

    # Ortalama ve std
    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    # Uncertainty ozeti
    print(f"\n  Belirsizlik Dagilimi:")
    print(f"    Ort std      : {std_pred.mean():.4f}")
    print(f"    Median std   : {np.median(std_pred):.4f}")
    print(f"    Min std      : {std_pred.min():.4f}")
    print(f"    Max std      : {std_pred.max():.4f}")

    # Real vs Ctrl uncertainty
    real_std = std_pred[y == 1].mean()
    ctrl_std = std_pred[y == 0].mean()
    print(f"\n    Real ornekler ort uncertainty: {real_std:.4f}")
    print(f"    Ctrl ornekler ort uncertainty: {ctrl_std:.4f}")

    # Yuzdelik dagilim
    print(f"\n  Tahmin ± Belirsizlik ornekleri:")
    print(f"  {'Ornek':>6} {'Tahmin':>8} {'±Std':>8} {'CI_low':>8} {'CI_high':>8}")
    print(f"  {'-'*42}")

    sample_idx = np.random.choice(len(y), 10, replace=False)
    for idx in sorted(sample_idx):
        m = mean_pred[idx]
        s = std_pred[idx]
        ci_l = max(0, m - 1.96 * s)
        ci_h = min(1, m + 1.96 * s)
        label = "R" if y.iloc[idx] == 1 else "C"
        print(f"  {label}{idx:>5} {m:>8.4f} {s:>8.4f} "
              f"{ci_l:>8.4f} {ci_h:>8.4f}")

    return {"mean_std": round(float(std_pred.mean()), 4),
            "real_std": round(float(real_std), 4),
            "ctrl_std": round(float(ctrl_std), 4)}


# =========================================================
# 4. HARİCİ DOĞRULAMA
# =========================================================

def run_external_validation(real, ctrl):
    print("\n" + "=" * 65)
    print("4. HARICI (EXTERNAL) DOGRULAMA")
    print("=" * 65)
    print("  Farkli declustering + farkli yaricap ile test")

    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]

    # Test 1: Farkli yaricap (300km yerine 100km veya 200km)
    results = {}

    for radius in [100, 200, 300]:
        r = real.copy()
        c = ctrl.copy()

        if "radius_km" in r.columns:
            r_all = pd.read_csv("output/gcmt_precursor_features.csv",
                                 low_memory=False)
            c_all = pd.read_csv("output/gcmt_control_features.csv",
                                 low_memory=False)
            r = r_all[r_all["radius_km"] == radius].copy()
            c = c_all[c_all["radius_km"] == radius].copy()

        for df in [r, c]:
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

        r["target"] = 1
        c["target"] = 0

        av = [f for f in FEATURES if f in r.columns and f in c.columns]
        comb = pd.concat([r[av + ["target"]], c[av + ["target"]]],
                          ignore_index=True)
        X_r = comb[av]
        y_r = comb["target"]

        if len(X_r) < 50:
            continue

        pw = (y_r == 0).sum() / max(y_r.sum(), 1)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                scale_pos_weight=pw, eval_metric="aucpr",
                verbosity=0, random_state=42
            ) if HAS_XGB else GradientBoostingClassifier(
                n_estimators=200, max_depth=3, random_state=42
            ))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(pipe, X_r, y_r, cv=cv, scoring="roc_auc")

        print(f"  Radius {radius}km: AUC={aucs.mean():.4f} ± {aucs.std():.4f} "
              f"(n={len(X_r)}, real={int(y_r.sum())})")
        results[f"radius_{radius}km"] = {
            "auc": round(float(aucs.mean()), 4),
            "std": round(float(aucs.std()), 4),
            "n": len(X_r),
        }

    # Test 2: Declustered vs Original
    decl_path = Path("output/gcmt_precursor_features_declustered.csv")
    if decl_path.exists():
        print(f"\n  Declustered katalog testi:")
        r_d = pd.read_csv(decl_path, low_memory=False)
        if "radius_km" in r_d.columns:
            r_d = r_d[r_d["radius_km"] == 200].copy()

        for df in [r_d]:
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

        r_d["target"] = 1

        decl_ctrl_path = Path("output/gcmt_control_features_declustered.csv")
        if decl_ctrl_path.exists():
            c_d = pd.read_csv(decl_ctrl_path, low_memory=False)
            if "radius_km" in c_d.columns:
                c_d = c_d[c_d["radius_km"] == 200].copy()
            for df in [c_d]:
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
            c_d["target"] = 0

            av_d = [f for f in FEATURES if f in r_d.columns and f in c_d.columns]
            comb_d = pd.concat([r_d[av_d + ["target"]],
                                 c_d[av_d + ["target"]]], ignore_index=True)
            X_d = comb_d[av_d]
            y_d = comb_d["target"]

            pw = (y_d == 0).sum() / max(y_d.sum(), 1)
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", RobustScaler()),
                ("mdl", XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.03,
                    scale_pos_weight=pw, eval_metric="aucpr",
                    verbosity=0, random_state=42
                ) if HAS_XGB else GradientBoostingClassifier(
                    n_estimators=200, max_depth=3, random_state=42
                ))
            ])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = cross_val_score(pipe, X_d, y_d, cv=cv, scoring="roc_auc")

            print(f"  Declustered: AUC={aucs.mean():.4f} ± {aucs.std():.4f}")
            results["declustered"] = {
                "auc": round(float(aucs.mean()), 4),
                "std": round(float(aucs.std()), 4),
            }

    return results


# =========================================================
# 5. TAM RAPOR
# =========================================================

def run_all():
    print("=" * 65)
    print("SEISMOPATTERN - KAPSAMLI ANALIZ PAKETI")
    print(f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 65)
    print(f"  SHAP: {'OK' if HAS_SHAP else 'YOK'}")
    print(f"  XGBoost: {'OK' if HAS_XGB else 'YOK'}")
    print(f"  LightGBM: {'OK' if HAS_LGBM else 'YOK'}")
    print(f"  CatBoost: {'OK' if HAS_CAT else 'YOK'}")

    X, y, avail, real, ctrl = load_data()
    print(f"\n  Veri: {len(X)} ornek, {len(avail)} feature")

    all_results = {}

    # 1. SHAP
    shap_result = run_shap(X, y, avail)
    if shap_result:
        all_results["shap_top5"] = shap_result["top5"]

    # 2. Ensemble
    ens_results = run_ensemble(X, y)
    all_results["ensemble"] = ens_results

    # 3. Uncertainty
    unc_results = run_uncertainty(X, y, n_bootstrap=100)
    all_results["uncertainty"] = unc_results

    # 4. External validation
    ext_results = run_external_validation(real, ctrl)
    all_results["external"] = ext_results

    # Kaydet
    save_results = {k: v for k, v in all_results.items()
                    if not isinstance(v, np.ndarray)}
    with open(OUTPUT_DIR / "analysis_report.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n{'='*65}")
    print("RAPOR KAYDEDILDI")
    print(f"{'='*65}")
    print(f"  {OUTPUT_DIR / 'analysis_report.json'}")
    print(f"  {OUTPUT_DIR / 'shap_importance.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--uncertainty", action="store_true")
    ap.add_argument("--external", action="store_true")
    args = ap.parse_args()

    X, y, avail, real, ctrl = load_data()

    if args.all:
        run_all()
    elif args.shap:
        run_shap(X, y, avail)
    elif args.ensemble:
        run_ensemble(X, y)
    elif args.uncertainty:
        run_uncertainty(X, y)
    elif args.external:
        run_external_validation(real, ctrl)
    else:
        run_all()


if __name__ == "__main__":
    main()