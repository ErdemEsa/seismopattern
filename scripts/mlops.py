#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 5.3: MLOps Otomasyon
=============================================
Model yasam dongusu yonetimi:
  1. Veri guncelleme (yeni GCMT verisi)
  2. Model yeniden egitim
  3. A/B karsilastirma
  4. Otomatik deploy
  5. Performans izleme

Kullanim:
  python scripts/mlops.py --status
  python scripts/mlops.py --retrain
  python scripts/mlops.py --compare
  python scripts/mlops.py --deploy
  python scripts/mlops.py --full-pipeline
"""

import json
import shutil
import hashlib
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MODEL_DIR = Path("output/models")
ARCHIVE_DIR = Path("output/model_archive")
ARCHIVE_DIR.mkdir(exist_ok=True)
MLOPS_LOG = Path("output/mlops_log.json")

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
# LOG YONETIMI
# =========================================================

def load_log():
    if MLOPS_LOG.exists():
        with open(MLOPS_LOG) as f:
            return json.load(f)
    return {"runs": [], "current_version": "1.0.0", "deploy_history": []}


def save_log(log):
    with open(MLOPS_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)


def next_version(current):
    parts = current.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


# =========================================================
# MODEL DURUMU
# =========================================================

def show_status():
    print("=" * 60)
    print("MLOPS DURUM RAPORU")
    print("=" * 60)

    log = load_log()
    print(f"\n  Mevcut versiyon: {log['current_version']}")
    print(f"  Toplam egitim  : {len(log['runs'])}")
    print(f"  Deploy sayisi  : {len(log['deploy_history'])}")

    # Model dosyalari
    print(f"\n  Model dosyalari:")
    for pt in ["TIP_A", "TIP_B", "TIP_C"]:
        p = MODEL_DIR / f"model_{pt}.joblib"
        if p.exists():
            size = p.stat().st_size / 1024
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            print(f"    {pt}: {size:.0f} KB, {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"    {pt}: YOK")

    # Feature listesi
    fl_path = MODEL_DIR / "feature_lists.json"
    if fl_path.exists():
        with open(fl_path) as f:
            fl = json.load(f)
        for pt, feats in fl.items():
            print(f"    {pt} features: {len(feats)}")

    # Son egitim
    if log["runs"]:
        last = log["runs"][-1]
        print(f"\n  Son egitim:")
        print(f"    Tarih    : {last.get('timestamp', '?')[:16]}")
        print(f"    Versiyon : {last.get('version', '?')}")
        print(f"    AUC      : {last.get('auc', '?')}")
        print(f"    AP       : {last.get('ap', '?')}")
        print(f"    Veri     : {last.get('n_real', '?')} real, "
              f"{last.get('n_ctrl', '?')} ctrl")

    # Arsiv
    archives = list(ARCHIVE_DIR.glob("v*"))
    if archives:
        print(f"\n  Arsivlenen versiyonlar: {len(archives)}")
        for a in sorted(archives)[-5:]:
            print(f"    {a.name}")

    # Veri guncelligi
    csv_path = Path("output/all_earthquakes.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False, usecols=["datetime_utc"])
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
        latest = df["datetime_utc"].max()
        age = (datetime.utcnow() - latest).days
        print(f"\n  Katalog:")
        print(f"    Son kayit  : {latest}")
        print(f"    Yas (gun)  : {age}")
        if age > 30:
            print(f"    UYARI: Katalog {age} gun eski, guncelleme onerilir")


# =========================================================
# VERI HAZIRLAMA
# =========================================================

def rule_based_type(r):
    qr = r.get("quiescence_ratio")
    acc = r.get("accel_90d")
    n3 = r.get("w3_n_events", 0) or 0
    if qr is None or pd.isna(qr) or n3 < 3: return "TIP_C"
    if qr < 0.5: return "TIP_B"
    if qr >= 1.0: return "TIP_A"
    if 0.5 <= qr < 0.8:
        return "TIP_A" if (acc and not pd.isna(acc) and acc >= 1.5) else "TIP_B"
    return "TIP_A"


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


def prepare_data(real_path=None, ctrl_path=None, radius=200):
    if real_path is None:
        real_path = Path("output/real_normalized.csv")
        if not real_path.exists():
            real_path = Path("output/gcmt_precursor_features.csv")
    if ctrl_path is None:
        ctrl_path = Path("output/ctrl_normalized.csv")
        if not ctrl_path.exists():
            ctrl_path = Path("output/gcmt_control_features.csv")

    real = pd.read_csv(real_path, low_memory=False)
    ctrl = pd.read_csv(ctrl_path, low_memory=False)

    if "radius_km" in real.columns:
        real = real[real["radius_km"] == radius]
    if "radius_km" in ctrl.columns:
        ctrl = ctrl[ctrl["radius_km"] == radius]

    real = add_derived(real)
    ctrl = add_derived(ctrl)
    real["target"] = 1
    ctrl["target"] = 0
    real["pattern_type"] = real.apply(rule_based_type, axis=1)

    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]

    return real, ctrl, avail


# =========================================================
# MODEL EGITIMI
# =========================================================

def train_models(real, ctrl, avail):
    """Iki asamali model egitimi (Faz 8-9 ile ayni mimari)."""
    results = {}

    for pt in ["TIP_A", "TIP_B", "TIP_C"]:
        real_pt = real[real["pattern_type"] == pt].copy()
        if len(real_pt) < 10:
            print(f"  {pt}: Yetersiz veri ({len(real_pt)})")
            continue

        real_pt_data = real_pt[avail + ["target"]]
        ctrl_data = ctrl[avail + ["target"]]
        combined = pd.concat([real_pt_data, ctrl_data], ignore_index=True)

        X = combined[avail]
        y = combined["target"]

        pw = (y == 0).sum() / max(y.sum(), 1)

        if HAS_XGB:
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", RobustScaler()),
                ("mdl", XGBClassifier(
                    n_estimators=300, max_depth=4,
                    learning_rate=0.03, subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=pw,
                    eval_metric="aucpr",
                    verbosity=0, random_state=42
                ))
            ])
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", RobustScaler()),
                ("mdl", GradientBoostingClassifier(
                    n_estimators=300, max_depth=3,
                    learning_rate=0.03, random_state=42
                ))
            ])

        # 5-Fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(pipe, X, y, cv=cv,
                                 scoring={"auc": "roc_auc",
                                          "ap": "average_precision"})

        auc = cv_res["test_auc"].mean()
        ap = cv_res["test_ap"].mean()

        # Tam veri ile egit
        pipe.fit(X, y)

        results[pt] = {
            "pipe": pipe,
            "auc": round(float(auc), 4),
            "ap": round(float(ap), 4),
            "n_real": int(y.sum()),
            "n_ctrl": int((y == 0).sum()),
            "n_features": len(avail),
        }

        print(f"  {pt}: AUC={auc:.4f}, AP={ap:.4f} "
              f"({int(y.sum())} real, {int((y==0).sum())} ctrl)")

    return results


# =========================================================
# KARSILASTIRMA (A/B TEST)
# =========================================================

def compare_models(new_results, old_version=None):
    """Yeni model vs mevcut model karsilastirmasi."""
    print(f"\n{'='*60}")
    print("MODEL KARSILASTIRMASI")
    print(f"{'='*60}")

    log = load_log()

    if old_version is None and log["runs"]:
        old_run = log["runs"][-1]
    else:
        old_run = None

    print(f"\n  {'Tip':<8} {'Yeni AUC':>10} {'Eski AUC':>10} {'Fark':>8} {'Karar':<10}")
    print(f"  {'-'*50}")

    decisions = {}
    overall_better = True

    for pt in ["TIP_A", "TIP_B", "TIP_C"]:
        new = new_results.get(pt, {})
        new_auc = new.get("auc", 0)

        old_auc = 0
        if old_run and "tip_results" in old_run:
            old_auc = old_run["tip_results"].get(pt, {}).get("auc", 0)

        delta = new_auc - old_auc
        if delta > 0.005:
            decision = "IYILESTI"
        elif delta < -0.005:
            decision = "KOTULESTI"
            overall_better = False
        else:
            decision = "AYNI"

        decisions[pt] = decision

        print(f"  {pt:<8} {new_auc:>10.4f} {old_auc:>10.4f} "
              f"{delta:>+8.4f} {decision:<10}")

    # Genel karar
    if overall_better:
        overall = "DEPLOY ONERILIR"
    else:
        overall = "DEPLOY ONERILMEZ"

    print(f"\n  Genel karar: {overall}")
    return overall_better, decisions


# =========================================================
# DEPLOY
# =========================================================

def deploy_models(new_results, avail):
    """Yeni modelleri uretim ortamina deploy et."""
    log = load_log()
    old_version = log["current_version"]
    new_version = next_version(old_version)

    print(f"\n  Deploy: v{old_version} -> v{new_version}")

    # Mevcut modelleri arsivle
    archive_path = ARCHIVE_DIR / f"v{old_version}"
    archive_path.mkdir(exist_ok=True)

    for pt in ["TIP_A", "TIP_B", "TIP_C"]:
        src = MODEL_DIR / f"model_{pt}.joblib"
        if src.exists():
            shutil.copy2(src, archive_path / f"model_{pt}.joblib")

    fl_src = MODEL_DIR / "feature_lists.json"
    if fl_src.exists():
        shutil.copy2(fl_src, archive_path / "feature_lists.json")

    mc_src = MODEL_DIR / "model_card.json"
    if mc_src.exists():
        shutil.copy2(mc_src, archive_path / "model_card.json")

    print(f"  Eski model arsivlendi: {archive_path}")

    # Yeni modelleri kaydet
    feature_lists = {}
    for pt, res in new_results.items():
        pipe = res["pipe"]
        joblib.dump(pipe, MODEL_DIR / f"model_{pt}.joblib")
        feature_lists[pt] = avail
        print(f"  {pt} deploy edildi")

    with open(MODEL_DIR / "feature_lists.json", "w") as f:
        json.dump(feature_lists, f, indent=2)

    # Model kartini guncelle
    tip_aucs = {pt: res["auc"] for pt, res in new_results.items()}
    avg_auc = np.mean(list(tip_aucs.values()))

    model_card = {
        "version": new_version,
        "created": datetime.utcnow().isoformat(),
        "description": "SeismoPattern Iki Asamali Model",
        "previous_version": old_version,
        "performance": {
            "overall_auc_estimate": round(float(avg_auc), 4),
            **{f"{pt}_cv_auc": auc for pt, auc in tip_aucs.items()},
        },
        "training_data": {
            "real_samples": sum(r["n_real"] for r in new_results.values()),
            "ctrl_samples": new_results.get("TIP_A", {}).get("n_ctrl", 0),
            "n_features": len(avail),
        },
    }
    with open(MODEL_DIR / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2, default=str)

    # Log guncelle
    run_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": new_version,
        "auc": round(float(avg_auc), 4),
        "ap": round(float(np.mean([r["ap"] for r in new_results.values()])), 4),
        "n_real": sum(r["n_real"] for r in new_results.values()),
        "n_ctrl": new_results.get("TIP_A", {}).get("n_ctrl", 0),
        "tip_results": {pt: {"auc": r["auc"], "ap": r["ap"]}
                        for pt, r in new_results.items()},
    }
    log["runs"].append(run_entry)
    log["current_version"] = new_version
    log["deploy_history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "from_version": old_version,
        "to_version": new_version,
        "reason": "retrain",
    })
    save_log(log)

    print(f"\n  Deploy tamamlandi: v{new_version}")
    print(f"  Ortalama AUC: {avg_auc:.4f}")
    return new_version


# =========================================================
# PERFORMANS IZLEME
# =========================================================

def check_model_health():
    """Model performansinin zamanla dusuyor mu kontrol et."""
    log = load_log()

    if len(log["runs"]) < 2:
        print("  Yeterli egitim gecmisi yok (min 2 run)")
        return True

    runs = log["runs"]
    aucs = [r.get("auc", 0) for r in runs[-5:]]

    if len(aucs) >= 3:
        trend = np.polyfit(range(len(aucs)), aucs, 1)[0]
        print(f"  Son {len(aucs)} run AUC trendi: {trend:+.4f}")

        if trend < -0.01:
            print(f"  UYARI: Model performansi dusuyor!")
            return False

    latest_auc = aucs[-1]
    if latest_auc < 0.80:
        print(f"  UYARI: AUC cok dusuk ({latest_auc:.4f})")
        return False

    print(f"  Model sagligi: OK (AUC={latest_auc:.4f})")
    return True


# =========================================================
# TAM PIPELINE
# =========================================================

def full_pipeline(force_deploy=False):
    """Tam MLOps pipeline: veri → egitim → karsilastirma → deploy."""
    print("=" * 60)
    print("MLOPS TAM PIPELINE")
    print(f"Tarih: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 1. Veri hazirla
    print("\n[1/5] Veri hazirlaniyor...")
    real, ctrl, avail = prepare_data()
    print(f"  real={len(real)}, ctrl={len(ctrl)}, features={len(avail)}")

    # 2. Model egit
    print("\n[2/5] Model egitiliyor...")
    new_results = train_models(real, ctrl, avail)

    if not new_results:
        print("  HATA: Model egitilemedi")
        return

    # 3. Karsilastir
    print("\n[3/5] Karsilastirma...")
    is_better, decisions = compare_models(new_results)

    # 4. Deploy karari
    print("\n[4/5] Deploy karari...")
    if is_better or force_deploy:
        new_version = deploy_models(new_results, avail)
        print(f"  DEPLOY YAPILDI: v{new_version}")
    else:
        print("  Deploy atlanıyor (yeni model daha iyi degil)")

    # 5. Saglik kontrolu
    print("\n[5/5] Saglik kontrolu...")
    healthy = check_model_health()

    # Ozet
    print(f"\n{'='*60}")
    print("PIPELINE OZETI")
    print(f"{'='*60}")
    for pt, res in new_results.items():
        print(f"  {pt}: AUC={res['auc']:.4f}, AP={res['ap']:.4f}")
    print(f"  Deploy: {'EVET' if (is_better or force_deploy) else 'HAYIR'}")
    print(f"  Saglik: {'OK' if healthy else 'UYARI'}")


# =========================================================
# ROLLBACK
# =========================================================

def rollback(target_version=None):
    """Onceki versiyona geri don."""
    log = load_log()

    if target_version is None:
        archives = sorted(ARCHIVE_DIR.glob("v*"))
        if not archives:
            print("Arsiv bos, rollback yapilamaz")
            return
        target = archives[-1]
        target_version = target.name[1:]
    else:
        target = ARCHIVE_DIR / f"v{target_version}"

    if not target.exists():
        print(f"Arsiv bulunamadi: {target}")
        return

    print(f"Rollback: v{log['current_version']} -> v{target_version}")

    # Mevcut modelleri yedekle
    backup = ARCHIVE_DIR / f"v{log['current_version']}_rollback_backup"
    backup.mkdir(exist_ok=True)
    for f in MODEL_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, backup / f.name)

    # Arsivden geri yukle
    for f in target.glob("*"):
        shutil.copy2(f, MODEL_DIR / f.name)

    log["current_version"] = target_version
    log["deploy_history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "from_version": log["current_version"],
        "to_version": target_version,
        "reason": "rollback",
    })
    save_log(log)

    print(f"Rollback tamamlandi: v{target_version}")


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser(
        description="SeismoPattern MLOps"
    )
    ap.add_argument("--status", action="store_true",
                    help="Mevcut model durumu")
    ap.add_argument("--retrain", action="store_true",
                    help="Modeli yeniden egit")
    ap.add_argument("--compare", action="store_true",
                    help="Yeni vs eski karsilastirma")
    ap.add_argument("--deploy", action="store_true",
                    help="Yeni modeli deploy et")
    ap.add_argument("--full-pipeline", action="store_true",
                    help="Tam pipeline calistir")
    ap.add_argument("--health", action="store_true",
                    help="Model saglik kontrolu")
    ap.add_argument("--rollback", action="store_true",
                    help="Onceki versiyona don")
    ap.add_argument("--version", type=str, default=None,
                    help="Rollback hedef versiyonu")
    ap.add_argument("--force", action="store_true",
                    help="Deploy'u zorla")
    args = ap.parse_args()

    if args.status:
        show_status()
    elif args.retrain:
        real, ctrl, avail = prepare_data()
        results = train_models(real, ctrl, avail)
    elif args.compare:
        real, ctrl, avail = prepare_data()
        results = train_models(real, ctrl, avail)
        compare_models(results)
    elif args.deploy:
        real, ctrl, avail = prepare_data()
        results = train_models(real, ctrl, avail)
        deploy_models(results, avail)
    elif args.full_pipeline:
        full_pipeline(force_deploy=args.force)
    elif args.health:
        check_model_health()
    elif args.rollback:
        rollback(args.version)
    else:
        print("Kullanim:")
        print("  python scripts/mlops.py --status")
        print("  python scripts/mlops.py --full-pipeline")
        print("  python scripts/mlops.py --retrain")
        print("  python scripts/mlops.py --rollback")


if __name__ == "__main__":
    main()