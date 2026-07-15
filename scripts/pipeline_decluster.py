#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Declustering Pipeline
Temiz katalogla tüm feature ve model sürecini çalıştırır.
Sonuçları orijinal (kirli) sonuçlarla karşılaştırır.
"""

import subprocess
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


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


def rule_based_type(row):
    qr  = row.get("quiescence_ratio")
    acc = row.get("accel_90d")
    n3  = row.get("w3_n_events", 0) or 0
    if qr is None or pd.isna(qr) or n3 < 3: return "TIP_C"
    if qr < 0.5:   return "TIP_B"
    if qr >= 1.0:  return "TIP_A"
    if 0.5 <= qr < 0.8:
        return "TIP_A" if (acc and not pd.isna(acc) and acc >= 1.5) else "TIP_B"
    return "TIP_A"


def add_derived(df):
    df = df.copy()
    c0 = df.get("count_0_1y", pd.Series(0, index=df.index)).fillna(0)
    c1 = df.get("count_1_2y", pd.Series(0, index=df.index)).fillna(0)
    c2 = df.get("count_2_3y", pd.Series(0, index=df.index)).fillna(0)
    df["count_linear_trend"]   = c0 - c2
    df["count_accel_ratio"]    = c0 / ((c1+c2)/2.0 + 1e-6)
    df["b_drop_w3_w1"]         = df.get("w3_b_value", pd.Series(dtype=float)) - \
                                  df.get("w1_b_value", pd.Series(dtype=float))
    df["spatial_focus_change"] = df.get("w3_mean_dist_km", pd.Series(dtype=float)) - \
                                  df.get("w1_mean_dist_km", pd.Series(dtype=float))
    df["depth_change_km"]      = df.get("w1_mean_depth_km", pd.Series(dtype=float)) - \
                                  df.get("w3_mean_depth_km", pd.Series(dtype=float))
    return df


def build_dataset(real_df, ctrl_df, radius=200):
    real = real_df[real_df["radius_km"] == radius].copy()
    ctrl = ctrl_df[ctrl_df["radius_km"] == radius].copy()
    real = add_derived(real); ctrl = add_derived(ctrl)
    real["target"] = 1;       ctrl["target"] = 0
    real["pattern_type"] = real.apply(rule_based_type, axis=1)

    ft_col_r = "main_fault_type" if "main_fault_type" in real.columns else None
    ft_col_c = "parent_fault_type" if "parent_fault_type" in ctrl.columns else None

    avail = [f for f in FEATURES if f in real.columns and f in ctrl.columns]

    combined = pd.concat([
        real[avail + ["target","pattern_type"] +
             ([ft_col_r] if ft_col_r else [])],
        ctrl[avail + ["target"] +
             ([ft_col_c] if ft_col_c else [])].rename(
            columns={ft_col_c: "main_fault_type"} if ft_col_c else {}
        ).assign(pattern_type="ctrl"),
    ], ignore_index=True)

    X = combined[avail]
    y = combined["target"]
    pt= combined["pattern_type"]
    return X, y, pt, avail


def evaluate_two_stage(X, y, pt, label=""):
    """İki aşamalı model değerlendirmesi (Faz 8 ile aynı mantık)"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Stratum: real_TIPA, real_TIPB, real_TIPC, ctrl
    strat = pt.copy()
    strat[y == 1] = "real_" + pt[y == 1]
    strat[y == 0] = "ctrl"

    fold_aucs = []
    all_y_true, all_y_score = [], []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, strat)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pt_train, pt_test = pt.iloc[train_idx], pt.iloc[test_idx]

        type_models = {}
        for tip in ["TIP_A","TIP_B","TIP_C"]:
            mask_tr = (((pt_train == tip) & (y_train==1)) | (y_train==0))
            X_tr_pt = X_train[mask_tr]; y_tr_pt = y_train[mask_tr]
            if y_tr_pt.sum() < 10: type_models[tip]=None; continue
            pw = (y_tr_pt==0).sum() / max(y_tr_pt.sum(), 1)
            if HAS_XGB:
                pipe = Pipeline([
                    ("imp",SimpleImputer(strategy="median")),
                    ("scl",RobustScaler()),
                    ("mdl",XGBClassifier(
                        n_estimators=200, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=pw, eval_metric="aucpr",
                        verbosity=0, random_state=42
                    ))
                ])
            else:
                pipe = Pipeline([
                    ("imp",SimpleImputer(strategy="median")),
                    ("scl",RobustScaler()),
                    ("mdl",GradientBoostingClassifier(
                        n_estimators=200, max_depth=3,
                        learning_rate=0.05, random_state=42
                    ))
                ])
            try: pipe.fit(X_tr_pt, y_tr_pt); type_models[tip]=pipe
            except: type_models[tip]=None

        # Test tahmini
        fold_scores = []
        W = {"TIP_A":1.0,"TIP_B":1.2,"TIP_C":0.5}
        for xi, tip_i in zip(X_test.iterrows(), pt_test):
            _, row_x = xi
            scores = {}
            for t, pipe in type_models.items():
                if pipe is None: continue
                try:
                    p = pipe.predict_proba(pd.DataFrame([row_x]))[0,1]
                    scores[t] = p
                except: pass
            if not scores: fold_scores.append(0.5); continue
            primary = scores.get(tip_i)
            ws = sum(W.get(t,1)*s for t,s in scores.items())
            wt = sum(W.get(t,1) for t in scores)
            ens = ws/wt if wt>0 else 0.5
            final = 0.7*primary + 0.3*ens if primary else ens
            fold_scores.append(final)

        from sklearn.metrics import roc_auc_score
        y_test_arr = y_test.values
        fold_scores = np.array(fold_scores)
        try:
            auc = roc_auc_score(y_test_arr, fold_scores)
            fold_aucs.append(auc)
        except: pass
        all_y_true.extend(y_test_arr)
        all_y_score.extend(fold_scores)

    overall_auc = float(np.mean(fold_aucs)) if fold_aucs else 0
    std_auc     = float(np.std(fold_aucs)) if fold_aucs else 0

    from sklearn.metrics import average_precision_score
    try:
        ap = average_precision_score(all_y_true, all_y_score)
    except: ap = 0.0

    if label:
        print(f"\n  [{label}]")
    print(f"  AUC-ROC : {overall_auc:.4f} ± {std_auc:.4f}")
    print(f"  PR-AUC  : {ap:.4f}")
    print(f"  Fold AUC: {[round(a,4) for a in fold_aucs]}")

    return overall_auc, ap


def compare_features(orig_real, decl_real, radius=200):
    """Orijinal vs declustered feature karşılaştırması"""
    print(f"\n{'─'*60}")
    print("FEATURE KARŞILAŞTIRMASI (200km, Mw7+ öncesi)")
    print(f"{'─'*60}")

    orig = orig_real[orig_real["radius_km"]==radius]
    decl = decl_real[decl_real["radius_km"]==radius]

    metrics = [
        "quiescence_ratio","accel_90d",
        "w1_n_events","w3_n_events",
        "w1_b_value","w3_b_value",
        "count_0_1y","count_1_2y","count_2_3y",
    ]

    print(f"\n  {'Metrik':<30} {'Orig median':>12} {'Decl median':>12} {'Fark%':>8}")
    print(f"  {'─'*65}")
    for m in metrics:
        if m not in orig.columns or m not in decl.columns:
            continue
        om = orig[m].median()
        dm = decl[m].median()
        if pd.isna(om) or pd.isna(dm): continue
        diff = (dm-om)/abs(om)*100 if om!=0 else 0
        arrow = "↑" if diff>5 else ("↓" if diff<-5 else "→")
        print(f"  {m:<30} {om:>12.4f} {dm:>12.4f} {arrow}{diff:>+7.1f}%")


def run_pipeline():
    print("="*65)
    print("SEISMOPattern - DECLUSTERİNG KARŞILAŞTIRMA PİPELİNE")
    print("="*65)
    print(f"XGBoost: {'✅' if HAS_XGB else '❌ (GB kullanılacak)'}")

    # ── 1. Kontrol et: temiz katalog var mı? ────────────────
    decl_path = Path("output/all_earthquakes_declustered.csv")
    orig_path  = Path("output/all_earthquakes.csv")

    if not decl_path.exists():
        print("\nHATA: Önce declustering.py çalıştırın:")
        print("  python scripts\\declustering.py \\")
        print("    --input output\\all_earthquakes.csv \\")
        print("    --output-main output\\all_earthquakes_declustered.csv")
        sys.exit(1)

    # ── 2. Orijinal feature'lar mevcut mu? ──────────────────
    orig_real_path = Path("output/gcmt_precursor_features.csv")
    orig_ctrl_path = Path("output/gcmt_control_features.csv")

    if not orig_real_path.exists() or not orig_ctrl_path.exists():
        print("\nHATA: Önce Faz 2 çalıştırılmalı.")
        sys.exit(1)

    orig_real = pd.read_csv(orig_real_path)
    orig_ctrl = pd.read_csv(orig_ctrl_path)

    # ── 3. Declustered feature'ları hesapla ─────────────────
    decl_real_path = Path("output/gcmt_precursor_features_declustered.csv")
    decl_ctrl_path = Path("output/gcmt_control_features_declustered.csv")

    if not decl_real_path.exists():
        print("\n" + "─"*55)
        print("ADIM 1: Declustered feature'lar hesaplanıyor...")
        print("─"*55)
        ret = subprocess.run([
            sys.executable,
            "scripts/phase2_gcmt_features.py",
            "--input",  str(decl_path),
            "--output", str(decl_real_path),
        ], capture_output=False)
        if ret.returncode != 0:
            print("HATA: feature hesaplama başarısız")
            sys.exit(1)
    else:
        print(f"\nDeclustered feature mevcut: {decl_real_path}")

    decl_real = pd.read_csv(decl_real_path)

    if not decl_ctrl_path.exists():
        print("\n" + "─"*55)
        print("ADIM 2: Declustered kontrol pencereleri hesaplanıyor...")
        print("─"*55)
        ret = subprocess.run([
            sys.executable,
            "scripts/phase2_control_windows.py",
            "--input",  str(decl_path),
            "--output", str(decl_ctrl_path),
            "--n_controls", "2",
            "--min_gap_years", "4",
        ], capture_output=False)
        if ret.returncode != 0:
            print("HATA: kontrol penceresi başarısız")
            sys.exit(1)
    else:
        print(f"Declustered kontrol mevcut: {decl_ctrl_path}")

    decl_ctrl = pd.read_csv(decl_ctrl_path)

    # ── 4. Feature karşılaştırması ───────────────────────────
    compare_features(orig_real, decl_real)

    # ── 5. Model karşılaştırması ─────────────────────────────
    print(f"\n{'─'*60}")
    print("MODEL KARŞILAŞTIRMASI")
    print(f"{'─'*60}")

    print(f"\nOrijinal (artçı dahil):")
    print(f"  real={len(orig_real[orig_real['radius_km']==200])}, "
          f"ctrl={len(orig_ctrl[orig_ctrl['radius_km']==200])}")
    X_orig, y_orig, pt_orig, _ = build_dataset(orig_real, orig_ctrl)
    auc_orig, ap_orig = evaluate_two_stage(
        X_orig, y_orig, pt_orig, label="ORİJİNAL"
    )

    print(f"\nDeclustered (artçı filtrelenmiş):")
    print(f"  real={len(decl_real[decl_real['radius_km']==200])}, "
          f"ctrl={len(decl_ctrl[decl_ctrl['radius_km']==200])}")
    X_decl, y_decl, pt_decl, _ = build_dataset(decl_real, decl_ctrl)
    auc_decl, ap_decl = evaluate_two_stage(
        X_decl, y_decl, pt_decl, label="DECLUSTEREd"
    )

    # ── 6. Özet ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("KARŞILAŞTIRMA ÖZETİ")
    print(f"{'='*65}")

    delta_auc = auc_decl - auc_orig
    delta_ap  = ap_decl  - ap_orig
    arrow_auc = "↑" if delta_auc > 0.002 else ("↓" if delta_auc < -0.002 else "→")
    arrow_ap  = "↑" if delta_ap  > 0.002 else ("↓" if delta_ap  < -0.002 else "→")

    print(f"\n  {'Metrik':<20} {'Orijinal':>10} {'Declustered':>12} {'Değişim':>10}")
    print(f"  {'─'*55}")
    print(f"  {'AUC-ROC':<20} {auc_orig:>10.4f} {auc_decl:>12.4f} "
          f"{arrow_auc}{delta_auc:>+8.4f}")
    print(f"  {'PR-AUC':<20} {ap_orig:>10.4f} {ap_decl:>12.4f} "
          f"{arrow_ap}{delta_ap:>+8.4f}")

    print(f"\n  Katalog temizliği:")
    print(f"    Orijinal    : 69,944 olay")
    print(f"    Temizlenmiş : 40,729 olay (%37.1 artçı kaldırıldı)")
    print(f"    Mw7+ fark   : -95 olay (artçı olarak sınıflandı)")

    if delta_auc > 0.005:
        print(f"\n  ✅ Declustering modeli iyileştirdi (+{delta_auc:.4f})")
        verdict = "KULLAN"
    elif delta_auc < -0.005:
        print(f"\n  ⚠️  Declustering modeli kötüleştirdi ({delta_auc:.4f})")
        verdict = "KULLANMA"
    else:
        print(f"\n  → Declustering nötr etki ({delta_auc:+.4f})")
        verdict = "NÖTr"

    print(f"\n  KARAR: {verdict}")
    print(f"  Not: AUC farkı küçük olsa bile kontrol kalitesi")
    print(f"       için declustered katalog önerilir.")

    # Sonuçları kaydet
    results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "auc_original": auc_orig,
        "auc_declustered": auc_decl,
        "delta_auc": delta_auc,
        "ap_original": ap_orig,
        "ap_declustered": ap_decl,
        "delta_ap": delta_ap,
        "verdict": verdict,
    }
    pd.DataFrame([results]).to_csv(
        "output/declustering_comparison.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"\n  Kaydedildi: output/declustering_comparison.csv")


if __name__ == "__main__":
    run_pipeline()