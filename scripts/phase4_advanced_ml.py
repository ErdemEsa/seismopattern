#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 4 Gelişmiş ML
===================================
1. Normalize edilmiş feature'lar (sızıntı önleme)
2. Yeni türetilmiş feature'lar
3. SMOTE ile dengeleme
4. XGBoost + hiperparametre optimizasyonu
5. Fay tipine özel modeller
6. Eşik analizi ve operasyonel skor
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      GridSearchCV, cross_validate)
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, average_precision_score,
                              precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost yüklü değil. pip install xgboost ile yükleyin.")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn yüklü değil. pip install imbalanced-learn")


# ═══════════════════════════════════════════════════════════
# 1. GELİŞMİŞ FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def engineer_features(df, is_real=True):
    """
    Normalize edilmiş ve türetilmiş feature'lar oluştur.
    is_real=True: gerçek öncü pencere (main_mw mevcut)
    is_real=False: kontrol penceresi (parent_mw kullan)
    """
    df = df.copy()

    # Ana deprem büyüklüğü
    mw_col = "main_mw" if is_real else "parent_mw"
    if mw_col not in df.columns:
        df["_main_mw"] = 7.5  # varsayılan
    else:
        df["_main_mw"] = df[mw_col]

    # ── Normalize edilmiş max_mw (sızıntı önleme) ──────────
    # Büyük bölgesel deprem mi var, yoksa bölgenin doğal seviyesi mi?
    for w in ["w1", "w2", "w3"]:
        col = f"{w}_max_mw"
        if col in df.columns:
            # Ana deprem büyüklüğüne göre normalize
            df[f"{w}_max_mw_norm"] = df[col] - df["_main_mw"]
            # -1'den küçük = önceki büyük depremler ana depremin
            #                1 büyüklük altında
            # 0'a yakın = önceki büyük depremler neredeyse aynı büyüklükte

    # ── b-değeri sinyali ──────────────────────────────────
    # b düşüşü: w3 → w2 → w1 arasındaki fark
    if all(f in df.columns for f in ["w1_b_value", "w3_b_value"]):
        df["b_value_drop_w3_w1"] = df["w3_b_value"] - df["w1_b_value"]
        # Pozitif = b düşmüş (eski - yeni) = gerilim artıyor

    if all(f in df.columns for f in ["w1_b_value", "w2_b_value"]):
        df["b_value_drop_w2_w1"] = df["w2_b_value"] - df["w1_b_value"]

    # ── Aktivite oranı trendi ─────────────────────────────
    # count_2_3y → count_1_2y → count_0_1y
    if all(c in df.columns for c in ["count_0_1y", "count_1_2y", "count_2_3y"]):
        # Toplam aktivite
        df["count_total_3y"] = (df["count_0_1y"] +
                                 df["count_1_2y"] +
                                 df["count_2_3y"])

        # Aktivite trendi (son dönem ağırlıklı)
        # Eğer artıyorsa pozitif, azalıyorsa negatif
        c0 = df["count_0_1y"].fillna(0)
        c1 = df["count_1_2y"].fillna(0)
        c2 = df["count_2_3y"].fillna(0)

        # Lineer trend katsayısı
        df["count_linear_trend"] = c0 - c2  # basit fark

        # Hızlanma: son yıl / önceki 2 yıl ortalaması
        avg_prev = (c1 + c2) / 2.0 + 1e-6
        df["count_accel_ratio"] = c0 / avg_prev

    # ── Mekânsal odaklanma normalize ──────────────────────
    if all(c in df.columns for c in ["w1_mean_dist_km", "w3_mean_dist_km"]):
        df["spatial_focus_change"] = (df["w3_mean_dist_km"] -
                                       df["w1_mean_dist_km"])
        # Pozitif = son 1 yılda daha uzaklaşmış (kötü)
        # Negatif = son 1 yılda merkezleşmiş (iyi)

    # Yarıçapa normalize uzaklık
    if "w1_mean_dist_km" in df.columns and "radius_km" in df.columns:
        df["w1_dist_ratio"] = df["w1_mean_dist_km"] / df["radius_km"]
    if "w3_mean_dist_km" in df.columns and "radius_km" in df.columns:
        df["w3_dist_ratio"] = df["w3_mean_dist_km"] / df["radius_km"]

    # ── Derinlik anomalisi ────────────────────────────────
    if all(c in df.columns for c in ["w1_mean_depth_km", "w3_mean_depth_km"]):
        df["depth_change_km"] = (df["w1_mean_depth_km"] -
                                  df["w3_mean_depth_km"])
        # Negatif = sığlaşma (potansiyel sinyal)

    # ── Moment yoğunluğu ──────────────────────────────────
    # Olay başına moment = ne kadar büyük olaylar var?
    for w in ["w1", "w3"]:
        n_col = f"{w}_n_events"
        m_col = f"{w}_cum_moment_nm"
        if n_col in df.columns and m_col in df.columns:
            df[f"{w}_moment_per_event"] = np.where(
                df[n_col] > 0,
                np.log10(df[m_col].clip(lower=1)) / df[n_col].clip(lower=1),
                np.nan
            )

    # ── quiescence × accel etkileşim terimi ──────────────
    if "quiescence_ratio" in df.columns and "accel_90d" in df.columns:
        qr = df["quiescence_ratio"].fillna(1)
        acc = df["accel_90d"].fillna(1)
        df["quiescence_x_accel"] = qr * acc

    # ── Bölgesel sismisiteden sapma ───────────────────────
    # (Gelecekte eklenecek: bölge bazlı ortalamadan fark)

    return df


def get_feature_list():
    """Kullanılacak feature listesi"""
    return [
        # Orijinal özellikler (normalize edilmiş versiyonlar dahil)
        "quiescence_ratio",
        "accel_90d",
        "monthly_slope_36m",
        "count_0_1y",
        "count_1_2y",
        "count_2_3y",

        # Türetilmiş özellikler
        "count_total_3y",
        "count_linear_trend",
        "count_accel_ratio",

        # Büyüklük (normalize)
        "w1_max_mw_norm",
        "w3_max_mw_norm",
        "w1_mean_mw",
        "w3_mean_mw",
        "w1_std_mw",

        # b-değeri (en güçlü sinyal)
        "w1_b_value",
        "w3_b_value",
        "b_value_drop_w3_w1",

        # Derinlik
        "w1_mean_depth_km",
        "w3_mean_depth_km",
        "depth_change_km",
        "w1_std_depth_km",

        # Mekânsal
        "w1_dist_ratio",
        "w3_dist_ratio",
        "spatial_focus_change",
        "w1_std_dist_km",

        # Moment
        "w1_moment_per_event",
        "w3_moment_per_event",

        # Etkileşim
        "quiescence_x_accel",

        # Zaman serisi
        "w1_migration_slope_km_day",
        "w3_migration_slope_km_day",
    ]


# ═══════════════════════════════════════════════════════════
# 2. MODEL HAZIRLAMA
# ═══════════════════════════════════════════════════════════

def prepare_dataset(real_df, ctrl_df, radius=200):
    """Veri seti hazırla"""
    real = real_df[real_df["radius_km"] == radius].copy()
    ctrl = ctrl_df[ctrl_df["radius_km"] == radius].copy()

    # Feature engineering
    real = engineer_features(real, is_real=True)
    ctrl = engineer_features(ctrl, is_real=False)

    real["target"] = 1
    ctrl["target"] = 0

    # Fay tipi sütununu standartlaştır
    if "main_fault_type" in real.columns:
        real["fault_type"] = real["main_fault_type"]
    if "parent_fault_type" in ctrl.columns:
        ctrl["fault_type"] = ctrl["parent_fault_type"]
    elif "main_fault_type" in ctrl.columns:
        ctrl["fault_type"] = ctrl["main_fault_type"]

    feature_list = get_feature_list()
    avail_real = [f for f in feature_list if f in real.columns]
    avail_ctrl = [f for f in feature_list if f in ctrl.columns]
    common = list(set(avail_real) & set(avail_ctrl))

    extra_cols = ["target", "fault_type"]
    combined = pd.concat([
        real[common + [c for c in extra_cols if c in real.columns]],
        ctrl[common + [c for c in extra_cols if c in ctrl.columns]]
    ], ignore_index=True)

    X = combined[common]
    y = combined["target"]
    fault = combined.get("fault_type", pd.Series(["UNKNOWN"] * len(combined)))

    print(f"Dataset: {len(X)} örnek, {len(common)} feature")
    print(f"Dağılım: {int(y.sum())} real, {int((y==0).sum())} control")
    print(f"Oran: 1:{(y==0).sum()/y.sum():.2f}")

    return X, y, fault, common


# ═══════════════════════════════════════════════════════════
# 3. GELİŞMİŞ MODEL EĞİTİMİ
# ═══════════════════════════════════════════════════════════

def build_models(has_smote=False):
    """Model listesi oluştur"""
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    models = {}

    # Lojistik Regresyon
    models["LR"] = Pipeline([
        ("imp", imputer),
        ("scl", scaler),
        ("mdl", LogisticRegression(
            max_iter=2000, random_state=42,
            class_weight="balanced", C=0.1
        ))
    ])

    # Random Forest
    models["RF"] = Pipeline([
        ("imp", imputer),
        ("scl", scaler),
        ("mdl", RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42,
            class_weight="balanced", min_samples_leaf=8,
            max_features="sqrt"
        ))
    ])

    # Gradient Boosting
    models["GB"] = Pipeline([
        ("imp", imputer),
        ("scl", scaler),
        ("mdl", GradientBoostingClassifier(
            n_estimators=300, max_depth=3, random_state=42,
            learning_rate=0.03, min_samples_leaf=15,
            subsample=0.8
        ))
    ])

    # XGBoost
    if HAS_XGB:
        models["XGB"] = Pipeline([
            ("imp", imputer),
            ("scl", scaler),
            ("mdl", XGBClassifier(
                n_estimators=300, max_depth=4, random_state=42,
                learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss",
                scale_pos_weight=1137/642,  # class imbalance
                use_label_encoder=False,
                verbosity=0
            ))
        ])

    return models


def evaluate_models(X, y, models, cv_folds=5):
    """Cross-validation ile model değerlendirme"""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "average_precision": "average_precision",
    }

    results = {}
    for name, pipe in models.items():
        print(f"\n{'─'*55}")
        print(f"Model: {name}")
        print(f"{'─'*55}")

        try:
            cv_res = cross_validate(pipe, X, y, cv=cv,
                                     scoring=scoring,
                                     return_train_score=False)

            res = {
                metric: {
                    "mean": scores.mean(),
                    "std": scores.std()
                }
                for metric, scores in cv_res.items()
                if metric.startswith("test_")
            }
            results[name] = res

            print(f"  Accuracy         : "
                  f"{res['test_accuracy']['mean']:.4f} "
                  f"± {res['test_accuracy']['std']:.4f}")
            print(f"  AUC-ROC          : "
                  f"{res['test_roc_auc']['mean']:.4f} "
                  f"± {res['test_roc_auc']['std']:.4f}")
            print(f"  F1               : "
                  f"{res['test_f1']['mean']:.4f} "
                  f"± {res['test_f1']['std']:.4f}")
            print(f"  Precision        : "
                  f"{res['test_precision']['mean']:.4f} "
                  f"± {res['test_precision']['std']:.4f}")
            print(f"  Recall           : "
                  f"{res['test_recall']['mean']:.4f} "
                  f"± {res['test_recall']['std']:.4f}")
            print(f"  Avg Precision    : "
                  f"{res['test_average_precision']['mean']:.4f} "
                  f"± {res['test_average_precision']['std']:.4f}")

        except Exception as e:
            print(f"  HATA: {e}")
            results[name] = None

    return results


def analyze_feature_importance(X, y, feature_names, top_n=20):
    """Feature importance analizi - permutation based"""
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE ANALİZİ")
    print(f"{'='*70}")

    # RF ile fit
    pipe_rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("mdl", RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=42,
            class_weight="balanced", min_samples_leaf=8
        ))
    ])
    pipe_rf.fit(X, y)

    # Impute edilmiş X
    X_imp = pipe_rf.named_steps["imp"].transform(X)
    X_scl = pipe_rf.named_steps["scl"].transform(X_imp)
    X_proc = pd.DataFrame(X_scl, columns=feature_names)

    # 1. Built-in importance
    imp_builtin = pipe_rf.named_steps["mdl"].feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "builtin_importance": imp_builtin
    }).sort_values("builtin_importance", ascending=False)

    print(f"\nBuilt-in Feature Importance (Top {top_n}):")
    for _, row in imp_df.head(top_n).iterrows():
        bar = "█" * int(row["builtin_importance"] * 150)
        print(f"  {row['feature']:<40s} "
              f"{row['builtin_importance']:.4f} {bar}")

    # 2. Permutation importance (daha güvenilir)
    try:
        perm_imp = permutation_importance(
            pipe_rf.named_steps["mdl"], X_proc, y,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_df = pd.DataFrame({
            "feature": feature_names,
            "perm_importance": perm_imp.importances_mean,
            "perm_std": perm_imp.importances_std
        }).sort_values("perm_importance", ascending=False)

        print(f"\nPermutation Feature Importance (Top {top_n}):")
        for _, row in perm_df.head(top_n).iterrows():
            bar = "█" * max(0, int(row["perm_importance"] * 200))
            print(f"  {row['feature']:<40s} "
                  f"{row['perm_importance']:.4f} "
                  f"± {row['perm_std']:.4f} {bar}")

        return imp_df, perm_df

    except Exception as e:
        print(f"  Permutation importance hatası: {e}")
        return imp_df, None


def threshold_analysis(X, y, model_name="GB"):
    """Eşik değeri optimizasyonu"""
    print(f"\n{'='*70}")
    print(f"EŞİK ANALİZİ - {model_name}")
    print(f"{'='*70}")

    models = build_models()
    if model_name not in models:
        model_name = "GB"

    pipe = models[model_name]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_probs = []
    all_true = []

    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)[:, 1]
        all_probs.extend(probs)
        all_true.extend(y_val)

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    # Precision-Recall eğrisi
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    for th in thresholds:
        preds = (all_probs >= th).astype(int)
        tp = int(((preds == 1) & (all_true == 1)).sum())
        fp = int(((preds == 1) & (all_true == 0)).sum())
        fn = int(((preds == 0) & (all_true == 1)).sum())
        tn = int(((preds == 0) & (all_true == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / len(all_true)

        results.append({
            "threshold": th,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

    th_df = pd.DataFrame(results)
    th_df.to_csv("output/threshold_analysis.csv",
                  index=False, encoding="utf-8-sig")

    # En iyi eşikler
    best_f1_idx = th_df["f1"].idxmax()
    best_bal_idx = (th_df["precision"] - th_df["recall"]).abs().idxmin()

    print(f"\nEn iyi F1 eşiği:")
    row = th_df.iloc[best_f1_idx]
    print(f"  Eşik={row['threshold']:.2f}  "
          f"F1={row['f1']:.4f}  "
          f"Prec={row['precision']:.4f}  "
          f"Recall={row['recall']:.4f}")
    print(f"  TP={row['tp']:.0f}  FP={row['fp']:.0f}  "
          f"FN={row['fn']:.0f}  TN={row['tn']:.0f}")

    print(f"\nEn dengeli eşik (Prec ≈ Recall):")
    row2 = th_df.iloc[best_bal_idx]
    print(f"  Eşik={row2['threshold']:.2f}  "
          f"F1={row2['f1']:.4f}  "
          f"Prec={row2['precision']:.4f}  "
          f"Recall={row2['recall']:.4f}")

    print(f"\nEşik tablosu (tüm değerler):")
    print(f"  {'Eşik':>6} {'Prec':>7} {'Recall':>8} "
          f"{'F1':>7} {'Acc':>7} {'TP':>5} {'FP':>5}")
    print(f"  {'─'*55}")
    for _, row in th_df.iterrows():
        flag = " ← OPT" if row.name == best_f1_idx else ""
        print(f"  {row['threshold']:>6.2f} "
              f"{row['precision']:>7.4f} "
              f"{row['recall']:>8.4f} "
              f"{row['f1']:>7.4f} "
              f"{row['accuracy']:>7.4f} "
              f"{row['tp']:>5.0f} "
              f"{row['fp']:>5.0f}{flag}")

    return th_df


def fault_specific_models(real_df, ctrl_df):
    """Fay tipine özel model değerlendirmesi"""
    print(f"\n{'='*70}")
    print("FAY TİPİNE ÖZEL GELİŞMİŞ MODELLER")
    print(f"{'='*70}")

    fault_results = {}
    for fault_type in ["REVERSE", "STRIKE_SLIP", "NORMAL"]:
        print(f"\n{'━'*50}")
        print(f"FAY: {fault_type}")

        real_ft = real_df[
            (real_df["radius_km"] == 200) &
            (real_df["main_fault_type"] == fault_type)
        ].copy()

        ft_col = ("parent_fault_type" if "parent_fault_type" in ctrl_df.columns
                  else "main_fault_type")
        ctrl_ft = ctrl_df[
            (ctrl_df["radius_km"] == 200) &
            (ctrl_df.get(ft_col, pd.Series()) == fault_type)
        ].copy()

        real_ft = engineer_features(real_ft, is_real=True)
        ctrl_ft = engineer_features(ctrl_ft, is_real=False)

        real_ft["target"] = 1
        ctrl_ft["target"] = 0

        feature_list = get_feature_list()
        avail_r = [f for f in feature_list if f in real_ft.columns]
        avail_c = [f for f in feature_list if f in ctrl_ft.columns]
        common = list(set(avail_r) & set(avail_c))

        combined = pd.concat([
            real_ft[common + ["target"]],
            ctrl_ft[common + ["target"]]
        ], ignore_index=True)

        X_ft = combined[common]
        y_ft = combined["target"]

        if len(X_ft) < 40 or y_ft.sum() < 15:
            print(f"  Yetersiz veri: {len(X_ft)} örnek")
            continue

        print(f"  Veri: {len(X_ft)} ({int(y_ft.sum())} real, "
              f"{int((y_ft==0).sum())} ctrl)")

        models = build_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_auc = 0
        best_name = ""
        ft_res = {}
        for name, pipe in models.items():
            try:
                auc = cross_val_score(pipe, X_ft, y_ft, cv=cv,
                                       scoring="roc_auc")
                f1 = cross_val_score(pipe, X_ft, y_ft, cv=cv,
                                      scoring="f1")
                rec = cross_val_score(pipe, X_ft, y_ft, cv=cv,
                                       scoring="recall")
                print(f"  {name}: AUC={auc.mean():.4f}±{auc.std():.4f}  "
                      f"F1={f1.mean():.4f}  Recall={rec.mean():.4f}")
                ft_res[name] = {"auc": auc.mean(), "f1": f1.mean(),
                                "recall": rec.mean()}
                if auc.mean() > best_auc:
                    best_auc = auc.mean()
                    best_name = name
            except Exception as e:
                print(f"  {name}: HATA - {e}")

        fault_results[fault_type] = {
            "best_model": best_name,
            "best_auc": best_auc,
            "results": ft_res
        }

        # En iyi modelin feature importance'ı
        if best_name and best_name in models:
            try:
                best_pipe = models[best_name]
                best_pipe.fit(X_ft, y_ft)
                imp = best_pipe.named_steps["mdl"].feature_importances_
                imp_df = pd.DataFrame({
                    "feature": common,
                    "importance": imp
                }).sort_values("importance", ascending=False)
                print(f"\n  Top 5 feature ({best_name}):")
                for _, row in imp_df.head(5).iterrows():
                    print(f"    {row['feature']:<35s} {row['importance']:.4f}")
            except Exception as e:
                print(f"  Feature importance hatası: {e}")

    return fault_results


# ═══════════════════════════════════════════════════════════
# 4. ANA FONKSİYON
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SeismoPattern - FAZ 4 GELİŞMİŞ ML")
    print("=" * 70)
    print(f"XGBoost: {'✅' if HAS_XGB else '❌'}")
    print(f"SMOTE:   {'✅' if HAS_SMOTE else '❌'}")

    # Veri yükle
    real = pd.read_csv("output/gcmt_precursor_features.csv")
    ctrl = pd.read_csv("output/gcmt_control_features.csv")

    # Veri hazırla
    print("\n" + "─" * 50)
    print("VERİ HAZIRLAMA")
    print("─" * 50)
    X, y, fault, features = prepare_dataset(real, ctrl, radius=200)

    # Model listesi
    print("\n" + "─" * 50)
    print("MODEL DEĞERLENDİRME (5-Fold CV)")
    print("─" * 50)
    models = build_models()
    results = evaluate_models(X, y, models)

    # Özet karşılaştırma
    print(f"\n{'='*70}")
    print("MODEL KARŞILAŞTIRMA ÖZETİ")
    print(f"{'='*70}")
    print(f"  {'Model':<8} {'AUC-ROC':>10} {'F1':>10} "
          f"{'Precision':>12} {'Recall':>10}")
    print(f"  {'─'*55}")
    for name, res in results.items():
        if res:
            print(f"  {name:<8} "
                  f"{res['test_roc_auc']['mean']:>10.4f} "
                  f"{res['test_f1']['mean']:>10.4f} "
                  f"{res['test_precision']['mean']:>12.4f} "
                  f"{res['test_recall']['mean']:>10.4f}")

    # Feature importance
    imp_df, perm_df = analyze_feature_importance(X, y, features)

    # Eşik analizi
    th_df = threshold_analysis(X, y, model_name="GB")

    # Fay tipine özel modeller
    fault_results = fault_specific_models(real, ctrl)

    # Sonuç özeti
    print(f"\n{'='*70}")
    print("FAZ 4 ÖZET")
    print(f"{'='*70}")
    print("\nFay tipine göre en iyi model:")
    for ft, res in fault_results.items():
        if res:
            print(f"  {ft:<15}: {res['best_model']} "
                  f"(AUC={res['best_auc']:.4f})")

    # Kaydet
    imp_df.to_csv("output/feature_importance.csv",
                   index=False, encoding="utf-8-sig")
    if perm_df is not None:
        perm_df.to_csv("output/permutation_importance.csv",
                        index=False, encoding="utf-8-sig")

    print("\nKaydedildi:")
    print("  output/feature_importance.csv")
    print("  output/permutation_importance.csv")
    print("  output/threshold_analysis.csv")


if __name__ == "__main__":
    main()