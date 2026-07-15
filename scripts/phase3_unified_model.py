#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 3 Birleşik Model
======================================
GCMT + USGS sonuçlarını birleştirerek:
1. Şablon tiplerini sınıflandırır (Tip A / Tip B / Tip C)
2. Gelişmiş öncü skor sistemi kurar
3. İlk ML denemesini başlatır
4. Fay tipine göre alt modeller oluşturur
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve,
                              average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════
# 1. ŞABlon TİPİ SINIFLANDIRMA
# ═══════════════════════════════════════════════════════════

def classify_pattern_type(row):
    """
    Her büyük deprem öncü penceresini Tip A / Tip B / Tip C olarak sınıfla.

    Kurallar:
      Tip A (Aktivasyon):
        - quiescence_ratio >= 1.0 VE accel_90d >= 1.0
        - VEYA w1_n_events > median VE quiescence_ratio > 0.8

      Tip B (Sessizlik):
        - quiescence_ratio < 0.7
        - VEYA (ratio_1y < 0.5 ve veri yeterliyse)

      Tip C (Belirsiz):
        - Hiçbirine uymayan veya yetersiz veri
    """
    qr = row.get("quiescence_ratio", np.nan)
    acc = row.get("accel_90d", np.nan)
    n_w1 = row.get("w1_n_events", 0)
    n_w3 = row.get("w3_n_events", 0)

    # Yetersiz veri kontrolü
    if n_w3 < 3:
        return "TIP_C"

    if pd.isna(qr):
        return "TIP_C"

    # Tip B: Sessizlik
    if qr < 0.5:
        return "TIP_B"

    # Tip A: Aktivasyon
    if qr >= 1.0 and not pd.isna(acc) and acc >= 1.0:
        return "TIP_A"

    if qr >= 0.8 and n_w1 >= 3:
        return "TIP_A"

    # Sınırda olan: quiescence 0.5-0.8 arası
    if 0.5 <= qr < 0.8:
        if not pd.isna(acc) and acc >= 1.5:
            return "TIP_A"  # Sessizlik var ama son dakika hızlanma
        return "TIP_B"

    return "TIP_C"


# ═══════════════════════════════════════════════════════════
# 2. GELİŞMİŞ SKOR SİSTEMİ
# ═══════════════════════════════════════════════════════════

def compute_advanced_score(row, weights=None):
    """
    Gelişmiş öncü skor hesabı.
    Fay tipine göre ağırlıklar değişir.
    """
    if weights is None:
        # Varsayılan ağırlıklar (GCMT + USGS bulgularından)
        weights = {
            "b_value_signal": 0.30,     # En güçlü sinyal
            "max_mw_signal": 0.20,      # İkinci en güçlü
            "count_signal": 0.15,       # Olay sayısı
            "spatial_signal": 0.15,     # Mekânsal odaklanma
            "moment_signal": 0.10,      # Kümülatif moment
            "acceleration_signal": 0.10 # Son dönem hızlanma
        }

    score = 0.0
    weight_used = 0.0

    # b-değeri sinyali (düşük b = yüksek sinyal)
    b_val = row.get("w3_b_value", np.nan)
    if pd.notna(b_val):
        # b < 0.8 → güçlü sinyal, b > 1.2 → negatif sinyal
        b_signal = max(-2, min(2, (1.0 - b_val) / 0.2))
        score += b_signal * weights["b_value_signal"]
        weight_used += weights["b_value_signal"]

    # max_mw sinyali
    max_mw = row.get("w1_max_mw", np.nan)
    if pd.notna(max_mw):
        mw_signal = max(-2, min(2, (max_mw - 5.5) / 0.5))
        score += mw_signal * weights["max_mw_signal"]
        weight_used += weights["max_mw_signal"]

    # Olay sayısı sinyali
    n_events = row.get("w1_n_events", 0)
    if n_events > 0:
        count_signal = max(-1, min(2, np.log10(n_events) - 0.5))
        score += count_signal * weights["count_signal"]
        weight_used += weights["count_signal"]

    # Mekânsal odaklanma sinyali
    mean_dist = row.get("w1_mean_dist_km", np.nan)
    radius = row.get("radius_km", 200)
    if pd.notna(mean_dist) and radius > 0:
        # Normalize: düşük oran = daha yakın = daha iyi
        dist_ratio = mean_dist / radius
        spatial_signal = max(-2, min(2, (0.6 - dist_ratio) / 0.2))
        score += spatial_signal * weights["spatial_signal"]
        weight_used += weights["spatial_signal"]

    # Kümülatif moment sinyali
    cum_mom = row.get("w1_cum_moment_nm", np.nan)
    if pd.notna(cum_mom) and cum_mom > 0:
        log_mom = np.log10(cum_mom)
        mom_signal = max(-2, min(2, (log_mom - 18) / 1.0))
        score += mom_signal * weights["moment_signal"]
        weight_used += weights["moment_signal"]

    # Hızlanma sinyali
    accel = row.get("accel_90d", np.nan)
    if pd.notna(accel):
        if accel > 0:
            acc_signal = max(-1, min(2, np.log10(max(accel, 0.1))))
        else:
            acc_signal = -1
        score += acc_signal * weights["acceleration_signal"]
        weight_used += weights["acceleration_signal"]

    # Ağırlık normalizasyonu
    if weight_used > 0:
        score = score / weight_used
    else:
        score = np.nan

    return round(score, 4)


def fault_specific_weights(fault_type):
    """Fay tipine göre ağırlık ayarlaması"""
    if fault_type == "REVERSE":
        return {
            "b_value_signal": 0.30,
            "max_mw_signal": 0.25,      # Reverse'te max_mw çok güçlü
            "count_signal": 0.15,
            "spatial_signal": 0.10,
            "moment_signal": 0.10,
            "acceleration_signal": 0.10
        }
    elif fault_type == "STRIKE_SLIP":
        return {
            "b_value_signal": 0.25,
            "max_mw_signal": 0.15,
            "count_signal": 0.15,
            "spatial_signal": 0.20,      # SS'te mekânsal odaklanma güçlü
            "moment_signal": 0.15,
            "acceleration_signal": 0.10
        }
    elif fault_type == "NORMAL":
        return {
            "b_value_signal": 0.25,
            "max_mw_signal": 0.15,
            "count_signal": 0.20,
            "spatial_signal": 0.10,
            "moment_signal": 0.10,
            "acceleration_signal": 0.20  # Normal'de hızlanma daha anlamlı
        }
    else:
        return None


# ═══════════════════════════════════════════════════════════
# 3. ML MODEL HAZIRLIGI
# ═══════════════════════════════════════════════════════════

def prepare_ml_data(real_df, ctrl_df, radius=200):
    """ML için veri hazırla: REAL=1, CONTROL=0"""
    real = real_df[real_df["radius_km"] == radius].copy()
    ctrl = ctrl_df[ctrl_df["radius_km"] == radius].copy()

    real["target"] = 1
    ctrl["target"] = 0

    # Ortak feature sütunları
    feature_cols = [
        "quiescence_ratio", "accel_90d", "monthly_slope_36m",
        "count_0_1y", "count_1_2y", "count_2_3y",
        "w1_n_events", "w1_max_mw", "w1_mean_mw", "w1_std_mw",
        "w1_b_value", "w1_cum_moment_nm",
        "w1_mean_depth_km", "w1_std_depth_km",
        "w1_mean_dist_km", "w1_std_dist_km",
        "w1_migration_slope_km_day",
        "w3_n_events", "w3_max_mw", "w3_mean_mw",
        "w3_b_value", "w3_cum_moment_nm",
        "w3_mean_depth_km", "w3_mean_dist_km",
        "w3_migration_slope_km_day",
    ]

    available_real = [c for c in feature_cols if c in real.columns]
    available_ctrl = [c for c in feature_cols if c in ctrl.columns]
    common_cols = list(set(available_real) & set(available_ctrl))

    if not common_cols:
        print("HATA: Ortak feature sütunu bulunamadı!")
        return None, None, None

    # Birleştir
    real_sub = real[common_cols + ["target"]].copy()
    ctrl_sub = ctrl[common_cols + ["target"]].copy()

    combined = pd.concat([real_sub, ctrl_sub], ignore_index=True)

    # cum_moment_nm log dönüşümü
    for col in ["w1_cum_moment_nm", "w3_cum_moment_nm"]:
        if col in combined.columns:
            combined[f"log_{col}"] = np.log10(
                combined[col].clip(lower=1)
            )
            common_cols.append(f"log_{col}")
            common_cols.remove(col)

    X = combined[common_cols]
    y = combined["target"]

    return X, y, common_cols


def run_ml_models(X, y, feature_names):
    """Birden fazla ML modeli dene"""
    print("\n" + "=" * 70)
    print("MAKİNE ÖĞRENMESİ - İLK DENEMELER")
    print("=" * 70)

    # Imputer + Scaler pipeline
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    models = {
        "Lojistik Regresyon": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42,
            class_weight="balanced", min_samples_leaf=5
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42,
            learning_rate=0.05, min_samples_leaf=10
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Cross-validation
        try:
            accuracy = cross_val_score(pipe, X, y, cv=cv,
                                        scoring="accuracy")
            auc = cross_val_score(pipe, X, y, cv=cv,
                                   scoring="roc_auc")
            f1 = cross_val_score(pipe, X, y, cv=cv,
                                  scoring="f1")
            precision = cross_val_score(pipe, X, y, cv=cv,
                                         scoring="precision")
            recall = cross_val_score(pipe, X, y, cv=cv,
                                      scoring="recall")

            results[name] = {
                "accuracy": accuracy,
                "auc": auc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

            print(f"\n{'─'*50}")
            print(f"Model: {name}")
            print(f"{'─'*50}")
            print(f"  Accuracy  : {accuracy.mean():.4f} ± {accuracy.std():.4f}")
            print(f"  AUC-ROC   : {auc.mean():.4f} ± {auc.std():.4f}")
            print(f"  F1        : {f1.mean():.4f} ± {f1.std():.4f}")
            print(f"  Precision : {precision.mean():.4f} ± {precision.std():.4f}")
            print(f"  Recall    : {recall.mean():.4f} ± {recall.std():.4f}")

        except Exception as e:
            print(f"\n  {name}: HATA - {e}")
            results[name] = None

    # En iyi model ile feature importance
    print(f"\n{'='*70}")
    print("FEATURE ÖNEMLİLİĞİ (Random Forest)")
    print(f"{'='*70}")

    try:
        pipe_rf = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42,
                class_weight="balanced", min_samples_leaf=5
            )),
        ])
        pipe_rf.fit(X, y)

        importances = pipe_rf.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        for _, row in imp_df.head(15).iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"  {row['feature']:<40s} {row['importance']:.4f} {bar}")

    except Exception as e:
        print(f"  Feature importance hatası: {e}")

    # Gradient Boosting ile de feature importance
    print(f"\n{'='*70}")
    print("FEATURE ÖNEMLİLİĞİ (Gradient Boosting)")
    print(f"{'='*70}")

    try:
        pipe_gb = Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, random_state=42,
                learning_rate=0.05, min_samples_leaf=10
            )),
        ])
        pipe_gb.fit(X, y)

        importances_gb = pipe_gb.named_steps["model"].feature_importances_
        imp_gb_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances_gb
        }).sort_values("importance", ascending=False)

        for _, row in imp_gb_df.head(15).iterrows():
            bar = "█" * int(row["importance"] * 100)
            print(f"  {row['feature']:<40s} {row['importance']:.4f} {bar}")

    except Exception as e:
        print(f"  Feature importance hatası: {e}")

    return results


# ═══════════════════════════════════════════════════════════
# 4. ANA FONKSİYON
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SeismoPattern - FAZ 3 BİRLEŞİK MODEL")
    print("=" * 70)

    # Dosyaları oku
    real = pd.read_csv("output/gcmt_precursor_features.csv")
    ctrl = pd.read_csv("output/gcmt_control_features.csv")

    # ─── 1. Şablon tipi sınıflandırma ────────────────────────
    print("\n" + "─" * 50)
    print("1. ŞABLON TİPİ SINIFLANDIRMA")
    print("─" * 50)

    # 200km yarıçap üzerinde çalış
    real_200 = real[real["radius_km"] == 200].copy()

    real_200["pattern_type"] = real_200.apply(classify_pattern_type, axis=1)

    print("\nŞablon tipi dağılımı (200km):")
    pt_counts = real_200["pattern_type"].value_counts()
    for pt, count in pt_counts.items():
        pct = count / len(real_200) * 100
        print(f"  {pt}: {count} ({pct:.1f}%)")

    # Fay tipine göre şablon dağılımı
    print("\nFay tipine göre şablon dağılımı:")
    ct = pd.crosstab(real_200["main_fault_type"], real_200["pattern_type"],
                      margins=True)
    print(ct.to_string())

    # Şablon tiplerine göre metrik karşılaştırma
    print("\nŞablon tiplerine göre median metrikler:")
    pattern_metrics = [
        "quiescence_ratio", "accel_90d", "w1_n_events",
        "w1_max_mw", "w3_b_value", "w1_mean_dist_km"
    ]
    avail_metrics = [m for m in pattern_metrics if m in real_200.columns]
    print(real_200.groupby("pattern_type")[avail_metrics].median().to_string())

    # ─── 2. Gelişmiş skor sistemi ────────────────────────────
    print("\n" + "─" * 50)
    print("2. GELİŞMİŞ SKOR SİSTEMİ")
    print("─" * 50)

    # Genel skor
    real_200["advanced_score"] = real_200.apply(
        compute_advanced_score, axis=1
    )

    # Fay tipine özel skor
    def fault_score(row):
        ft = row.get("main_fault_type", "UNKNOWN")
        w = fault_specific_weights(ft)
        return compute_advanced_score(row, weights=w)

    real_200["fault_specific_score"] = real_200.apply(fault_score, axis=1)

    print("\nGenel skor özeti:")
    print(real_200["advanced_score"].describe())

    print("\nFay tipine göre skor:")
    print(real_200.groupby("main_fault_type")[
        ["advanced_score", "fault_specific_score"]
    ].agg(["median", "mean", "std"]).to_string())

    print("\nŞablon tipine göre skor:")
    print(real_200.groupby("pattern_type")[
        ["advanced_score", "fault_specific_score"]
    ].agg(["median", "mean", "std"]).to_string())

    # Kontrol grubu için skor
    ctrl_200 = ctrl[ctrl["radius_km"] == 200].copy()
    ctrl_200["advanced_score"] = ctrl_200.apply(
        compute_advanced_score, axis=1
    )

    print(f"\nGerçek median skor  : {real_200['advanced_score'].median():.4f}")
    print(f"Kontrol median skor : {ctrl_200['advanced_score'].median():.4f}")

    # İstatistiksel test
    r_scores = real_200["advanced_score"].dropna()
    c_scores = ctrl_200["advanced_score"].dropna()
    if len(r_scores) > 10 and len(c_scores) > 10:
        _, p = stats.mannwhitneyu(r_scores, c_scores, alternative="two-sided")
        pooled = np.sqrt((r_scores.std()**2 + c_scores.std()**2) / 2)
        d = (r_scores.mean() - c_scores.mean()) / pooled if pooled > 0 else np.nan
        print(f"Mann-Whitney p       : {p:.8f}")
        print(f"Cohen's d            : {d:+.4f}")

    # ─── 3. En yüksek/düşük skorlu olaylar ──────────────────
    print("\n" + "─" * 50)
    print("3. EN YÜKSEK VE EN DÜŞÜK SKORLU OLAYLAR")
    print("─" * 50)

    show_cols = [
        "main_event_id", "main_datetime_utc", "main_mw",
        "main_region", "main_fault_type", "pattern_type",
        "advanced_score", "fault_specific_score",
        "w1_max_mw", "w3_b_value", "quiescence_ratio", "accel_90d"
    ]
    avail_show = [c for c in show_cols if c in real_200.columns]

    print("\nEn yüksek skorlu 10:")
    print(real_200.nlargest(10, "advanced_score")[avail_show].to_string(index=False))

    print("\nEn düşük skorlu 10:")
    print(real_200.nsmallest(10, "advanced_score")[avail_show].to_string(index=False))

    # ─── 4. ML Modelleri ─────────────────────────────────────
    print("\n" + "─" * 50)
    print("4. MAKİNE ÖĞRENMESİ MODELLERİ")
    print("─" * 50)

    X, y, feature_names = prepare_ml_data(real, ctrl, radius=200)

    if X is not None:
        print(f"\nVeri seti: {len(X)} örnek ({y.sum()} gerçek, {len(y)-y.sum()} kontrol)")
        print(f"Feature sayısı: {len(feature_names)}")
        print(f"Features: {feature_names}")

        ml_results = run_ml_models(X, y, feature_names)

        # Sonuçları kaydet
        ml_summary = []
        for name, res in ml_results.items():
            if res:
                ml_summary.append({
                    "model": name,
                    "accuracy_mean": res["accuracy"].mean(),
                    "accuracy_std": res["accuracy"].std(),
                    "auc_mean": res["auc"].mean(),
                    "auc_std": res["auc"].std(),
                    "f1_mean": res["f1"].mean(),
                    "f1_std": res["f1"].std(),
                    "precision_mean": res["precision"].mean(),
                    "recall_mean": res["recall"].mean(),
                })
        ml_df = pd.DataFrame(ml_summary)
        ml_df.to_csv("output/ml_results_summary.csv",
                      index=False, encoding="utf-8-sig")

    # ─── 5. Fay tipine göre ayrı ML ─────────────────────────
    print("\n" + "─" * 50)
    print("5. FAY TİPİNE GÖRE ML MODELLERİ")
    print("─" * 50)

    for fault_type in ["REVERSE", "STRIKE_SLIP", "NORMAL"]:
        print(f"\n{'━'*40}")
        print(f"FAY TİPİ: {fault_type}")
        print(f"{'━'*40}")

        real_ft = real[
            (real["radius_km"] == 200) &
            (real["main_fault_type"] == fault_type)
        ].copy()

        # Kontrol verisinde fault type eşlemesi
        ft_col = "parent_fault_type" if "parent_fault_type" in ctrl.columns else "main_fault_type"
        ctrl_ft = ctrl[
            (ctrl["radius_km"] == 200) &
            (ctrl.get(ft_col, pd.Series()) == fault_type)
        ].copy()

        if len(real_ft) < 20 or len(ctrl_ft) < 20:
            print(f"  Yetersiz veri: {len(real_ft)} real, {len(ctrl_ft)} ctrl")
            continue

        real_ft["target"] = 1
        ctrl_ft["target"] = 0

        feature_cols = [
            "quiescence_ratio", "accel_90d", "monthly_slope_36m",
            "count_0_1y", "w1_n_events", "w1_max_mw",
            "w1_b_value", "w1_mean_dist_km",
            "w3_n_events", "w3_b_value",
        ]
        avail = [c for c in feature_cols
                 if c in real_ft.columns and c in ctrl_ft.columns]

        combined_ft = pd.concat([
            real_ft[avail + ["target"]],
            ctrl_ft[avail + ["target"]]
        ], ignore_index=True)

        X_ft = combined_ft[avail]
        y_ft = combined_ft["target"]

        print(f"  Veri: {len(X_ft)} ({y_ft.sum()} real, {len(y_ft)-y_ft.sum()} ctrl)")

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=150, max_depth=3, random_state=42,
                learning_rate=0.05, min_samples_leaf=5
            )),
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            auc = cross_val_score(pipe, X_ft, y_ft, cv=cv,
                                   scoring="roc_auc")
            f1 = cross_val_score(pipe, X_ft, y_ft, cv=cv,
                                  scoring="f1")
            print(f"  AUC-ROC : {auc.mean():.4f} ± {auc.std():.4f}")
            print(f"  F1      : {f1.mean():.4f} ± {f1.std():.4f}")

            pipe.fit(X_ft, y_ft)
            importances = pipe.named_steps["model"].feature_importances_
            imp_df = pd.DataFrame({
                "feature": avail,
                "importance": importances
            }).sort_values("importance", ascending=False)
            print(f"  Top features:")
            for _, row in imp_df.head(5).iterrows():
                print(f"    {row['feature']:<30s} {row['importance']:.4f}")
        except Exception as e:
            print(f"  Hata: {e}")

    # ─── Kaydet ──────────────────────────────────────────────
    real_200.to_csv("output/gcmt_real_with_patterns.csv",
                     index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("BİRLEŞİK MODEL TAMAMLANDI")
    print("=" * 70)
    print(f"  Şablon sınıflandırması: output/gcmt_real_with_patterns.csv")
    print(f"  ML sonuçları: output/ml_results_summary.csv")


if __name__ == "__main__":
    main()