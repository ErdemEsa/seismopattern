#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 7: Kontrol Penceresi Temizleme ve Yeniden Model
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      cross_validate)
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


# ═══════════════════════════════════════════════════════════
# 0. SÜTUN ADLARINI TESPİT ET
# ═══════════════════════════════════════════════════════════

def detect_columns(df, candidates):
    """Aday sütun adlarından hangisi mevcut, onu döndür."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ═══════════════════════════════════════════════════════════
# 1. KONTROL PENCERESİ TEMİZLEME
# ═══════════════════════════════════════════════════════════

def clean_control_windows(ctrl_df, all_majors_df,
                           contamination_radius_km=300,
                           contamination_years=3):
    """
    Kontrol penceresi [ref_time-3y, ref_time] aralığında
    bölgede Mw7+ deprem varsa → kirli say ve kaldır.
    """
    ctrl = ctrl_df.copy()

    # Zaman sütununu bul
    time_col = detect_columns(ctrl, [
        "ref_datetime_utc", "datetime_utc", "main_datetime_utc"
    ])
    lat_col = detect_columns(ctrl, [
        "ref_lat", "lat", "main_lat"
    ])
    lon_col = detect_columns(ctrl, [
        "ref_lon", "lon", "main_lon"
    ])

    print(f"  Kontrol zaman sütunu : {time_col}")
    print(f"  Kontrol lat sütunu   : {lat_col}")
    print(f"  Kontrol lon sütunu   : {lon_col}")

    if time_col is None or lat_col is None or lon_col is None:
        print("  ⚠️  Gerekli sütunlar bulunamadı!")
        print(f"  Mevcut sütunlar: {list(ctrl.columns[:20])}")
        return ctrl, pd.DataFrame()

    ctrl["_ref_time"] = pd.to_datetime(ctrl[time_col], errors="coerce")
    ctrl["_ref_lat"] = pd.to_numeric(ctrl[lat_col], errors="coerce")
    ctrl["_ref_lon"] = pd.to_numeric(ctrl[lon_col], errors="coerce")

    majors = all_majors_df.copy()
    majors["datetime_utc"] = pd.to_datetime(
        majors["datetime_utc"], errors="coerce"
    )
    majors = majors.dropna(subset=["datetime_utc", "eff_lat", "eff_lon"])

    contaminated_idx = []
    clean_idx = []

    for i, row in ctrl.iterrows():
        ref_time = row["_ref_time"]
        lat0 = row["_ref_lat"]
        lon0 = row["_ref_lon"]

        if pd.isna(ref_time) or pd.isna(lat0) or pd.isna(lon0):
            clean_idx.append(i)
            continue

        window_start = ref_time - pd.Timedelta(
            days=int(contamination_years * 365.25)
        )

        # Bu zaman penceresindeki büyük depremler
        candidates = majors[
            (majors["datetime_utc"] >= window_start) &
            (majors["datetime_utc"] <= ref_time)
        ]

        if len(candidates) == 0:
            clean_idx.append(i)
            continue

        dists = haversine_km(
            lat0, lon0,
            candidates["eff_lat"].values,
            candidates["eff_lon"].values
        )

        if (dists <= contamination_radius_km).any():
            contaminated_idx.append(i)
        else:
            clean_idx.append(i)

    ctrl_clean = ctrl.loc[clean_idx].drop(
        columns=["_ref_time", "_ref_lat", "_ref_lon"], errors="ignore"
    ).copy()
    ctrl_dirty = ctrl.loc[contaminated_idx].drop(
        columns=["_ref_time", "_ref_lat", "_ref_lon"], errors="ignore"
    ).copy()

    print(f"\n  Toplam kontrol : {len(ctrl)}")
    print(f"  Kirli          : {len(contaminated_idx)} "
          f"({len(contaminated_idx)/len(ctrl)*100:.1f}%)")
    print(f"  Temiz          : {len(ctrl_clean)} "
          f"({len(ctrl_clean)/len(ctrl)*100:.1f}%)")

    return ctrl_clean, ctrl_dirty


# ═══════════════════════════════════════════════════════════
# 2. FEATURE SETİ
# ═══════════════════════════════════════════════════════════

CORE_FEATURES = [
    "count_0_1y", "count_1_2y", "count_2_3y",
    "w1_n_events", "w3_n_events",
    "quiescence_ratio", "accel_90d",
    "monthly_slope_36m",
    "count_linear_trend", "count_accel_ratio",
    "w1_mean_mw", "w1_std_mw", "w1_max_mw",
    "w3_mean_mw", "w3_max_mw",
    "w1_b_value", "w3_b_value", "b_drop_w3_w1",
    "w1_mean_depth_km", "w1_std_depth_km",
    "w3_mean_depth_km", "depth_change_km",
    "w1_mean_dist_km", "w1_std_dist_km",
    "w3_mean_dist_km", "spatial_focus_change",
    "w1_migration_slope_km_day",
    "w3_migration_slope_km_day",
    "z_rate_1y", "z_rate_3y",
    "z_b_value_1y", "z_b_value_3y",
    "z_max_mw_1y", "z_depth_1y", "z_dist_1y",
]


def add_derived_features(df):
    """Türetilmiş feature'ları hesapla ve ekle"""
    df = df.copy()

    c0 = df.get("count_0_1y", pd.Series(0, index=df.index)).fillna(0)
    c1 = df.get("count_1_2y", pd.Series(0, index=df.index)).fillna(0)
    c2 = df.get("count_2_3y", pd.Series(0, index=df.index)).fillna(0)
    df["count_linear_trend"] = c0 - c2
    avg = (c1 + c2) / 2.0 + 1e-6
    df["count_accel_ratio"] = c0 / avg

    b1 = df.get("w1_b_value", pd.Series(np.nan, index=df.index))
    b3 = df.get("w3_b_value", pd.Series(np.nan, index=df.index))
    df["b_drop_w3_w1"] = b3 - b1

    d1 = df.get("w1_mean_dist_km", pd.Series(np.nan, index=df.index))
    d3 = df.get("w3_mean_dist_km", pd.Series(np.nan, index=df.index))
    df["spatial_focus_change"] = d3 - d1

    dep1 = df.get("w1_mean_depth_km", pd.Series(np.nan, index=df.index))
    dep3 = df.get("w3_mean_depth_km", pd.Series(np.nan, index=df.index))
    df["depth_change_km"] = dep1 - dep3

    return df


def prepare_features(real_df, ctrl_df, feature_list=None):
    """Feature tablosu hazırla"""
    if feature_list is None:
        feature_list = CORE_FEATURES

    real = add_derived_features(real_df.copy())
    ctrl = add_derived_features(ctrl_df.copy())

    real["target"] = 1
    ctrl["target"] = 0

    # Fay tipi sütununu bul
    ft_col_r = detect_columns(real, ["fault_type", "main_fault_type"])
    ft_col_c = detect_columns(ctrl, ["fault_type", "parent_fault_type",
                                      "main_fault_type"])

    avail_r = [f for f in feature_list if f in real.columns]
    avail_c = [f for f in feature_list if f in ctrl.columns]
    common = list(set(avail_r) & set(avail_c))

    # Birleştir
    cols_r = common + ["target"] + ([ft_col_r] if ft_col_r else [])
    cols_c = common + ["target"] + ([ft_col_c] if ft_col_c else [])

    df_r = real[[c for c in cols_r if c in real.columns]].copy()
    df_c = ctrl[[c for c in cols_c if c in ctrl.columns]].copy()

    if ft_col_r and ft_col_r != "fault_type":
        df_r = df_r.rename(columns={ft_col_r: "fault_type"})
    if ft_col_c and ft_col_c != "fault_type":
        df_c = df_c.rename(columns={ft_col_c: "fault_type"})

    combined = pd.concat([df_r, df_c], ignore_index=True)

    X = combined[common]
    y = combined["target"]
    fault = combined.get(
        "fault_type", pd.Series("UNKNOWN", index=combined.index)
    )

    return X, y, fault, common


# ═══════════════════════════════════════════════════════════
# 3. MODEL DEĞERLENDİRME
# ═══════════════════════════════════════════════════════════

def build_models(pos_weight=1.0):
    """Model konfigürasyonları"""
    models = {
        "LR": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", LogisticRegression(
                max_iter=2000, C=0.05,
                class_weight={0: 1, 1: 3},
                random_state=42
            ))
        ]),
        "RF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", RandomForestClassifier(
                n_estimators=300, max_depth=5,
                min_samples_leaf=5,
                class_weight={0: 1, 1: 2},
                max_features="sqrt", random_state=42
            ))
        ]),
        "GB": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", GradientBoostingClassifier(
                n_estimators=300, max_depth=3,
                learning_rate=0.03,
                min_samples_leaf=10,
                subsample=0.8, random_state=42
            ))
        ]),
    }

    if HAS_XGB:
        models["XGB"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", XGBClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight * 2,
                eval_metric="aucpr",
                verbosity=0, random_state=42
            ))
        ])

    return models


def evaluate_models(X, y, models, label=""):
    """Cross-validation ile değerlendirme"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    if label:
        print(f"\n  [{label}]")
    print(f"  {'Model':<8} {'ROC-AUC':>9} {'PR-AUC':>9} "
          f"{'F1':>7} {'Prec':>7} {'Recall':>8}")
    print(f"  {'─'*55}")

    results = {}
    for name, pipe in models.items():
        try:
            res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
            r = {
                "auc": res["test_roc_auc"].mean(),
                "ap":  res["test_average_precision"].mean(),
                "f1":  res["test_f1"].mean(),
                "prec": res["test_precision"].mean(),
                "rec":  res["test_recall"].mean(),
                "auc_std": res["test_roc_auc"].std(),
            }
            results[name] = r
            print(f"  {name:<8} "
                  f"{r['auc']:>9.4f} "
                  f"{r['ap']:>9.4f} "
                  f"{r['f1']:>7.4f} "
                  f"{r['prec']:>7.4f} "
                  f"{r['rec']:>8.4f}  "
                  f"±{r['auc_std']:.4f}")
        except Exception as e:
            print(f"  {name:<8} HATA: {e}")
            results[name] = None

    best = max(
        [(v["auc"], k) for k, v in results.items() if v],
        default=(0, "")
    )
    return results, best[0], best[1]


# ═══════════════════════════════════════════════════════════
# 4. TİP BAZLI MODELLER
# ═══════════════════════════════════════════════════════════

def classify_pattern(row):
    qr = row.get("quiescence_ratio", np.nan)
    acc = row.get("accel_90d", np.nan)
    n_w3 = row.get("w3_n_events", 0) or 0
    if n_w3 < 3 or pd.isna(qr):
        return "TIP_C"
    if qr < 0.5:
        return "TIP_B"
    if qr >= 1.0:
        return "TIP_A"
    if 0.5 <= qr < 0.8:
        return "TIP_A" if (pd.notna(acc) and acc >= 1.5) else "TIP_B"
    return "TIP_A"


def train_pattern_models(real_norm, ctrl_clean):
    """Tip A / Tip B için ayrı modeller"""
    print(f"\n{'─'*60}")
    print("TİP BAZLI MODELLER")
    print(f"{'─'*60}")

    real_norm = real_norm.copy()
    real_norm["pattern_type"] = real_norm.apply(classify_pattern, axis=1)

    pt_dist = real_norm["pattern_type"].value_counts()
    for pt, cnt in pt_dist.items():
        print(f"  {pt}: {cnt} ({cnt/len(real_norm)*100:.1f}%)")

    for pt in ["TIP_A", "TIP_B"]:
        real_pt = real_norm[real_norm["pattern_type"] == pt].copy()
        if len(real_pt) < 20:
            print(f"\n  {pt}: Yetersiz veri ({len(real_pt)})")
            continue

        print(f"\n  {pt}: {len(real_pt)} real, {len(ctrl_clean)} ctrl")
        X_pt, y_pt, _, _ = prepare_features(real_pt, ctrl_clean)

        pw = (y_pt == 0).sum() / max(y_pt.sum(), 1)
        models_pt = build_models(pos_weight=pw)

        cv = StratifiedKFold(
            n_splits=min(5, int(y_pt.sum())),
            shuffle=True, random_state=42
        )

        print(f"  {'Model':<8} {'AUC':>8} {'AP':>8} {'Recall':>8}")
        print(f"  {'─'*35}")
        for name, pipe in models_pt.items():
            try:
                auc = cross_val_score(
                    pipe, X_pt, y_pt, cv=cv, scoring="roc_auc"
                ).mean()
                ap = cross_val_score(
                    pipe, X_pt, y_pt, cv=cv,
                    scoring="average_precision"
                ).mean()
                rec = cross_val_score(
                    pipe, X_pt, y_pt, cv=cv, scoring="recall"
                ).mean()
                print(f"  {name:<8} {auc:>8.4f} {ap:>8.4f} {rec:>8.4f}")
            except Exception as e:
                print(f"  {name:<8} HATA: {e}")


# ═══════════════════════════════════════════════════════════
# 5. ANA FONKSİYON
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SeismoPattern - FAZ 7: KONTROL TEMİZLEME + GELİŞMİŞ MODEL")
    print("=" * 70)
    print(f"XGBoost: {'✅' if HAS_XGB else '❌'}")

    # ── Veri yükle ───────────────────────────────────────
    real_norm = pd.read_csv("output/real_normalized.csv")
    ctrl_norm = pd.read_csv("output/ctrl_normalized.csv")
    all_eq = pd.read_csv("output/all_earthquakes.csv", low_memory=False)

    # all_eq hazırla
    all_eq["datetime_utc"] = pd.to_datetime(
        all_eq["datetime_utc"], errors="coerce"
    )
    all_eq["eff_lat"] = all_eq["centroid_lat"].fillna(all_eq["hypo_lat"])
    all_eq["eff_lon"] = all_eq["centroid_lon"].fillna(all_eq["hypo_lon"])
    all_eq["eff_depth_km"] = (all_eq["centroid_depth_km"]
                               .fillna(all_eq["hypo_depth_km"]))

    all_majors = all_eq[all_eq["mw"] >= 7.0].dropna(
        subset=["datetime_utc", "eff_lat", "eff_lon"]
    ).copy()

    print(f"\nYüklendi:")
    print(f"  real_norm: {len(real_norm)} satır")
    print(f"  ctrl_norm: {len(ctrl_norm)} satır")
    print(f"  Mw7+ katalog: {len(all_majors)} deprem")

    # Kontrol sütunlarını göster (debug için)
    print(f"\n  ctrl_norm sütunları (ilk 15): "
          f"{list(ctrl_norm.columns[:15])}")

    # ── 1. Kontrol temizleme ──────────────────────────────
    print("\n" + "─" * 50)
    print("1. KONTROL PENCERESİ TEMİZLEME")
    print("─" * 50)

    ctrl_clean, ctrl_dirty = clean_control_windows(
        ctrl_norm, all_majors,
        contamination_radius_km=300,
        contamination_years=3
    )

    ctrl_clean.to_csv("output/ctrl_clean.csv",
                       index=False, encoding="utf-8-sig")
    ctrl_dirty.to_csv("output/ctrl_dirty.csv",
                       index=False, encoding="utf-8-sig")
    print(f"  Kaydedildi: ctrl_clean.csv, ctrl_dirty.csv")

    # ── 2. Orijinal (kirli) model baseline ───────────────
    print("\n" + "─" * 50)
    print("2. ORİJİNAL (KİRLİ) KONTROL - BASELINE")
    print("─" * 50)

    pos_w_dirty = (
        len(ctrl_norm) / max(len(real_norm), 1)
    )
    models_dirty = build_models(pos_weight=pos_w_dirty)
    X_dirty, y_dirty, _, feats = prepare_features(
        real_norm, ctrl_norm
    )
    print(f"  Veri: {len(X_dirty)} ({int(y_dirty.sum())} real, "
          f"{int((y_dirty==0).sum())} ctrl)")

    res_dirty, best_dirty_auc, best_dirty_name = evaluate_models(
        X_dirty, y_dirty, models_dirty, label="KİRLİ"
    )

    # ── 3. Temiz kontrol ile model ────────────────────────
    print("\n" + "─" * 50)
    print("3. TEMİZ KONTROL İLE MODEL")
    print("─" * 50)

    if len(ctrl_clean) < 20:
        print("  ⚠️  Temiz kontrol sayısı çok az, analiz atlanıyor.")
        best_clean_auc = best_dirty_auc
        best_clean_name = best_dirty_name
    else:
        pos_w_clean = len(ctrl_clean) / max(len(real_norm), 1)
        models_clean = build_models(pos_weight=pos_w_clean)
        X_clean, y_clean, fault_clean, _ = prepare_features(
            real_norm, ctrl_clean
        )
        print(f"  Veri: {len(X_clean)} ({int(y_clean.sum())} real, "
              f"{int((y_clean==0).sum())} ctrl)")

        res_clean, best_clean_auc, best_clean_name = evaluate_models(
            X_clean, y_clean, models_clean, label="TEMİZ"
        )

        # ── 4. Fay tipine göre ────────────────────────────
        print("\n" + "─" * 50)
        print("4. FAY TİPİNE GÖRE (TEMİZ KONTROL)")
        print("─" * 50)

        ft_col_r = detect_columns(real_norm,
                                   ["fault_type", "main_fault_type"])
        ft_col_c = detect_columns(ctrl_clean,
                                   ["fault_type", "parent_fault_type"])

        for ft in ["REVERSE", "STRIKE_SLIP", "NORMAL"]:
            try:
                real_ft = real_norm[
                    real_norm[ft_col_r] == ft
                ] if ft_col_r else pd.DataFrame()
                ctrl_ft = ctrl_clean[
                    ctrl_clean[ft_col_c] == ft
                ] if ft_col_c else pd.DataFrame()

                if len(real_ft) < 15 or len(ctrl_ft) < 15:
                    print(f"  {ft}: Yetersiz "
                          f"(real={len(real_ft)}, ctrl={len(ctrl_ft)})")
                    continue

                X_ft, y_ft, _, _ = prepare_features(real_ft, ctrl_ft)
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scl", RobustScaler()),
                    ("mdl", GradientBoostingClassifier(
                        n_estimators=200, max_depth=3,
                        learning_rate=0.05, random_state=42
                    ))
                ])
                cv_ft = StratifiedKFold(
                    n_splits=min(5, int(y_ft.sum())),
                    shuffle=True, random_state=42
                )
                auc = cross_val_score(
                    pipe, X_ft, y_ft, cv=cv_ft, scoring="roc_auc"
                )
                ap = cross_val_score(
                    pipe, X_ft, y_ft, cv=cv_ft,
                    scoring="average_precision"
                )
                rec = cross_val_score(
                    pipe, X_ft, y_ft, cv=cv_ft, scoring="recall"
                )
                print(f"  {ft:<15}: "
                      f"AUC={auc.mean():.4f}±{auc.std():.4f}  "
                      f"AP={ap.mean():.4f}  "
                      f"Recall={rec.mean():.4f}")
            except Exception as e:
                print(f"  {ft}: {e}")

    # ── 5. Tip bazlı modeller ─────────────────────────────
    print("\n" + "─" * 50)
    print("5. TİP BAZLI MODELLER")
    print("─" * 50)

    train_pattern_models(real_norm, ctrl_clean)

    # ── 6. Özet ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FAZ 7 ÖZET")
    print(f"{'='*70}")

    print(f"\n  Kontrol temizleme:")
    print(f"    Orijinal : {len(ctrl_norm)}")
    print(f"    Kirli    : {len(ctrl_dirty)} "
          f"({len(ctrl_dirty)/len(ctrl_norm)*100:.1f}%)")
    print(f"    Temiz    : {len(ctrl_clean)} "
          f"({len(ctrl_clean)/len(ctrl_norm)*100:.1f}%)")

    print(f"\n  AUC karşılaştırması:")
    print(f"    Faz 6 (kirli ctrl) : 0.7016")
    print(f"    Kirli baseline     : {best_dirty_auc:.4f} "
          f"({best_dirty_name})")
    print(f"    Temiz kontrol      : {best_clean_auc:.4f} "
          f"({best_clean_name})")

    delta = best_clean_auc - 0.7016
    arrow = "↑" if delta > 0 else "↓"
    print(f"    Değişim (Faz6→7)  : {arrow} {abs(delta):.4f}")

    print(f"\n  Tüm fazlar:")
    history = [
        ("Faz 3 (GCMT basic)",         0.7217),
        ("Faz 4 (Feature eng.)",        0.7217),
        ("Faz 5 (Regional norm.)",      0.7131),
        ("Faz 6 (Z-score+hibrit)",      0.7016),
        ("Faz 7 (Temiz kontrol)",       best_clean_auc),
    ]
    for label, auc in history:
        bar = "█" * int(auc * 50)
        print(f"    {label:<30}: {auc:.4f} {bar}")

    if best_clean_auc >= 0.75:
        print(f"\n  ✅ 0.75 hedefi aşıldı! Uygulama katmanına hazır.")
    elif best_clean_auc >= 0.72:
        print(f"\n  ⚠️  0.72+ seviyesinde. ISC ile devam önerilir.")
    else:
        print(f"\n  → Yapısal veri sorunu. ISC/mikro-sismik katman şart.")


if __name__ == "__main__":
    main()