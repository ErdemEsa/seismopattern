#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Faz 5: Bölgesel Arka Plan Normalizasyonu
=========================================================
Her büyük deprem için:
1. Uzun dönem bölgesel arka plan sismisitesi hesapla
2. Öncü pencereleri bu arka plana göre normalize et
3. Bölgeden bağımsız "sapma" feature'ları oluştur
4. Gelişmiş kontrol penceresi kalite filtresi uygula
5. Yeniden model eğit

Yöntem:
  - Her ana deprem konumu için katalogdaki TÜM veriyi kullan
  - Analiz penceresinin DIŞINDA kalan dönemler = arka plan
  - Arka plan istatistiklerini hesapla
  - Öncü pencereleri arka plandan Z-score olarak ifade et
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians,
                                   [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (np.sin(dlat/2)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def b_value_mle(mags, mc=None, dm=0.05):
    mags = np.array(mags, dtype=float)
    mags = mags[np.isfinite(mags)]
    if mc is None:
        mc = np.min(mags) if len(mags) > 0 else 4.0
    mags = mags[mags >= mc]
    n = len(mags)
    if n < 15:
        return np.nan
    mean_m = np.mean(mags)
    denom = mean_m - (mc - dm / 2.0)
    if denom <= 0:
        return np.nan
    return np.log10(np.e) / denom


# ═══════════════════════════════════════════════════════════
# 1. BÖLGESEL ARKA PLAN HESAPLAMA
# ═══════════════════════════════════════════════════════════

def compute_regional_baseline(main_time, lat0, lon0, all_df,
                               radius_km=300,
                               exclude_years_before=3,
                               exclude_years_after=1,
                               min_baseline_years=3):
    """
    Bir ana deprem için bölgesel arka plan istatistikleri.

    Arka plan: Ana depremden exclude_years_before yıl öncesi
               ve exclude_years_after yıl sonrasının DIŞI.
               Yani: [katalog_başı, t-3yıl] ∪ [t+1yıl, katalog_sonu]

    Returns dict with baseline statistics.
    """
    exclude_start = main_time - pd.Timedelta(
        days=int(exclude_years_before * 365.25))
    exclude_end = main_time + pd.Timedelta(
        days=int(exclude_years_after * 365.25))

    # Bölgedeki tüm depremler
    all_df = all_df.copy()
    all_df["_dist"] = haversine_km(
        lat0, lon0,
        all_df["eff_lat"].values,
        all_df["eff_lon"].values
    )
    local = all_df[all_df["_dist"] <= radius_km].copy()

    # Arka plan dönemi (analiz penceresinin dışı)
    baseline = local[
        (local["datetime_utc"] < exclude_start) |
        (local["datetime_utc"] > exclude_end)
    ].copy()

    # Yeterli arka plan verisi var mı?
    if len(baseline) < 10:
        return None

    # Arka plan süresini hesapla
    cat_start = all_df["datetime_utc"].min()
    cat_end = all_df["datetime_utc"].max()

    # Arka plan dönemi uzunluğu (yıl)
    baseline_days = (
        max(0, (exclude_start - cat_start).total_seconds() / 86400) +
        max(0, (cat_end - exclude_end).total_seconds() / 86400)
    )
    baseline_years = baseline_days / 365.25

    if baseline_years < min_baseline_years:
        return None

    # Arka plan istatistikleri
    n_baseline = len(baseline)
    rate_per_year = n_baseline / baseline_years

    mags = baseline["mw"].dropna().values
    b_baseline = b_value_mle(mags) if len(mags) >= 15 else np.nan
    mean_mw_baseline = np.nanmean(mags) if len(mags) > 0 else np.nan
    std_mw_baseline = np.nanstd(mags) if len(mags) > 0 else np.nan
    max_mw_baseline = np.nanmax(mags) if len(mags) > 0 else np.nan

    depths = baseline["eff_depth_km"].dropna().values
    mean_depth_baseline = np.nanmean(depths) if len(depths) > 0 else np.nan
    std_depth_baseline = np.nanstd(depths) if len(depths) > 0 else np.nan

    dists = baseline["_dist"].values
    mean_dist_baseline = np.nanmean(dists) if len(dists) > 0 else np.nan
    std_dist_baseline = np.nanstd(dists) if len(dists) > 0 else np.nan

    # Aylık rate istatistikleri (dağılım için)
    monthly_rates = []
    for m in range(int(baseline_years * 12)):
        s = cat_start + pd.Timedelta(days=m*30)
        e = s + pd.Timedelta(days=30)
        if (exclude_start <= s <= exclude_end or
                exclude_start <= e <= exclude_end):
            continue
        n = ((baseline["datetime_utc"] >= s) &
             (baseline["datetime_utc"] < e)).sum()
        monthly_rates.append(n)

    rate_mean = np.mean(monthly_rates) if monthly_rates else 0
    rate_std = np.std(monthly_rates) if monthly_rates else 1

    return {
        "baseline_years": baseline_years,
        "n_baseline": n_baseline,
        "rate_per_year": rate_per_year,
        "rate_monthly_mean": rate_mean,
        "rate_monthly_std": max(rate_std, 0.1),
        "b_baseline": b_baseline,
        "mean_mw_baseline": mean_mw_baseline,
        "std_mw_baseline": max(std_mw_baseline, 0.1),
        "max_mw_baseline": max_mw_baseline,
        "mean_depth_baseline": mean_depth_baseline,
        "std_depth_baseline": max(std_depth_baseline, 0.1),
        "mean_dist_baseline": mean_dist_baseline,
        "std_dist_baseline": max(std_dist_baseline, 0.1),
    }


# ═══════════════════════════════════════════════════════════
# 2. Z-SCORE FEATURE HESAPLAMA
# ═══════════════════════════════════════════════════════════

def compute_zscore_features(row, baseline):
    """
    Öncü pencere değerlerini arka plan Z-score olarak ifade et.
    Z = (observed - baseline_mean) / baseline_std
    """
    if baseline is None:
        return {}

    z = {}
    eps = 1e-6

    # Aktivite oranı Z-score (1 yıllık)
    if "count_0_1y" in row and baseline["rate_per_year"] > 0:
        rate_1y = row.get("count_0_1y", 0)
        bl_rate = baseline["rate_per_year"]
        bl_std = max(baseline["rate_monthly_std"] * 12, eps)
        z["z_rate_1y"] = (rate_1y - bl_rate) / bl_std

    # Aktivite oranı Z-score (3 yıllık)
    if "w3_n_events" in row and baseline["rate_per_year"] > 0:
        rate_3y = row.get("w3_n_events", 0) / 3.0
        bl_rate = baseline["rate_per_year"]
        bl_std = max(baseline["rate_monthly_std"] * 12, eps)
        z["z_rate_3y"] = (rate_3y - bl_rate) / bl_std

    # b-değeri sapması
    if "w1_b_value" in row and pd.notna(baseline["b_baseline"]):
        b_val = row.get("w1_b_value", np.nan)
        if pd.notna(b_val):
            z["z_b_value_1y"] = (
                (baseline["b_baseline"] - b_val) /
                max(0.15, abs(baseline["b_baseline"]) * 0.1)
            )
            # Pozitif = b düştü = gerilim arttı

    if "w3_b_value" in row and pd.notna(baseline["b_baseline"]):
        b_val = row.get("w3_b_value", np.nan)
        if pd.notna(b_val):
            z["z_b_value_3y"] = (
                (baseline["b_baseline"] - b_val) /
                max(0.15, abs(baseline["b_baseline"]) * 0.1)
            )

    # Maksimum büyüklük sapması
    if "w1_max_mw" in row and pd.notna(baseline["max_mw_baseline"]):
        max_mw = row.get("w1_max_mw", np.nan)
        if pd.notna(max_mw):
            z["z_max_mw_1y"] = (
                (max_mw - baseline["mean_mw_baseline"]) /
                max(baseline["std_mw_baseline"], eps)
            )

    # Derinlik sapması
    if "w1_mean_depth_km" in row and pd.notna(baseline["mean_depth_baseline"]):
        depth = row.get("w1_mean_depth_km", np.nan)
        if pd.notna(depth):
            z["z_depth_1y"] = (
                (baseline["mean_depth_baseline"] - depth) /
                max(baseline["std_depth_baseline"], eps)
            )
            # Pozitif = sığlaşma = arka plandan daha sığ

    # Uzaklık sapması (odaklanma)
    if "w1_mean_dist_km" in row and pd.notna(baseline["mean_dist_baseline"]):
        dist = row.get("w1_mean_dist_km", np.nan)
        if pd.notna(dist):
            z["z_dist_1y"] = (
                (baseline["mean_dist_baseline"] - dist) /
                max(baseline["std_dist_baseline"], eps)
            )
            # Pozitif = daha yakın = odaklanma

    return z


# ═══════════════════════════════════════════════════════════
# 3. ANA İŞLEM: FEATURE TABLOSU OLUŞTUR
# ═══════════════════════════════════════════════════════════

def build_normalized_features(precursor_df, all_eq_df,
                               radius_km=200,
                               is_real=True):
    """
    Tüm öncü/kontrol pencereleri için normalize feature'lar oluştur.
    """
    df = precursor_df[precursor_df["radius_km"] == radius_km].copy()

    # Ana deprem bilgilerini belirle
    if is_real:
        time_col = "main_datetime_utc"
        lat_col = "main_lat"
        lon_col = "main_lon"
        ft_col = "main_fault_type"
        mw_col = "main_mw"
    else:
        time_col = "ref_datetime_utc"
        lat_col = "ref_lat"
        lon_col = "ref_lon"
        ft_col = "parent_fault_type"
        mw_col = "parent_mw"

    # all_eq_df hazırla
    aeq = all_eq_df.copy()
    aeq["datetime_utc"] = pd.to_datetime(aeq["datetime_utc"], errors="coerce")
    aeq["eff_lat"] = aeq["centroid_lat"].fillna(aeq["hypo_lat"])
    aeq["eff_lon"] = aeq["centroid_lon"].fillna(aeq["hypo_lon"])
    aeq["eff_depth_km"] = aeq["centroid_depth_km"].fillna(aeq["hypo_depth_km"])
    aeq = aeq.dropna(subset=["datetime_utc", "mw", "eff_lat", "eff_lon"])

    rows = []
    n_total = len(df)

    for i, row in df.iterrows():
        main_time = pd.to_datetime(row[time_col])
        lat0 = row[lat_col]
        lon0 = row[lon_col]

        if pd.isna(main_time) or pd.isna(lat0) or pd.isna(lon0):
            continue

        # Arka plan hesapla
        baseline = compute_regional_baseline(
            main_time, lat0, lon0, aeq,
            radius_km=radius_km,
            exclude_years_before=3,
            exclude_years_after=1,
            min_baseline_years=3
        )

        # Z-score feature'lar
        z_features = compute_zscore_features(row, baseline)

        # Orijinal feature'ları koru
        orig_features = {
            "quiescence_ratio": row.get("quiescence_ratio"),
            "accel_90d": row.get("accel_90d"),
            "monthly_slope_36m": row.get("monthly_slope_36m"),
            "count_0_1y": row.get("count_0_1y"),
            "count_1_2y": row.get("count_1_2y"),
            "count_2_3y": row.get("count_2_3y"),
            "w1_n_events": row.get("w1_n_events"),
            "w1_mean_mw": row.get("w1_mean_mw"),
            "w1_std_mw": row.get("w1_std_mw"),
            "w1_b_value": row.get("w1_b_value"),
            "w3_b_value": row.get("w3_b_value"),
            "w1_mean_depth_km": row.get("w1_mean_depth_km"),
            "w1_std_depth_km": row.get("w1_std_depth_km"),
            "w1_mean_dist_km": row.get("w1_mean_dist_km"),
            "w1_std_dist_km": row.get("w1_std_dist_km"),
            "w3_n_events": row.get("w3_n_events"),
            "w3_mean_mw": row.get("w3_mean_mw"),
            "w3_mean_depth_km": row.get("w3_mean_depth_km"),
            "w3_mean_dist_km": row.get("w3_mean_dist_km"),
            "w1_migration_slope_km_day": row.get("w1_migration_slope_km_day"),
            "w3_migration_slope_km_day": row.get("w3_migration_slope_km_day"),
            
            "temporal_entropy_12m": row.get("temporal_entropy_12m"),
            "monthly_entropy_36m": row.get("monthly_entropy_36m"),
            "interevent_cv_12m": row.get("interevent_cv_12m"),
            "fractal_dim_12m": row.get("fractal_dim_12m"),
            "fractal_dim_36m": row.get("fractal_dim_36m"),
            "fractal_dim_change": row.get("fractal_dim_change"),
        }

        # Türetilmiş feature'lar
        c0 = row.get("count_0_1y", 0) or 0
        c1 = row.get("count_1_2y", 0) or 0
        c2 = row.get("count_2_3y", 0) or 0
        orig_features["count_linear_trend"] = c0 - c2
        avg_prev = (c1 + c2) / 2.0 + 1e-6
        orig_features["count_accel_ratio"] = c0 / avg_prev

        # b-değeri düşüşü
        b1 = row.get("w1_b_value")
        b3 = row.get("w3_b_value")
        if pd.notna(b1) and pd.notna(b3):
            orig_features["b_drop_w3_w1"] = b3 - b1
        else:
            orig_features["b_drop_w3_w1"] = np.nan

        # Mekânsal odaklanma
        d1 = row.get("w1_mean_dist_km")
        d3 = row.get("w3_mean_dist_km")
        if pd.notna(d1) and pd.notna(d3):
            orig_features["spatial_focus_change"] = d3 - d1
        else:
            orig_features["spatial_focus_change"] = np.nan

        # Derinlik değişimi
        dep1 = row.get("w1_mean_depth_km")
        dep3 = row.get("w3_mean_depth_km")
        if pd.notna(dep1) and pd.notna(dep3):
            orig_features["depth_change_km"] = dep1 - dep3
        else:
            orig_features["depth_change_km"] = np.nan

        # Arka plan bilgisi
        baseline_info = {}
        if baseline:
            baseline_info = {
                "baseline_rate_per_year": baseline["rate_per_year"],
                "baseline_b": baseline["b_baseline"],
                "baseline_years": baseline["baseline_years"],
                "has_baseline": True
            }
        else:
            baseline_info = {"has_baseline": False}

        # Meta bilgi
        meta = {
            "event_id": row.get("main_event_id" if is_real else "parent_event_id"),
            "datetime_utc": main_time,
            "fault_type": row.get(ft_col),
            "mw": row.get(mw_col),
            "lat": lat0,
            "lon": lon0,
        }

        combined_row = {}
        combined_row.update(meta)
        combined_row.update(orig_features)
        combined_row.update(z_features)
        combined_row.update(baseline_info)
        rows.append(combined_row)

        if (i + 1) % 100 == 0:
            pct = (i + 1) / n_total * 100
            print(f"  {i+1}/{n_total} ({pct:.0f}%) işlendi...")

    result_df = pd.DataFrame(rows)

    # Arka plan oranı
    n_with_baseline = result_df.get("has_baseline", pd.Series()).sum()
    print(f"  Arka plan hesaplanan: {n_with_baseline}/{len(result_df)}")

    return result_df


# ═══════════════════════════════════════════════════════════
# 4. MODEL EĞİTİMİ
# ═══════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Orijinal normalize
    "quiescence_ratio",
    "accel_90d",
    "monthly_slope_36m",
    "count_0_1y",
    "count_1_2y",
    "count_2_3y",
    "count_linear_trend",
    "count_accel_ratio",
    "w1_n_events",
    "w1_mean_mw",
    "w1_std_mw",
    "w1_b_value",
    "w3_b_value",
    "b_drop_w3_w1",
    "w1_mean_depth_km",
    "w1_std_depth_km",
    "depth_change_km",
    "w1_mean_dist_km",
    "w1_std_dist_km",
    "spatial_focus_change",
    "w3_n_events",
    "w3_mean_mw",
    "w3_mean_depth_km",
    "w3_mean_dist_km",
    "w1_migration_slope_km_day",
    "w3_migration_slope_km_day",
    # Z-score feature'lar (arka plana göre)
    "z_rate_1y",
    "z_rate_3y",
    "z_b_value_1y",
    "z_b_value_3y",
    "z_max_mw_1y",
    "z_depth_1y",
    "z_dist_1y",
    "temporal_entropy_12m",
    "monthly_entropy_36m",
    "interevent_cv_12m",
    "fractal_dim_12m",
    "fractal_dim_36m",
]


def train_and_evaluate(real_norm, ctrl_norm):
    """Normalize edilmiş verilerle model eğit"""
    real_norm = real_norm.copy()
    ctrl_norm = ctrl_norm.copy()
    real_norm["target"] = 1
    ctrl_norm["target"] = 0

    avail_r = [f for f in FEATURE_COLS if f in real_norm.columns]
    avail_c = [f for f in FEATURE_COLS if f in ctrl_norm.columns]
    common = list(set(avail_r) & set(avail_c))

    combined = pd.concat([
        real_norm[common + ["target", "fault_type"]],
        ctrl_norm[common + ["target", "fault_type"]],
    ], ignore_index=True)

    X = combined[common]
    y = combined["target"]
    fault = combined["fault_type"]

    print(f"\nVeri seti: {len(X)} örnek, {len(common)} feature")
    print(f"  {int(y.sum())} real, {int((y==0).sum())} control")
    print(f"  Z-score feature'lar: "
          f"{[f for f in common if f.startswith('z_')]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "average_precision": "average_precision",
    }

    models = {
        "LR": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", LogisticRegression(
                max_iter=2000, C=0.1, class_weight="balanced",
                random_state=42
            ))
        ]),
        "RF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=8,
                class_weight="balanced", random_state=42,
                max_features="sqrt"
            ))
        ]),
        "GB": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=15, subsample=0.8, random_state=42
            ))
        ]),
    }

    if HAS_XGB:
        from xgboost import XGBClassifier
        models["XGB"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", RobustScaler()),
            ("mdl", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(y==0).sum()/y.sum(),
                eval_metric="logloss", verbosity=0,
                random_state=42
            ))
        ])

    print(f"\n{'='*70}")
    print("MODEL SONUÇLARI (Z-score normalize feature'larla)")
    print(f"{'='*70}")
    print(f"  {'Model':<6} {'AUC-ROC':>10} {'F1':>8} "
          f"{'Prec':>8} {'Recall':>8} {'AvgPrec':>9}")
    print(f"  {'─'*55}")

    best_auc = 0
    best_name = ""
    all_results = {}

    for name, pipe in models.items():
        try:
            res = cross_validate(pipe, X, y, cv=cv,
                                  scoring=scoring)
            auc = res["test_roc_auc"].mean()
            f1 = res["test_f1"].mean()
            prec = res["test_precision"].mean()
            rec = res["test_recall"].mean()
            ap = res["test_average_precision"].mean()

            print(f"  {name:<6} "
                  f"{auc:>10.4f} "
                  f"{f1:>8.4f} "
                  f"{prec:>8.4f} "
                  f"{rec:>8.4f} "
                  f"{ap:>9.4f}")

            all_results[name] = {
                "auc": auc, "f1": f1,
                "prec": prec, "rec": rec, "ap": ap
            }
            if auc > best_auc:
                best_auc = auc
                best_name = name
        except Exception as e:
            print(f"  {name:<6} HATA: {e}")

    # Feature importance (en iyi model)
    print(f"\n{'─'*60}")
    print(f"FEATURE IMPORTANCE ({best_name}, AUC={best_auc:.4f})")
    print(f"{'─'*60}")
    best_pipe = models[best_name]
    best_pipe.fit(X, y)

    try:
        imp = best_pipe.named_steps["mdl"].feature_importances_
        imp_df = pd.DataFrame({
            "feature": common,
            "importance": imp
        }).sort_values("importance", ascending=False)

        for _, row in imp_df.head(20).iterrows():
            is_z = "★" if row["feature"].startswith("z_") else " "
            bar = "█" * int(row["importance"] * 150)
            print(f"  {is_z} {row['feature']:<40s} "
                  f"{row['importance']:.4f} {bar}")
    except Exception as e:
        print(f"  Feature importance hatası: {e}")

    # Fay tipine göre AUC
    print(f"\n{'─'*60}")
    print("FAY TİPİNE GÖRE AUC (en iyi model)")
    print(f"{'─'*60}")

    for ft in ["REVERSE", "STRIKE_SLIP", "NORMAL"]:
        mask = (fault == ft)
        X_ft = X[mask]
        y_ft = y[mask]
        if len(X_ft) < 30 or y_ft.sum() < 10:
            continue
        try:
            auc_ft = cross_val_score(
                models[best_name], X_ft, y_ft,
                cv=StratifiedKFold(n_splits=3, shuffle=True,
                                    random_state=42),
                scoring="roc_auc"
            )
            print(f"  {ft:<15}: AUC = {auc_ft.mean():.4f} "
                  f"± {auc_ft.std():.4f}")
        except Exception as e:
            print(f"  {ft:<15}: {e}")

    return all_results, best_name, best_auc


# ═══════════════════════════════════════════════════════════
# 5. ANA FONKSİYON
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SeismoPattern - FAZ 5: BÖLGESEL ARKA PLAN NORMALİZASYONU")
    print("=" * 70)

    # Veri yükle
    real_df = pd.read_csv("output/gcmt_precursor_features.csv")
    ctrl_df = pd.read_csv("output/gcmt_control_features.csv")
    all_eq = pd.read_csv("output/all_earthquakes.csv", low_memory=False)

    print(f"\nYüklendi:")
    print(f"  real_df  : {len(real_df):,} satır")
    print(f"  ctrl_df  : {len(ctrl_df):,} satır")
    print(f"  all_eq   : {len(all_eq):,} deprem")

    # Normalize feature oluştur
    print("\n" + "─" * 50)
    print("REAL - Arka Plan Normalizasyonu (200km)")
    print("─" * 50)
    real_norm = build_normalized_features(
        real_df, all_eq, radius_km=200, is_real=True
    )

    print("\n" + "─" * 50)
    print("CONTROL - Arka Plan Normalizasyonu (200km)")
    print("─" * 50)
    ctrl_norm = build_normalized_features(
        ctrl_df, all_eq, radius_km=200, is_real=False
    )

    # Kaydet
    real_norm.to_csv("output/real_normalized.csv",
                      index=False, encoding="utf-8-sig")
    ctrl_norm.to_csv("output/ctrl_normalized.csv",
                      index=False, encoding="utf-8-sig")
    print(f"\nNormalize veri kaydedildi.")
    print(f"  real_normalized.csv: {len(real_norm):,} satır")
    print(f"  ctrl_normalized.csv: {len(ctrl_norm):,} satır")

    # Z-score feature dağılımı
    print("\n" + "─" * 50)
    print("Z-SCORE FEATURE ÖZETİ")
    print("─" * 50)
    z_cols = [c for c in real_norm.columns if c.startswith("z_")]
    if z_cols:
        real_norm["label"] = "REAL"
        ctrl_norm["label"] = "CTRL"
        for zc in z_cols:
            r_vals = real_norm[zc].dropna()
            c_vals = ctrl_norm[zc].dropna()
            if len(r_vals) > 10 and len(c_vals) > 10:
                _, p = stats.mannwhitneyu(
                    r_vals, c_vals, alternative="two-sided"
                )
                sig = "***" if p < 0.001 else ("*" if p < 0.05 else " ")
                print(f"  {sig} {zc:<25s} "
                      f"real={r_vals.median():>8.3f}  "
                      f"ctrl={c_vals.median():>8.3f}  "
                      f"p={p:.4f}")

    # Model eğitimi
    print("\n" + "─" * 50)
    print("MODEL EĞİTİMİ")
    print("─" * 50)
    results, best_name, best_auc = train_and_evaluate(
        real_norm, ctrl_norm
    )

    # Önceki versiyonla karşılaştır
    print(f"\n{'='*70}")
    print("KARŞILAŞTIRMA: Önceki vs Normalize Edilmiş")
    print(f"{'='*70}")
    print(f"  Önceki en iyi AUC  : 0.7217 (Faz 4 - GB)")
    print(f"  Şimdiki en iyi AUC : {best_auc:.4f} ({best_name})")
    delta = best_auc - 0.7217
    arrow = "↑" if delta > 0 else "↓"
    print(f"  Değişim            : {arrow} {abs(delta):.4f}")

    if best_auc >= 0.75:
        print(f"\n  ✅ Hedef aşıldı (0.75+)")
    elif best_auc >= 0.72:
        print(f"\n  ⚠️  Marjinal iyileşme")
    else:
        print(f"\n  ❌ İyileşme yok → USGS mikro-sismik katmanı gerekiyor")


if __name__ == "__main__":
    main()