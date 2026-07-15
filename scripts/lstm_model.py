#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeismoPattern - Seviye 5.1: LSTM Zaman Serisi Modeli
=====================================================
36 aylik deprem zaman serisinden risk tahmini.

Mimari:
  Input: (batch, 36, n_channels)
  LSTM(64) -> Dropout(0.3) -> LSTM(32) -> Dense(16) -> Dense(1) -> Sigmoid

Kanallar:
  0: Aylik olay sayisi
  1: Aylik max Mw
  2: Aylik ortalama Mw
  3: Aylik b-degeri (kayan pencere)
  4: Aylik ortalama derinlik
  5: Aylik ortalama mesafe

Kullanim:
  python scripts/lstm_model.py --prepare
  python scripts/lstm_model.py --train
  python scripts/lstm_model.py --evaluate
"""

import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch yuklu degil: pip install torch")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

OUTPUT_DIR = Path("output/lstm")
OUTPUT_DIR.mkdir(exist_ok=True)

N_MONTHS = 36
N_CHANNELS = 6
EARTH_R = 6371.0


# =========================================================
# VERI HAZIRLAMA
# =========================================================

def haversine_km(lat1, lon1, lat2, lon2):
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return EARTH_R * 2 * math.asin(math.sqrt(a))


def b_value_simple(mags, dm=0.1):
    mags = np.array(mags)
    mags = mags[np.isfinite(mags)]
    if len(mags) < 10:
        return np.nan
    mmin = np.min(mags)
    mean_m = np.mean(mags)
    denom = mean_m - (mmin - dm / 2.0)
    if denom <= 0:
        return np.nan
    return np.log10(np.e) / denom


def compute_monthly_series(catalog_df, lat, lon, main_time,
                            radius_km=200, n_months=36):
    """
    Bir ana deprem icin 36 aylik zaman serisi hesapla.
    Her ay icin 6 kanal.
    """
    df = catalog_df.copy()

    time_col = next((c for c in ["datetime_utc", "time"]
                     if c in df.columns), None)
    lat_col = next((c for c in ["eff_lat", "centroid_lat", "hypo_lat", "lat"]
                    if c in df.columns), None)
    lon_col = next((c for c in ["eff_lon", "centroid_lon", "hypo_lon", "lon"]
                    if c in df.columns), None)
    mag_col = next((c for c in ["mw", "magnitude", "mag"]
                    if c in df.columns), None)
    dep_col = next((c for c in ["eff_depth_km", "centroid_depth_km",
                                 "hypo_depth_km", "depth_km"]
                    if c in df.columns), None)

    if not all([time_col, lat_col, lon_col, mag_col]):
        return None

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, lat_col, lon_col, mag_col])

    # Mesafe filtresi
    dists = np.array([
        haversine_km(lat, lon, float(r[lat_col]), float(r[lon_col]))
        for _, r in df.iterrows()
    ])
    df = df[dists <= radius_km].copy()

    # Zaman filtresi: main_time'dan onceki n_months ay
    end_time = main_time
    start_time = end_time - timedelta(days=n_months * 30.44)

    df = df[(df[time_col] >= start_time) & (df[time_col] < end_time)].copy()

    if len(df) < 5:
        return None

    # Mesafeleri yeniden hesapla (filtrelenmis set icin)
    df["_dist"] = [
        haversine_km(lat, lon, float(r[lat_col]), float(r[lon_col]))
        for _, r in df.iterrows()
    ]

    # Aylik pencereler
    series = np.zeros((n_months, N_CHANNELS))

    for m in range(n_months):
        month_end = end_time - timedelta(days=m * 30.44)
        month_start = end_time - timedelta(days=(m + 1) * 30.44)

        mask = (df[time_col] >= month_start) & (df[time_col] < month_end)
        month_df = df[mask]

        n_events = len(month_df)
        series[n_months - 1 - m, 0] = n_events

        if n_events > 0:
            mags = month_df[mag_col].values
            series[n_months - 1 - m, 1] = np.max(mags)
            series[n_months - 1 - m, 2] = np.mean(mags)
            series[n_months - 1 - m, 3] = b_value_simple(mags)

            if dep_col and dep_col in month_df.columns:
                depths = month_df[dep_col].dropna().values
                if len(depths) > 0:
                    series[n_months - 1 - m, 4] = np.mean(depths)

            series[n_months - 1 - m, 5] = np.mean(month_df["_dist"].values)

    # NaN'lari 0 ile doldur
    series = np.nan_to_num(series, nan=0.0)

    return series


def prepare_dataset(radius_km=200, n_months=36):
    """
    GCMT katalogunden LSTM egitim verisi hazirla.
    real: Mw7+ depremden onceki 36 ay
    control: kontrol pencerelerinden onceki 36 ay
    """
    print("=" * 60)
    print("LSTM VERI HAZIRLAMA")
    print("=" * 60)

    catalog_path = Path("output/all_earthquakes.csv")
    if not catalog_path.exists():
        print("HATA: all_earthquakes.csv bulunamadi")
        return

    df = pd.read_csv(catalog_path, low_memory=False)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")

    for col_pair in [("eff_lat", "centroid_lat", "hypo_lat"),
                     ("eff_lon", "centroid_lon", "hypo_lon"),
                     ("eff_depth_km", "centroid_depth_km", "hypo_depth_km")]:
        if col_pair[0] not in df.columns:
            df[col_pair[0]] = df.get(col_pair[1],
                                      df.get(col_pair[2], pd.Series()))

    df = df.dropna(subset=["datetime_utc", "mw", "eff_lat", "eff_lon"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    print(f"  Katalog: {len(df)} deprem")

    # Buyuk depremler (real)
    majors = df[df["mw"] >= 7.0].copy()
    print(f"  Mw7+ deprem: {len(majors)}")

    # Kontrol pencerelerini yukle
    ctrl_path = Path("output/gcmt_control_features.csv")
    ctrl_df = None
    if ctrl_path.exists():
        ctrl_raw = pd.read_csv(ctrl_path, low_memory=False)
        # Her kontrol penceresi icin konum ve zaman bilgisi
        time_col_c = next((c for c in ["ref_datetime_utc", "datetime_utc",
                                        "main_datetime_utc"]
                           if c in ctrl_raw.columns), None)
        lat_col_c = next((c for c in ["ref_lat", "lat", "main_lat"]
                          if c in ctrl_raw.columns), None)
        lon_col_c = next((c for c in ["ref_lon", "lon", "main_lon"]
                          if c in ctrl_raw.columns), None)

        if all([time_col_c, lat_col_c, lon_col_c]):
            ctrl_df = ctrl_raw[[time_col_c, lat_col_c, lon_col_c]].copy()
            ctrl_df.columns = ["ctrl_time", "ctrl_lat", "ctrl_lon"]
            ctrl_df["ctrl_time"] = pd.to_datetime(ctrl_df["ctrl_time"],
                                                   errors="coerce")
            ctrl_df = ctrl_df.dropna()
            # Duplikatlari kaldir
            ctrl_df = ctrl_df.drop_duplicates(
                subset=["ctrl_lat", "ctrl_lon", "ctrl_time"]
            ).reset_index(drop=True)
            print(f"  Kontrol penceresi: {len(ctrl_df)}")

    # Real seriler
    print(f"\n  Real seriler hesaplaniyor...")
    real_series = []
    real_count = 0

    for i, row in majors.iterrows():
        main_time = row["datetime_utc"]
        lat = float(row["eff_lat"])
        lon = float(row["eff_lon"])

        # En az 3 yillik veri olmali
        catalog_start = df["datetime_utc"].min()
        if (main_time - catalog_start).days < n_months * 30:
            continue

        series = compute_monthly_series(df, lat, lon, main_time,
                                         radius_km, n_months)
        if series is not None:
            real_series.append(series)
            real_count += 1

        if real_count % 50 == 0 and real_count > 0:
            print(f"    {real_count} real islendi...")

    print(f"  Toplam real seri: {len(real_series)}")

    # Control seriler
    ctrl_series = []
    if ctrl_df is not None:
        print(f"\n  Kontrol serileri hesaplaniyor...")
        ctrl_count = 0

        for _, row in ctrl_df.iterrows():
            ctrl_time = row["ctrl_time"]
            lat = float(row["ctrl_lat"])
            lon = float(row["ctrl_lon"])

            if pd.isna(ctrl_time):
                continue

            series = compute_monthly_series(df, lat, lon, ctrl_time,
                                             radius_km, n_months)
            if series is not None:
                ctrl_series.append(series)
                ctrl_count += 1

            if ctrl_count % 50 == 0 and ctrl_count > 0:
                print(f"    {ctrl_count} kontrol islendi...")

            if ctrl_count >= len(real_series) * 2:
                break

        print(f"  Toplam kontrol seri: {len(ctrl_series)}")

    if len(real_series) < 20 or len(ctrl_series) < 20:
        print("HATA: Yetersiz veri")
        return

    # Numpy dizilerine cevir
    X_real = np.array(real_series)
    X_ctrl = np.array(ctrl_series)

    y_real = np.ones(len(X_real))
    y_ctrl = np.zeros(len(X_ctrl))

    X = np.concatenate([X_real, X_ctrl], axis=0)
    y = np.concatenate([y_real, y_ctrl])

    # Normalizasyon (kanal bazli)
    means = X.mean(axis=(0, 1), keepdims=True)
    stds = X.std(axis=(0, 1), keepdims=True)
    stds[stds == 0] = 1
    X_norm = (X - means) / stds

    # Kaydet
    np.save(OUTPUT_DIR / "X_series.npy", X_norm)
    np.save(OUTPUT_DIR / "y_labels.npy", y)
    np.save(OUTPUT_DIR / "norm_means.npy", means)
    np.save(OUTPUT_DIR / "norm_stds.npy", stds)

    print(f"\n  Kaydedildi:")
    print(f"    X shape: {X_norm.shape}")
    print(f"    y shape: {y.shape}")
    print(f"    real: {int(y.sum())}, ctrl: {int((y == 0).sum())}")
    print(f"    Kanallar: [olay_sayisi, max_mw, mean_mw, b_value, depth, dist]")


# =========================================================
# LSTM MODEL
# =========================================================

class SeismoLSTM(nn.Module):
    def __init__(self, n_channels=N_CHANNELS, hidden_size=64,
                 n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Son zaman adiminin ciktisi
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# EGITIM
# =========================================================

def train_lstm(epochs=100, lr=0.001, batch_size=32):
    if not HAS_TORCH:
        print("PyTorch yuklu degil!")
        return

    print("=" * 60)
    print("LSTM MODEL EGITIMI")
    print("=" * 60)

    X = np.load(OUTPUT_DIR / "X_series.npy")
    y = np.load(OUTPUT_DIR / "y_labels.npy")

    print(f"  Veri: X={X.shape}, y={y.shape}")
    print(f"  real={int(y.sum())}, ctrl={int((y == 0).sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # 5-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    fold_aps = []
    best_model_state = None
    best_auc = 0

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n  Fold {fold + 1}/5")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Class weight
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = SeismoLSTM(n_channels=X.shape[2]).to(device)
        criterion = nn.BCELoss(
            weight=None
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        best_val_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                pred = model(X_batch)

                # Weighted loss
                weight = torch.where(y_batch == 1,
                                      torch.tensor(pos_weight).to(device),
                                      torch.tensor(1.0).to(device))
                loss = nn.functional.binary_cross_entropy(
                    pred, y_batch, weight=weight
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    pred = model(X_batch)
                    val_preds.extend(pred.cpu().numpy())
                    val_true.extend(y_batch.numpy())

            val_preds = np.array(val_preds)
            val_true = np.array(val_true)

            try:
                val_auc = roc_auc_score(val_true, val_preds)
                val_ap = average_precision_score(val_true, val_preds)
            except:
                val_auc = 0
                val_ap = 0

            scheduler.step(1 - val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f} "
                      f"val_AUC={val_auc:.4f} val_AP={val_ap:.4f}")

            if patience_counter >= 20:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        fold_aucs.append(best_val_auc)
        fold_aps.append(val_ap)
        print(f"  Fold {fold+1} AUC: {best_val_auc:.4f}")

    # Ozet
    print(f"\n{'='*60}")
    print("LSTM CV SONUCLARI")
    print(f"{'='*60}")
    print(f"  AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"  AP:  {np.mean(fold_aps):.4f} +/- {np.std(fold_aps):.4f}")
    print(f"  Fold AUCs: {[round(a, 4) for a in fold_aucs]}")

    # En iyi modeli kaydet
    if best_model_state:
        model = SeismoLSTM(n_channels=X.shape[2])
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), OUTPUT_DIR / "lstm_model.pth")
        print(f"\n  Model kaydedildi: {OUTPUT_DIR / 'lstm_model.pth'}")

    # Karsilastirma
    print(f"\n  Karsilastirma:")
    print(f"    XGBoost (Faz 9)  : AUC 0.9087")
    print(f"    LSTM (bu model)  : AUC {np.mean(fold_aucs):.4f}")

    delta = np.mean(fold_aucs) - 0.9087
    if delta > 0.01:
        print(f"    LSTM daha iyi (+{delta:.4f})")
    elif delta < -0.01:
        print(f"    XGBoost daha iyi ({delta:.4f})")
    else:
        print(f"    Benzer performans ({delta:+.4f})")

    # Sonuclari kaydet
    results = {
        "model": "LSTM",
        "n_channels": N_CHANNELS,
        "n_months": N_MONTHS,
        "cv_auc_mean": round(float(np.mean(fold_aucs)), 4),
        "cv_auc_std": round(float(np.std(fold_aucs)), 4),
        "cv_ap_mean": round(float(np.mean(fold_aps)), 4),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        "xgboost_auc": 0.9087,
        "improvement": round(float(delta), 4),
    }
    with open(OUTPUT_DIR / "lstm_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true",
                    help="Zaman serisi verisi hazirla")
    ap.add_argument("--train", action="store_true",
                    help="LSTM egit")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if args.prepare:
        prepare_dataset()
    elif args.train:
        if not HAS_TORCH:
            print("pip install torch")
            return
        train_lstm(epochs=args.epochs, lr=args.lr,
                   batch_size=args.batch)
    else:
        print("Kullanim:")
        print("  python scripts/lstm_model.py --prepare")
        print("  python scripts/lstm_model.py --train --epochs 100")


if __name__ == "__main__":
    main()