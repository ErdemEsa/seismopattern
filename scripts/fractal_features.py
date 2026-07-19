import numpy as np

LAT_CANDIDATES = (
    "eff_lat",
    "lat",
    "latitude",
    "hypo_lat",
    "centroid_lat",
    "main_lat",
)

LON_CANDIDATES = (
    "eff_lon",
    "lon",
    "longitude",
    "hypo_lon",
    "centroid_lon",
    "main_lon",
)

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def fractal_dimension_boxcount(lat, lon, min_points=8, box_sizes=(2, 4, 8, 16)):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    mask = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[mask]
    lon = lon[mask]

    if len(lat) < min_points:
        return np.nan

    # Boylamı yaklaşık enlemle düzelt
    lon = lon * np.cos(np.radians(np.nanmean(lat)))

    x = lon.copy()
    y = lat.copy()

    xr = np.nanmax(x) - np.nanmin(x)
    yr = np.nanmax(y) - np.nanmin(y)

    if xr < 1e-12 and yr < 1e-12:
        return 0.0

    if xr < 1e-12:
        x = np.zeros_like(x)
    else:
        x = (x - np.nanmin(x)) / xr

    if yr < 1e-12:
        y = np.zeros_like(y)
    else:
        y = (y - np.nanmin(y)) / yr

    log_n = []
    log_occ = []

    for n in box_sizes:
        ix = np.clip(np.floor(x * n).astype(int), 0, n - 1)
        iy = np.clip(np.floor(y * n).astype(int), 0, n - 1)
        occ = len(set(zip(ix.tolist(), iy.tolist())))
        if occ > 0:
            log_n.append(np.log(float(n)))
            log_occ.append(np.log(float(occ)))

    if len(log_n) < 3:
        return np.nan

    slope = np.polyfit(log_n, log_occ, 1)[0]
    if not np.isfinite(slope):
        return np.nan

    return float(np.clip(slope, 0.0, 2.0))

def fractal_dimension_boxcount_df(df, min_points=8, box_sizes=(2, 4, 8, 16)):
    if df is None or len(df) == 0:
        return np.nan

    lat_col = _pick_col(df, LAT_CANDIDATES)
    lon_col = _pick_col(df, LON_CANDIDATES)

    if lat_col is None or lon_col is None:
        return np.nan

    return fractal_dimension_boxcount(
        df[lat_col].values,
        df[lon_col].values,
        min_points=min_points,
        box_sizes=box_sizes,
    )
