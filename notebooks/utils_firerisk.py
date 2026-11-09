import math
import os
import unicodedata

import numpy as np
import pandas as pd
import requests

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

REGION_PROVINCES = {
    "TB": "lai chau|dien bien|son la|hoa binh",
    "DB": "ha giang|cao bang|bac kan|tuyen quang|lang son|thai nguyen|phu tho|bac giang|quang ninh|yen bai|lao cai",
    "DBSH": "ha noi|hai phong|vinh phuc|bac ninh|hai duong|hung yen|thai binh|ha nam|nam dinh|ninh binh",
    "BTB": "thanh hoa|nghe an|ha tinh|quang binh|quang tri|thua thien hue",
    "NTB": "da nang|quang nam|quang ngai|binh dinh|phu yen|khanh hoa|ninh thuan|binh thuan",
    "TN": "kon tum|gia lai|dak lak|dak nong|lam dong",
    "DNB": "binh phuoc|tay ninh|binh duong|dong nai|ba ria vung tau|ho chi minh",
    "DBSCL": "long an|tien giang|ben tre|tra vinh|vinh long|dong thap|an giang|kien giang|can tho|hau giang|soc trang|bac lieu|ca mau",
}
REGION = {name: region for region, names in REGION_PROVINCES.items() for name in names.split("|")}
assert len(REGION) == 63

ALIAS = {
    "dac lac": "dak lak",
    "dac nong": "dak nong",
    "ho chi minh city": "ho chi minh",
    "tp ho chi minh": "ho chi minh",
    "hue": "thua thien hue",
    "ba ria vung tau province": "ba ria vung tau",
    "can tho city": "can tho",
}

SEASON = {
    "TB": [11, 12, 1, 2, 3, 4],
    "DB": [11, 12, 1, 2, 3, 4],
    "DBSH": [11, 12, 1, 2, 3, 4],
    "BTB": [4, 5, 6, 7, 8],
    "NTB": [1, 2, 3, 4, 5, 6, 7, 8],
    "TN": [12, 1, 2, 3, 4, 5],
    "DNB": [12, 1, 2, 3, 4, 5],
    "DBSCL": [12, 1, 2, 3, 4, 5],
}

TET = [
    "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28", "2018-02-16",
    "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10", "2025-01-29",
]
GIOTO = [
    "2012-03-31", "2013-04-19", "2014-04-09", "2015-04-28", "2016-04-16", "2017-04-06", "2018-04-25",
    "2019-04-14", "2020-04-02", "2021-04-21", "2022-04-10", "2023-04-29", "2024-04-18", "2025-04-07",
]
SOLAR_HOLIDAYS = [101, 430, 501, 902]


def norm_name(s):
    s = unicodedata.normalize("NFD", str(s)).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    for prefix in ["thanh pho ", "tinh ", "tp. ", "tp "]:
        s = s.replace(prefix, "")
    return " ".join(s.replace("-", " ").replace("–", " ").replace("—", " ").split())


def canon_key(name):
    key = norm_name(name)
    return ALIAS.get(key, key)


def local_dt(h):
    t = h["acq_time"].astype(int)
    utc = pd.to_datetime(h["acq_date"]) + pd.to_timedelta((t // 100) * 60 + (t % 100), unit="m")
    return utc + pd.Timedelta(hours=7)


def season_label(ts, months):
    if ts.month not in months:
        return np.nan
    wraps = months[0] > months[-1]
    return ts.year - 1 if (wraps and ts.month <= months[-1]) else ts.year


def magnus_es(t_c):
    t_c = np.asarray(t_c, dtype=float)
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def daily_from_hourly(hourly, hour=13):
    hourly = hourly.copy()
    hourly["time"] = pd.to_datetime(hourly["time"])
    sel = hourly[hourly["time"].dt.hour == hour].copy()
    sel.index = sel["time"].dt.normalize()

    d = pd.DataFrame({"T13": sel["temperature_2m"], "Td13": sel["dew_point_2m"]})
    if "relative_humidity_2m" in sel:
        d["RH13"] = sel["relative_humidity_2m"]
    else:
        d["RH13"] = 100 * magnus_es(d["Td13"]) / magnus_es(d["T13"])
    d["D13_hpa"] = magnus_es(d["T13"]) - magnus_es(d["Td13"])
    d["dd13_c"] = d["T13"] - d["Td13"]

    precip = hourly.set_index("time")["precipitation"]
    d["rain_cal"] = precip.groupby(precip.index.normalize()).sum(min_count=24)
    shifted_day = (precip.index + pd.Timedelta(hours=4)).normalize()
    d["rain_19_19"] = precip.groupby(shifted_day).sum(min_count=24)
    d.index.name = "date"
    return d


def nesterov_series(T13, D, rain, a0, mode="linear", rain_floor=0.1):
    P, broken, out = 0.0, False, []
    for t, d, r in zip(T13, D, rain):
        rain_known = not pd.isna(r)
        if rain_known and r >= a0:
            P, broken = 0.0, False
            out.append(0.0)
            continue
        if broken or pd.isna(t) or pd.isna(d) or not rain_known:
            broken = True
            out.append(np.nan)
            continue
        r = 0.0 if r < rain_floor else r
        k = 1.0 if (mode == "binary" or r == 0.0) else (a0 - r) / a0
        P += k * max(t, 0.0) * d
        out.append(P)
    return np.array(out)


def days_since_rain(rain, threshold):
    r = np.asarray(rain, dtype=float)
    n = len(r)
    last = pd.Series(np.where(r >= threshold, np.arange(n), np.nan)).ffill().values
    return np.arange(n) - last


def add_lag_features(d, clim_end="2023-12-31"):
    d = d.copy()
    rain = d["rain_cal"]
    for w in (3, 7, 14, 30):
        d[f"rain_{w}d"] = rain.rolling(w, min_periods=w).sum()
    for thr in (1, 5, 10):
        d[f"dsr_{thr}"] = days_since_rain(rain.values, thr)
    d["T13_roll7"] = d["T13"].rolling(7, min_periods=7).mean()
    d["D13_roll7"] = d["D13_hpa"].rolling(7, min_periods=7).mean()

    base = d.loc[:clim_end]
    clim = base.groupby(base.index.dayofyear)[["T13", "D13_hpa"]].mean().reindex(range(1, 367)).ffill().bfill()
    smooth = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]]).rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
    smooth.index = range(1, 367)

    doy = d.index.dayofyear
    d["T13_anom"] = d["T13"].values - smooth["T13"].reindex(doy).values
    d["D13_anom"] = d["D13_hpa"].values - smooth["D13_hpa"].reindex(doy).values
    return d


def calendar_features(idx):
    idx = pd.DatetimeIndex(idx)
    c = pd.DataFrame(index=idx)
    c["dow"] = idx.dayofweek
    c["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    c["month"] = idx.month
    c["doy"] = idx.dayofyear
    c["year"] = idx.year

    def signed_days_to_nearest(ref):
        pos = ref.searchsorted(idx)
        left = ref[np.clip(pos - 1, 0, len(ref) - 1)]
        right = ref[np.clip(pos, 0, len(ref) - 1)]
        d_left = (idx - left).days.values
        d_right = (idx - right).days.values
        return np.where(np.abs(d_left) <= np.abs(d_right), d_left, d_right)

    c["days_from_tet"] = signed_days_to_nearest(pd.DatetimeIndex(pd.to_datetime(TET)))
    c["tet_win3"] = (np.abs(c["days_from_tet"]) <= 3).astype(int)
    c["days_from_giotuo"] = signed_days_to_nearest(pd.DatetimeIndex(pd.to_datetime(GIOTO)))
    c["giotuo_win1"] = (np.abs(c["days_from_giotuo"]) <= 1).astype(int)
    c["thanh_minh_win1"] = ((idx.month == 4) & (idx.day >= 4) & (idx.day <= 6)).astype(int)
    c["holiday_solar"] = np.isin(idx.month * 100 + idx.day, SOLAR_HOLIDAYS).astype(int)
    return c


def parse_oni(text):
    center_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in center_month:
            try:
                rows[pd.Timestamp(int(parts[1]), center_month[parts[0]], 1)] = float(parts[-1])
            except ValueError:
                pass
    return pd.Series(rows).sort_index()


def load_oni(path):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["month"])
        return df.set_index("month")["oni"]
    oni = parse_oni(requests.get(ONI_URL, timeout=60).text)
    oni.rename("oni").rename_axis("month").to_csv(path)
    return oni


def cluster_events(h, eps_km=3.5, day_km=1.5):
    from sklearn.cluster import DBSCAN

    h = h.copy().reset_index(drop=True)
    if h.empty:
        h["event_id"] = pd.Series(dtype=int)
        return h
    x = h["longitude"].values * 111.32 * np.cos(np.radians(h["latitude"].values))
    y = h["latitude"].values * 110.57
    t = (pd.to_datetime(h["ldate"]) - pd.Timestamp("2000-01-01")).dt.days.values * day_km
    h["event_id"] = DBSCAN(eps=eps_km, min_samples=1).fit_predict(np.c_[x, y, t])
    return h


def summarize_events(h):
    if h.empty:
        return pd.DataFrame(columns=["event_id", "start", "end", "n_hot", "n_days", "sum_frp"])
    g = h.groupby("event_id").agg(
        start=("ldate", "min"),
        end=("ldate", "max"),
        n_hot=("ldate", "size"),
        sum_frp=("frp", "sum"),
        max_frp=("frp", "max"),
    ).reset_index()
    g["n_days"] = (pd.to_datetime(g["end"]) - pd.to_datetime(g["start"])).dt.days + 1
    return g


def build_panel(days, hot_all, hot_forest, events, big_pixels=5, big_days=2):
    p = days.copy()
    n_all = hot_all.groupby(pd.to_datetime(hot_all["ldate"])).size() if len(hot_all) else pd.Series(dtype=int)
    n_forest = hot_forest.groupby(pd.to_datetime(hot_forest["ldate"])).size() if len(hot_forest) else pd.Series(dtype=int)
    p["n_hot_all"] = n_all.reindex(p.index).fillna(0).astype(int)
    p["n_hot_forest"] = n_forest.reindex(p.index).fillna(0).astype(int)
    p["fire_any_forest"] = (p["n_hot_forest"] > 0).astype(int)
    p["event_start"] = 0
    p["large_event_active"] = 0
    for _, e in events.iterrows():
        start, end = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        if start in p.index:
            p.loc[start, "event_start"] += 1
        if e["n_hot"] >= big_pixels or e["n_days"] >= big_days:
            p.loc[(p.index >= start) & (p.index <= end), "large_event_active"] = 1
    return p


def tile_name(lat, lon):
    la, lo = int(math.floor(lat / 3) * 3), int(math.floor(lon / 3) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def sample_tile(src, xs, ys, block=2048):
    a, c, e, f = src.transform.a, src.transform.c, src.transform.e, src.transform.f
    cols = np.floor((np.asarray(xs) - c) / a).astype(int)
    rows = np.floor((np.asarray(ys) - f) / e).astype(int)
    out = np.zeros(len(cols), dtype="uint8")
    valid = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    block_r, block_c = rows // block, cols // block
    for br, bc in set(zip(block_r[valid].tolist(), block_c[valid].tolist())):
        mask = valid & (block_r == br) & (block_c == bc)
        r0, c0 = br * block, bc * block
        window = ((r0, min(r0 + block, src.height)), (c0, min(c0 + block, src.width)))
        arr = src.read(1, window=window)
        out[mask] = arr[rows[mask] - r0, cols[mask] - c0]
    return out


def random_points_in(poly, n, seed=0):
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = poly.bounds
    out = []
    try:
        from shapely import contains_xy

        while len(out) < n:
            xs = rng.uniform(minx, maxx, n * 2)
            ys = rng.uniform(miny, maxy, n * 2)
            mask = contains_xy(poly, xs, ys)
            out.extend(zip(xs[mask], ys[mask]))
    except ImportError:
        from shapely.geometry import Point

        while len(out) < n:
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                out.append((x, y))
    return np.array(out[:n])
