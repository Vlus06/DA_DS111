import numpy as np
import pandas as pd

from utils_firerisk import (
    REGION, build_panel, calendar_features, canon_key, cluster_events,
    add_lag_features, daily_from_hourly, days_since_rain, local_dt, magnus_es, nesterov_series,
    norm_name, parse_oni, sample_tile, season_label, summarize_events, tile_name,
)


def test_nesterov():
    P = nesterov_series([30] * 7, [20] * 7, [0] * 5 + [8, 0], 7, "linear")
    assert abs(P[4] - 3000) < 1e-9 and P[5] == 0 and abs(P[6] - 600) < 1e-9
    assert abs(nesterov_series([30], [20], [3.5], 7, "linear")[0] - 300) < 1e-9
    assert abs(nesterov_series([30], [20], [3.5], 7, "binary")[0] - 600) < 1e-9
    P = nesterov_series([30] * 4, [20] * 4, [0, np.nan, 0, 9], 7)
    assert np.isnan(P[1]) and np.isnan(P[2]) and P[3] == 0


def test_seasons():
    months = [11, 12, 1, 2, 3, 4]
    assert season_label(pd.Timestamp("2024-12-15"), months) == 2024
    assert season_label(pd.Timestamp("2025-02-01"), months) == 2024
    assert np.isnan(season_label(pd.Timestamp("2025-07-01"), months))


def test_local_time():
    h = pd.DataFrame({"acq_date": ["2024-03-01"], "acq_time": [1830]})
    assert local_dt(h).iloc[0] == pd.Timestamp("2024-03-02 01:30")


def test_events():
    pts = pd.DataFrame({
        "latitude": [16.8, 16.801, 16.802, 17.5, 16.8],
        "longitude": [107.1] * 3 + [107.5, 107.1],
        "ldate": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-01", "2024-03-20"]),
        "frp": [1.0] * 5,
    })
    events = summarize_events(cluster_events(pts))
    assert len(events) == 3 and sorted(events["n_hot"].tolist()) == [1, 1, 3]

    days = pd.DataFrame({"x": 1}, index=pd.date_range("2024-03-01", periods=5, name="date"))
    hot = pts.iloc[:3].copy()
    hot["event_id"] = 0
    panel = build_panel(days, hot, hot, summarize_events(hot), big_pixels=5, big_days=2)
    assert panel["fire_any_forest"].tolist() == [1, 1, 1, 0, 0]
    assert panel["large_event_active"].tolist() == [1, 1, 1, 0, 0]


def test_names():
    assert norm_name("Tỉnh Quảng Trị") == "quang tri"
    assert norm_name("Đắk Lắk") == "dak lak"
    assert canon_key("Hồ Chí Minh City") == "ho chi minh"
    assert canon_key("Bà Rịa – Vũng Tàu") == "ba ria vung tau"
    assert REGION["quang tri"] == "BTB" and len(set(REGION.values())) == 8


def test_daily_from_hourly():
    idx = pd.date_range("2024-03-01", periods=48, freq="h")
    hourly = pd.DataFrame({
        "time": idx, "temperature_2m": 30.0, "dew_point_2m": 20.0,
        "relative_humidity_2m": 50.0, "wind_speed_10m": 5.0, "precipitation": 0.1,
    })
    d = daily_from_hourly(hourly)
    assert len(d) == 2 and abs(d["rain_cal"].iloc[0] - 2.4) < 1e-9
    assert abs(d["D13_hpa"].iloc[0] - (magnus_es(30) - magnus_es(20))) < 1e-9
    assert d["dd13_c"].iloc[0] == 10


def test_sample_tile():
    class Transform:
        a, c, e, f = 0.001, 105.0, -0.001, 18.0

    class Source:
        transform = Transform
        width = 3000
        height = 3000

        def read(self, band, window):
            (r0, r1), (c0, c1) = window
            rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
            return ((rr + cc) % 250).astype("uint8")

    xs = np.array([105.0005, 105.5005, 107.9995])
    ys = np.array([17.9995, 17.4995, 15.0005])
    assert sample_tile(Source(), xs, ys, block=1024).tolist() == [0, 1000 % 250, 5998 % 250]
    assert tile_name(16.8, 107.1) == "N15E105" and tile_name(10.9, 106.8) == "N09E105"


def test_lag_features():
    assert days_since_rain([0, 0, 6, 0, 0, 0, 7, 0], 5)[[2, 5, 6, 7]].tolist() == [0, 3, 0, 1]
    idx = pd.date_range("2011-01-01", "2025-12-31", name="date")
    rng = np.random.default_rng(0)
    n = len(idx)
    d = pd.DataFrame({
        "T13": 28 + 5 * rng.random(n),
        "D13_hpa": 15 + 10 * rng.random(n),
        "rain_cal": (rng.random(n) < 0.2) * rng.exponential(5, n),
    }, index=idx)
    f = add_lag_features(d)
    assert abs(f["rain_7d"].iloc[10] - d["rain_cal"].iloc[4:11].sum()) < 1e-9
    assert f["wx_missing"].sum() == 0
    assert abs(f.loc[:"2023-12-31", "T13_anom"].mean()) < 0.5


def test_calendar():
    c = calendar_features(pd.DatetimeIndex(["2024-02-10", "2024-02-08", "2024-03-01", "2024-04-18", "2024-05-01"]))
    assert c["days_from_tet"].tolist()[:2] == [0, -2]
    assert c["tet_win3"].tolist()[:3] == [1, 1, 0]
    assert c["giotuo_win1"].iloc[3] == 1 and c["holiday_solar"].iloc[4] == 1


def test_oni():
    oni = parse_oni("SEAS YR TOTAL ANOM\n DJF 2012 26.0 -0.9\n JFM 2012 26.3 -0.7\n NDJ 2012 27.0 -0.3\n")
    assert oni[pd.Timestamp("2012-01-01")] == -0.9
    assert oni[pd.Timestamp("2012-12-01")] == -0.3
