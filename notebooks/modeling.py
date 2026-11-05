import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FOREST_REGIONS = ["TB", "DB", "BTB", "NTB", "TN", "DNB"]
P_PRIMARY = "P_linear5_Ddew"

FEATURE_GROUPS = {
    "Nesterov P": ["logP"],
    "Rain history": ["rain_cal", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "dsr_1", "dsr_5", "dsr_10"],
    "Temp/humidity today": ["T13", "Td13", "RH13", "D13_hpa", "dd13_c"],
    "Weather anomaly/rolling": ["T13_roll7", "D13_roll7", "T13_anom", "D13_anom"],
    "Terrain/land/location": ["area_km2", "forest_share", "crop_share", "grass_share", "built_share", "rep_lat", "rep_lon"],
    "Day of year": ["doy_sin", "doy_cos", "month_in_season"],
}
M2_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]


def expit(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def climatology_logit(train, target, outcome, leave_season_out, alpha=20.0):
    cell = train.groupby(["prov", "month", "season"])[outcome].agg(k="sum", n="count").reset_index()
    total = cell.groupby(["prov", "month"])[["k", "n"]].sum().rename(columns={"k": "K", "n": "N"})
    region_cell = train.groupby(["month", "season"])[outcome].agg(rk="sum", rn="count").reset_index()
    region_total = region_cell.groupby("month")[["rk", "rn"]].sum().rename(columns={"rk": "RK", "rn": "RN"})

    t = target[["prov", "month", "season"]].join(total, on=["prov", "month"]).join(region_total, on="month")
    t = t.merge(cell, on=["prov", "month", "season"], how="left").merge(region_cell, on=["month", "season"], how="left")
    k, n = t["K"].fillna(0).values, t["N"].fillna(0).values
    rk, rn = t["RK"].fillna(0).values, t["RN"].fillna(0).values
    if leave_season_out:
        k, n = k - t["k"].fillna(0).values, n - t["n"].fillna(0).values
        rk, rn = rk - t["rk"].fillna(0).values, rn - t["rn"].fillna(0).values
    prior = np.where(rn > 0, rk / np.maximum(rn, 1), train[outcome].mean())
    return logit((k + alpha * prior) / (n + alpha))


def smoothed_climatology(group):
    clim = group.groupby("doy")[["T13", "D13_hpa"]].mean().reindex(range(1, 367)).ffill().bfill()
    padded = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]])
    smooth = padded.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
    smooth.index = range(1, 367)
    return smooth


def fold_anomalies(daily, held_out_season, target):
    train = daily[~((daily["in_season"] == 1) & (daily["season"] == held_out_season))]
    frames = []
    for prov, group in train.groupby("prov"):
        smooth = smoothed_climatology(group).rename(columns={"T13": "T13_clim", "D13_hpa": "D13_clim"})
        smooth["prov"] = prov
        frames.append(smooth.reset_index(names="doy"))
    clim = pd.concat(frames)
    merged = target[["prov", "doy", "T13", "D13_hpa"]].merge(clim, on=["prov", "doy"], how="left")
    return (merged["T13"] - merged["T13_clim"]).values, (merged["D13_hpa"] - merged["D13_clim"]).values


def fit_logistic(x_train, y_train, x_test):
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def make_gbm(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def group_importance(model, x_test, y_test, groups, rng, repeats=3):
    base = safe_auc(y_test, model.predict_proba(x_test)[:, 1])
    drops = {}
    for name, columns in groups.items():
        values = []
        for _ in range(repeats):
            shuffled = x_test.copy()
            shuffled[columns] = x_test[columns].values[rng.permutation(len(x_test))]
            values.append(base - safe_auc(y_test, model.predict_proba(shuffled)[:, 1]))
        drops[name] = float(np.mean(values))
    return drops


def cross_validate(df, outcome, p_columns=None, daily=None, fit_gbm=True, importance=False, seed=0):
    p_columns = p_columns or [P_PRIMARY]
    rng = np.random.default_rng(seed)
    parts, importances = [], []
    for season in sorted(df["season"].unique()):
        test = df[df["season"] == season]
        train = df[df["season"] != season]
        y_train, y_test = train[outcome].values, test[outcome].values
        if len(np.unique(y_train)) < 2:
            continue

        z_train = climatology_logit(train, train, outcome, True)
        z_test = climatology_logit(train, test, outcome, False)
        result = pd.DataFrame({"y": y_test, "m0": expit(z_test)}, index=test.index)

        for column in p_columns:
            x_train = np.c_[z_train, np.log1p(train[column].values)]
            x_test = np.c_[z_test, np.log1p(test[column].values)]
            result["m1" if column == P_PRIMARY else f"m1|{column}"] = fit_logistic(x_train, y_train, x_test)

        if fit_gbm:
            model = make_gbm(seed).fit(train[M2_FEATURES], y_train)
            result["m2"] = model.predict_proba(test[M2_FEATURES])[:, 1]
            if importance:
                drops = group_importance(model, test[M2_FEATURES], y_test, FEATURE_GROUPS, rng)
                importances.append({"season": season, **drops})
            if daily is not None:
                train_x, test_x = train[M2_FEATURES].copy(), test[M2_FEATURES].copy()
                for frame, source in [(train_x, train), (test_x, test)]:
                    frame["T13_anom"], frame["D13_anom"] = fold_anomalies(daily, season, source)
                clean_model = make_gbm(seed).fit(train_x, y_train)
                result["m2_noleak"] = clean_model.predict_proba(test_x)[:, 1]
        parts.append(result)

    oof = pd.concat(parts)
    oof = df[["prov", "region", "date", "season", "month"]].join(oof, how="inner")
    return oof, pd.DataFrame(importances)


def season_metrics(oof, models):
    rows = []
    for (region, season), g in oof.groupby(["region", "season"]):
        if g["y"].nunique() < 2:
            continue
        row = {"region": region, "season": season}
        for m in models:
            row[f"auc_{m}"] = roc_auc_score(g["y"], g[m])
            row[f"pr_{m}"] = average_precision_score(g["y"], g[m])
            row[f"brier_{m}"] = brier_score_loss(g["y"], g[m])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_one_sided = (1 + np.sum(means <= 0)) / (n_boot + 1)
    return values.mean(), lo, hi, p_one_sided


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def delta_table(metrics, new, old, regions, metric="auc", n_boot=2000):
    rows = []
    for region in regions:
        m = metrics[metrics["region"] == region]
        delta = m[f"{metric}_{new}"] - m[f"{metric}_{old}"]
        mean, lo, hi, p = bootstrap_mean(delta, n_boot)
        rows.append({
            "region": region, "seasons": len(m),
            f"{metric}_{old}": m[f"{metric}_{old}"].mean(), f"{metric}_{new}": m[f"{metric}_{new}"].mean(),
            "delta": mean, "ci_lo": lo, "ci_hi": hi, "p": p,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    return out


def brier_skill(metrics, model, reference="m0"):
    return 1 - metrics.groupby("region")[f"brier_{model}"].mean() / metrics.groupby("region")[f"brier_{reference}"].mean()
