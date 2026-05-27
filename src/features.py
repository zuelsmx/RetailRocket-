from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    history_days: int = 21
    label_days: int = 7


EVENT_MAP = {
    "view": "views",
    "addtocart": "addtocarts",
    "transaction": "transactions",
}


def build_rolling_dataset(events: pd.DataFrame, config: WindowConfig = WindowConfig()) -> pd.DataFrame:
    """Build rolling visitor-item samples from interacted pairs only."""
    min_time = events["event_time"].min().normalize()
    max_time = events["event_time"].max().normalize()
    total_days = (max_time - min_time).days
    step_days = config.label_days
    rows: list[pd.DataFrame] = []

    for start_offset in range(0, total_days - config.history_days - config.label_days + 1, step_days):
        history_start = min_time + pd.Timedelta(days=start_offset)
        label_start = history_start + pd.Timedelta(days=config.history_days)
        label_end = label_start + pd.Timedelta(days=config.label_days)
        window = build_single_window(events, history_start, label_start, label_end)
        if not window.empty:
            window["window_start"] = history_start
            window["label_start"] = label_start
            window["label_end"] = label_end
            rows.append(window)

    if not rows:
        raise ValueError("No rolling windows could be built. Check date range or window settings.")

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.sort_values(["label_start", "visitorid", "itemid"]).reset_index(drop=True)
    return dataset


def build_single_window(
    events: pd.DataFrame,
    history_start: pd.Timestamp,
    label_start: pd.Timestamp,
    label_end: pd.Timestamp,
) -> pd.DataFrame:
    history = events[(events["event_time"] >= history_start) & (events["event_time"] < label_start)].copy()
    labels = events[
        (events["event_time"] >= label_start)
        & (events["event_time"] < label_end)
        & (events["event"] == "transaction")
    ].copy()

    if history.empty:
        return pd.DataFrame()

    candidates = history[["visitorid", "itemid"]].drop_duplicates()
    user_features = aggregate_user_features(history, label_start)
    item_features = aggregate_item_features(history, label_start)
    pair_features = aggregate_pair_features(history, label_start)
    label_pairs = labels[["visitorid", "itemid"]].drop_duplicates().assign(label=1)

    dataset = (
        candidates.merge(user_features, on="visitorid", how="left")
        .merge(item_features, on="itemid", how="left")
        .merge(pair_features, on=["visitorid", "itemid"], how="left")
        .merge(label_pairs, on=["visitorid", "itemid"], how="left")
    )
    dataset["label"] = dataset["label"].fillna(0).astype(int)
    dataset = dataset.fillna(0)
    return dataset


def aggregate_user_features(history: pd.DataFrame, label_start: pd.Timestamp) -> pd.DataFrame:
    wide = event_counts(history, ["visitorid"], "user")
    activity = history.groupby("visitorid").agg(
        user_active_days=("event_date", "nunique"),
        user_unique_items=("itemid", "nunique"),
        user_last_time=("event_time", "max"),
        user_avg_hour=("hour", "mean"),
        user_weekend_share=("is_weekend", "mean"),
    )
    out = wide.merge(activity, on="visitorid", how="left")
    out["user_recency_days"] = (label_start - out["user_last_time"]).dt.total_seconds() / 86400
    out["user_conversion_rate"] = out["user_transactions"] / out["user_views"].replace(0, np.nan)
    out = out.drop(columns=["user_last_time"])
    return out.fillna(0).reset_index()


def aggregate_item_features(history: pd.DataFrame, label_start: pd.Timestamp) -> pd.DataFrame:
    wide = event_counts(history, ["itemid"], "item")
    activity = history.groupby("itemid").agg(
        item_unique_users=("visitorid", "nunique"),
        item_last_time=("event_time", "max"),
        item_avg_hour=("hour", "mean"),
        item_weekend_share=("is_weekend", "mean"),
    )
    out = wide.merge(activity, on="itemid", how="left")
    out["item_recency_days"] = (label_start - out["item_last_time"]).dt.total_seconds() / 86400
    out["item_conversion_rate"] = out["item_transactions"] / out["item_views"].replace(0, np.nan)
    out["item_addtocart_rate"] = out["item_addtocarts"] / out["item_views"].replace(0, np.nan)
    out = out.drop(columns=["item_last_time"])
    return out.fillna(0).reset_index()


def aggregate_pair_features(history: pd.DataFrame, label_start: pd.Timestamp) -> pd.DataFrame:
    wide = event_counts(history, ["visitorid", "itemid"], "pair")
    activity = history.groupby(["visitorid", "itemid"]).agg(
        pair_last_time=("event_time", "max"),
        pair_first_time=("event_time", "min"),
        pair_avg_hour=("hour", "mean"),
        pair_weekend_share=("is_weekend", "mean"),
    )
    out = wide.merge(activity, on=["visitorid", "itemid"], how="left")
    out["pair_recency_days"] = (label_start - out["pair_last_time"]).dt.total_seconds() / 86400
    out["pair_lifetime_days"] = (
        out["pair_last_time"] - out["pair_first_time"]
    ).dt.total_seconds().clip(lower=0) / 86400
    out["pair_has_addtocart"] = (out["pair_addtocarts"] > 0).astype(int)
    out["pair_has_purchase"] = (out["pair_transactions"] > 0).astype(int)
    out["pair_interaction_strength"] = (
        out["pair_views"] + 3 * out["pair_addtocarts"] + 5 * out["pair_transactions"]
    )
    out = out.drop(columns=["pair_last_time", "pair_first_time"])
    return out.fillna(0).reset_index()


def event_counts(history: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    counts = (
        history.pivot_table(index=keys, columns="event", values="timestamp", aggfunc="count", fill_value=0)
        .rename(columns=EVENT_MAP)
        .reset_index()
    )
    for name in EVENT_MAP.values():
        col = f"{prefix}_{name}"
        if name in counts.columns:
            counts = counts.rename(columns={name: col})
        else:
            counts[col] = 0
    keep = keys + [f"{prefix}_{name}" for name in EVENT_MAP.values()]
    return counts[keep].set_index(keys)


def temporal_train_valid_test_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_starts = sorted(dataset["label_start"].drop_duplicates())
    if len(label_starts) < 3:
        raise ValueError("Need at least 3 rolling windows for train/valid/test split.")
    train_ends_at = max(1, int(len(label_starts) * 0.6))
    valid_ends_at = max(train_ends_at + 1, int(len(label_starts) * 0.8))
    train_windows = set(label_starts[:train_ends_at])
    valid_windows = set(label_starts[train_ends_at:valid_ends_at])
    test_windows = set(label_starts[valid_ends_at:])
    return (
        dataset[dataset["label_start"].isin(train_windows)].copy(),
        dataset[dataset["label_start"].isin(valid_windows)].copy(),
        dataset[dataset["label_start"].isin(test_windows)].copy(),
    )
