from __future__ import annotations

import pandas as pd

from src.eda import compute_funnel
from src.features import WindowConfig, build_rolling_dataset, temporal_train_valid_test_split
from tests.create_sample_data import make_sample_events


def test_sample_pipeline_components() -> None:
    events = make_sample_events()
    events["event_time"] = pd.to_datetime(events["timestamp"], unit="ms")
    events["event_date"] = events["event_time"].dt.date
    events["hour"] = events["event_time"].dt.hour
    events["weekday"] = events["event_time"].dt.weekday
    events["is_weekend"] = events["weekday"].isin([5, 6]).astype(int)
    events = events.sort_values("event_time").reset_index(drop=True)

    assert set(events["event"].unique()) == {"view", "addtocart", "transaction"}
    assert events["event_time"].is_monotonic_increasing

    funnel = compute_funnel(events)
    assert {"view_to_addtocart_rate", "addtocart_to_transaction_rate", "view_to_transaction_rate"}.issubset(
        funnel.columns
    )

    dataset = build_rolling_dataset(events, WindowConfig(history_days=21, label_days=7))
    assert {"visitorid", "itemid", "label", "label_start", "label_end"}.issubset(dataset.columns)
    assert dataset["label"].isin([0, 1]).all()
    assert dataset.groupby(["visitorid", "itemid", "label_start"]).size().max() == 1

    train, valid, test = temporal_train_valid_test_split(dataset)
    assert train["label_start"].max() < valid["label_start"].min()
    assert valid["label_start"].max() < test["label_start"].min()
