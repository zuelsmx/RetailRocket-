from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EVENTS_FILE, VALID_EVENTS


def load_events(path: Path = EVENTS_FILE) -> pd.DataFrame:
    """Load and validate RetailRocket events.csv."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download Kaggle RetailRocket data and place events.csv in data/raw/."
        )

    events = pd.read_csv(path)
    required = {"timestamp", "visitorid", "event", "itemid"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events.csv missing required columns: {sorted(missing)}")

    unknown_events = set(events["event"].dropna().unique()).difference(VALID_EVENTS)
    if unknown_events:
        raise ValueError(f"Unexpected event values: {sorted(unknown_events)}")

    events = events.copy()
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="raise")
    events["event_time"] = pd.to_datetime(events["timestamp"], unit="ms")
    events["event_date"] = events["event_time"].dt.date
    events["hour"] = events["event_time"].dt.hour
    events["weekday"] = events["event_time"].dt.weekday
    events["is_weekend"] = events["weekday"].isin([5, 6]).astype(int)
    return events.sort_values("event_time").reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(events)),
        "visitors": int(events["visitorid"].nunique()),
        "items": int(events["itemid"].nunique()),
        "start_time": str(events["event_time"].min()),
        "end_time": str(events["event_time"].max()),
        "event_counts": events["event"].value_counts().to_dict(),
        "purchase_rate_events": float((events["event"] == "transaction").mean()),
    }
