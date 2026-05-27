from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EVENTS_FILE, ITEM_PROPERTIES_FILE, ITEM_PROPERTIES_PART_FILES, VALID_EVENTS


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


def load_item_categories(chunksize: int = 500_000) -> pd.DataFrame:
    """Load available item category snapshots from item_properties files.

    RetailRocket splits item_properties into two part files on Kaggle. Category
    is stored as rows where property == "categoryid"; not every item is
    guaranteed to have a stable or complete category.
    """
    files = []
    if ITEM_PROPERTIES_FILE.exists():
        files.append(ITEM_PROPERTIES_FILE)
    files.extend(path for path in ITEM_PROPERTIES_PART_FILES if path.exists())
    if not files:
        return pd.DataFrame(columns=["itemid", "categoryid", "category_timestamp"])

    chunks = []
    for path in files:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            category_rows = chunk[chunk["property"].astype(str) == "categoryid"].copy()
            if category_rows.empty:
                continue
            category_rows["categoryid"] = pd.to_numeric(category_rows["value"], errors="coerce")
            category_rows = category_rows.dropna(subset=["categoryid"])
            category_rows["categoryid"] = category_rows["categoryid"].astype(int)
            category_rows["category_timestamp"] = pd.to_numeric(category_rows["timestamp"], errors="coerce")
            chunks.append(category_rows[["itemid", "categoryid", "category_timestamp"]])

    if not chunks:
        return pd.DataFrame(columns=["itemid", "categoryid", "category_timestamp"])

    categories = pd.concat(chunks, ignore_index=True)
    categories = categories.sort_values("category_timestamp").drop_duplicates("itemid", keep="last")
    return categories.reset_index(drop=True)


def summarize_category_coverage(events: pd.DataFrame, item_categories: pd.DataFrame) -> dict[str, object]:
    event_items = pd.DataFrame({"itemid": events["itemid"].drop_duplicates()})
    covered = event_items.merge(item_categories[["itemid", "categoryid"]], on="itemid", how="left")
    covered_items = int(covered["categoryid"].notna().sum())
    return {
        "event_items": int(len(event_items)),
        "items_with_category": covered_items,
        "item_category_coverage": float(covered_items / len(event_items)) if len(event_items) else 0.0,
        "unique_categories": int(item_categories["categoryid"].nunique()) if not item_categories.empty else 0,
    }
