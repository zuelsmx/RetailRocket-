from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_sample_events(path: Path) -> None:
    events = make_sample_events()
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)


def make_sample_events() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2015-05-01")
    rows = []
    visitors = range(1, 41)
    items = range(100, 130)
    for day in range(70):
        current = start + pd.Timedelta(days=day)
        for visitor in visitors:
            item = int(rng.choice(list(items)))
            rows.append([to_ms(current + pd.Timedelta(hours=int(rng.integers(0, 24)))), visitor, "view", item])
            if rng.random() < 0.25:
                rows.append([to_ms(current + pd.Timedelta(hours=1)), visitor, "addtocart", item])
            if rng.random() < 0.08:
                rows.append([to_ms(current + pd.Timedelta(hours=2)), visitor, "transaction", item])
    events = pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid"])
    return events


def to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


if __name__ == "__main__":
    create_sample_events(Path("data/raw/events.csv"))
