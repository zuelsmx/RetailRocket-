from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR


def compute_funnel(events: pd.DataFrame) -> pd.DataFrame:
    """Compute item-interest funnel at visitor-item level."""
    pair_events = (
        events.assign(value=1)
        .pivot_table(
            index=["visitorid", "itemid"],
            columns="event",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    for col in ["view", "addtocart", "transaction"]:
        if col not in pair_events.columns:
            pair_events[col] = 0

    viewed = pair_events[pair_events["view"] == 1]
    added = pair_events[pair_events["addtocart"] == 1]

    view_pairs = len(viewed)
    add_pairs = len(added)
    purchase_pairs = int((pair_events["transaction"] == 1).sum())

    metrics = {
        "view_pairs": view_pairs,
        "addtocart_pairs": add_pairs,
        "transaction_pairs": purchase_pairs,
        "view_to_addtocart_rate": safe_rate(int((viewed["addtocart"] == 1).sum()), view_pairs),
        "addtocart_to_transaction_rate": safe_rate(int((added["transaction"] == 1).sum()), add_pairs),
        "view_to_transaction_rate": safe_rate(int((viewed["transaction"] == 1).sum()), view_pairs),
    }
    return pd.DataFrame([metrics])


def compute_segments(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    user = events.groupby("visitorid").agg(
        user_events=("event", "size"),
        user_views=("event", lambda x: (x == "view").sum()),
        user_addtocarts=("event", lambda x: (x == "addtocart").sum()),
        user_transactions=("event", lambda x: (x == "transaction").sum()),
        active_days=("event_date", "nunique"),
    )
    user["has_purchase"] = (user["user_transactions"] > 0).astype(int)
    user["activity_segment"] = pd.qcut(
        user["user_events"].rank(method="first"), q=4, labels=["low", "mid_low", "mid_high", "high"]
    )

    item = events.groupby("itemid").agg(
        item_events=("event", "size"),
        item_views=("event", lambda x: (x == "view").sum()),
        item_addtocarts=("event", lambda x: (x == "addtocart").sum()),
        item_transactions=("event", lambda x: (x == "transaction").sum()),
    )
    item["item_conversion_rate"] = item["item_transactions"] / item["item_views"].replace(0, pd.NA)
    item["heat_segment"] = pd.qcut(
        item["item_events"].rank(method="first"), q=4, labels=["cold", "warm", "hot", "top"]
    )

    add_no_buy = (
        events.assign(value=1)
        .pivot_table(
            index=["visitorid", "itemid"],
            columns="event",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )
    for col in ["view", "addtocart", "transaction"]:
        if col not in add_no_buy.columns:
            add_no_buy[col] = 0
    add_no_buy = add_no_buy[(add_no_buy["addtocart"] == 1) & (add_no_buy["transaction"] == 0)]

    return {
        "user_segments": user.reset_index(),
        "item_segments": item.reset_index(),
        "addtocart_without_purchase": add_no_buy,
    }


def compute_category_segments(events: pd.DataFrame, item_categories: pd.DataFrame) -> pd.DataFrame:
    if item_categories.empty:
        return pd.DataFrame()
    merged = events.merge(item_categories[["itemid", "categoryid"]], on="itemid", how="inner")
    if merged.empty:
        return pd.DataFrame()
    category = merged.groupby("categoryid").agg(
        category_events=("event", "size"),
        category_items=("itemid", "nunique"),
        category_users=("visitorid", "nunique"),
        category_views=("event", lambda x: (x == "view").sum()),
        category_addtocarts=("event", lambda x: (x == "addtocart").sum()),
        category_transactions=("event", lambda x: (x == "transaction").sum()),
    )
    category["category_view_to_addtocart_rate"] = (
        category["category_addtocarts"] / category["category_views"].replace(0, pd.NA)
    )
    category["category_view_to_transaction_rate"] = (
        category["category_transactions"] / category["category_views"].replace(0, pd.NA)
    )
    return category.fillna(0).sort_values("category_events", ascending=False).reset_index()


def save_eda_figures(events: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    events["event"].value_counts().reindex(["view", "addtocart", "transaction"]).plot(kind="bar", ax=ax)
    ax.set_title("Event distribution")
    ax.set_xlabel("Event")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "event_distribution.png", dpi=160)
    plt.close(fig)

    daily = events.groupby(["event_date", "event"]).size().reset_index(name="events")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=daily, x="event_date", y="events", hue="event", ax=ax)
    ax.set_title("Daily event trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Events")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "daily_event_trend.png", dpi=160)
    plt.close(fig)

    item_heat = events.groupby("itemid").size().sort_values(ascending=False).head(50)
    fig, ax = plt.subplots(figsize=(8, 4))
    item_heat.reset_index(drop=True).plot(kind="bar", ax=ax)
    ax.set_title("Top 50 item heat distribution")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(output_dir / "item_heat_top50.png", dpi=160)
    plt.close(fig)


def safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)
