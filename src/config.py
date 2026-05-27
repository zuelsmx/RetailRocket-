from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

EVENTS_FILE = RAW_DATA_DIR / "events.csv"
ITEM_PROPERTIES_FILE = RAW_DATA_DIR / "item_properties.csv"
CATEGORY_TREE_FILE = RAW_DATA_DIR / "category_tree.csv"

VALID_EVENTS = {"view", "addtocart", "transaction"}
DEFAULT_HISTORY_DAYS = 21
DEFAULT_LABEL_DAYS = 7
RANDOM_STATE = 42
