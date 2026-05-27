from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR
from .model import feature_columns


def save_shap_summary(model: object, sample: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> bool:
    """Save SHAP summary plot when shap and a compatible model are available."""
    try:
        import shap
    except ImportError:
        return False

    if not hasattr(model, "predict_proba") or not hasattr(model, "booster_"):
        return False

    X = sample[feature_columns(sample)].head(2000)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    return True
