from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FIGURES_DIR, RANDOM_STATE

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


META_COLUMNS = {"visitorid", "itemid", "label", "window_start", "label_start", "label_end"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[col])]


def train_models(train: pd.DataFrame) -> dict[str, object]:
    X = train[feature_columns(train)]
    y = train["label"]
    if y.nunique() < 2:
        raise ValueError("Training data must contain both positive and negative labels.")

    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos_weight = max(1.0, neg / max(pos, 1))

    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )

    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(X, y)
    return fitted


def evaluate_models(models: dict[str, object], valid: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, split in [("valid", valid), ("test", test)]:
        X = split[feature_columns(split)]
        y = split["label"]
        for model_name, model in models.items():
            scores = predict_scores(model, X)
            rows.append({"model": model_name, "split": split_name, **classification_metrics(y, scores)})
    return pd.DataFrame(rows)


def predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def classification_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "top_10pct_recall": topk_recall(y_true, scores, 0.10),
        "top_20pct_recall": topk_recall(y_true, scores, 0.20),
    }
    metrics["auc"] = roc_auc_score(y_true, scores) if y_true.nunique() == 2 else np.nan
    metrics["pr_auc"] = average_precision_score(y_true, scores) if y_true.nunique() == 2 else np.nan
    return {key: float(value) for key, value in metrics.items()}


def topk_recall(y_true: pd.Series, scores: np.ndarray, k: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0
    n = max(1, int(len(scores) * k))
    top_idx = np.argsort(scores)[-n:]
    return float(y_true.iloc[top_idx].sum() / positives)


def select_best_model(models: dict[str, object], metrics: pd.DataFrame) -> tuple[str, object]:
    valid_metrics = metrics[metrics["split"] == "valid"].copy()
    best_name = valid_metrics.sort_values(["pr_auc", "auc"], ascending=False).iloc[0]["model"]
    return str(best_name), models[str(best_name)]


def save_feature_importance(model: object, columns: list[str], output_dir: Path = FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, Pipeline):
        estimator = model.named_steps.get("model")
        if hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_[0])

    if importances is None:
        return

    feature_importance = (
        pd.DataFrame({"feature": columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_importance["feature"][::-1], feature_importance["importance"][::-1])
    ax.set_title("Top feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def score_top_intent_samples(model: object, test: pd.DataFrame, keep_top_pct: float = 0.20) -> pd.DataFrame:
    scored = test[["visitorid", "itemid", "label", "label_start", "label_end"]].copy()
    scored["score"] = predict_scores(model, test[feature_columns(test)])
    scored["score_rank_pct"] = scored["score"].rank(pct=True, ascending=False)
    scored["segment"] = np.select(
        [scored["score_rank_pct"] <= 0.10, scored["score_rank_pct"] <= 0.20],
        ["top_10pct", "top_20pct"],
        default="other",
    )
    scored = scored[scored["score_rank_pct"] <= keep_top_pct]
    return scored.sort_values("score", ascending=False)
