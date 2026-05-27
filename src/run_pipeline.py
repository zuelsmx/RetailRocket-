from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_HISTORY_DAYS,
    DEFAULT_LABEL_DAYS,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from .data import load_events, summarize_events
from .eda import compute_funnel, compute_segments, save_eda_figures
from .explain import save_shap_summary
from .features import WindowConfig, build_rolling_dataset, temporal_train_valid_test_split
from .model import (
    evaluate_models,
    feature_columns,
    save_feature_importance,
    score_top_intent_samples,
    select_best_model,
    train_models,
)


def run_pipeline(history_days: int = DEFAULT_HISTORY_DAYS, label_days: int = DEFAULT_LABEL_DAYS) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    events = load_events()
    summary = summarize_events(events)
    (REPORTS_DIR / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    funnel = compute_funnel(events)
    funnel.to_csv(REPORTS_DIR / "funnel_metrics.csv", index=False, encoding="utf-8-sig")

    segments = compute_segments(events)
    for name, frame in segments.items():
        frame.to_csv(REPORTS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    save_eda_figures(events)

    dataset = build_rolling_dataset(events, WindowConfig(history_days, label_days))
    dataset.to_csv(PROCESSED_DATA_DIR / "rolling_samples.csv", index=False)

    train, valid, test = temporal_train_valid_test_split(dataset)
    for name, frame in [("train", train), ("valid", valid), ("test", test)]:
        frame.to_csv(PROCESSED_DATA_DIR / f"{name}.csv", index=False)

    models = train_models(train)
    metrics = evaluate_models(models, valid, test)
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    best_name, best_model = select_best_model(models, metrics)
    save_feature_importance(best_model, feature_columns(test))
    shap_saved = save_shap_summary(best_model, test)

    scored = score_top_intent_samples(best_model, test)
    scored.to_csv(REPORTS_DIR / "top_intent_users.csv", index=False, encoding="utf-8-sig")

    report = build_markdown_report(summary, funnel, metrics, best_name, shap_saved)
    (REPORTS_DIR / "project_report.md").write_text(report, encoding="utf-8")


def build_markdown_report(
    summary: dict[str, object],
    funnel: pd.DataFrame,
    metrics: pd.DataFrame,
    best_name: str,
    shap_saved: bool,
) -> str:
    funnel_row = funnel.iloc[0].to_dict()
    metrics_md = metrics.round(4).to_markdown(index=False)
    return f"""# RetailRocket 电商用户行为漏斗分析与购买预测建模报告

## 数据概览
- 事件量：{summary["rows"]}
- 用户数：{summary["visitors"]}
- 商品数：{summary["items"]}
- 时间范围：{summary["start_time"]} 至 {summary["end_time"]}
- 事件分布：{summary["event_counts"]}

## 漏斗指标
- 浏览到加购转化率：{funnel_row["view_to_addtocart_rate"]:.4f}
- 加购到购买转化率：{funnel_row["addtocart_to_transaction_rate"]:.4f}
- 浏览到购买转化率：{funnel_row["view_to_transaction_rate"]:.4f}

## 建模结果
样本粒度为 `visitorid-itemid`，候选样本仅来自历史窗口内发生过交互的用户-商品对。

{metrics_md}

主模型选择：`{best_name}`。

## 模型解释
- 特征重要性图：`reports/figures/feature_importance.png`
- SHAP summary：{"已生成" if shap_saved else "未生成，通常是因为未安装 shap 或当前主模型不兼容"}

## 增长策略
- 对 Top 10% 高意向未购买用户进行加购提醒或召回。
- 对加购未购买用户设置差异化优惠触达。
- 对高热度高转化商品提高召回和排序权重。
- 对高频浏览低转化用户做商品替代推荐或价格敏感型触达。

## A/B Test 方案
- 实验对象：模型圈选的高意向未购买用户。
- 实验组：接收加购提醒、优惠触达或排序加权策略。
- 对照组：保持原策略。
- 核心指标：购买转化率。
- 辅助指标：加购转化率、触达响应率、复购率。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-days", type=int, default=DEFAULT_HISTORY_DAYS)
    parser.add_argument("--label-days", type=int, default=DEFAULT_LABEL_DAYS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.history_days, args.label_days)
