# RetailRocket 电商用户行为漏斗分析与购买预测建模

本项目基于 Kaggle RetailRocket 用户行为日志，围绕 `view -> addtocart -> transaction` 链路构建电商用户行为漏斗，并使用历史行为预测未来购买，识别高意向未购买用户，为用户增长、召回和推荐排序优化提供分析依据。

## 数据说明

请从 Kaggle 下载 RetailRocket 数据集，并将文件放到：

```text
data/raw/events.csv
data/raw/item_properties_part1.csv
data/raw/item_properties_part2.csv
data/raw/category_tree.csv
```

当前主流程只强依赖 `events.csv`。`item_properties_part1.csv`、`item_properties_part2.csv` 和 `category_tree.csv` 用于可用商品类目分析，但不能默认每个商品都有完整清晰类目。

原始数据和中间数据不会上传 GitHub。

## 分析口径

核心行为链路：

```text
浏览 view -> 加购 addtocart -> 购买 transaction
```

核心指标：

- 浏览到加购转化率
- 加购到购买转化率
- 浏览到购买转化率
- 复购率
- 用户活跃度
- 商品热度
- 商品转化率

建模目标：

- 样本粒度：`visitorid-itemid`
- 候选样本：历史窗口内发生过交互的用户-商品对
- 默认窗口：历史 21 天行为预测未来 7 天是否购买
- 切分方式：时间滚动窗口，避免未来信息泄漏

## 项目结构

```text
.
├── data/
│   ├── raw/              # Kaggle 原始数据，本地保存，不上传
│   └── processed/        # 训练样本和中间数据，本地保存，不上传
├── reports/
│   └── figures/          # 漏斗、趋势、特征重要性、SHAP 图
├── sql/
│   └── funnel_metrics.sql
├── src/
│   ├── data.py           # 数据读取和校验
│   ├── eda.py            # EDA、漏斗、分层分析
│   ├── features.py       # 滚动窗口样本和特征工程
│   ├── model.py          # 模型训练、评估、Top-K 圈选
│   ├── explain.py        # SHAP 解释
│   └── run_pipeline.py   # 一键运行主流程
└── tests/
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行完整流程：

```bash
python -m src.run_pipeline --history-days 21 --label-days 7
```

运行测试：

```bash
pytest
```

如果本地暂时没有 Kaggle 原始数据，可以先生成小型合成数据验证工程流程：

```bash
python tests/create_sample_data.py
python -m src.run_pipeline --history-days 21 --label-days 7
```

注意：合成数据只用于工程校验，不能作为简历指标。

## 输出结果

主流程会生成：

- `reports/data_summary.json`：数据概览
- `reports/funnel_metrics.csv`：漏斗指标
- `reports/user_segments.csv`：用户活跃与购买分层
- `reports/item_segments.csv`：商品热度和转化分层
- `reports/category_segments.csv`：可用商品类目的行为分层
- `reports/model_metrics.csv`：模型评估指标
- `reports/top_intent_users.csv`：Top 高意向用户-商品对
- `reports/project_report.md`：项目报告
- `reports/figures/*.png`：分析图表

当前已在完整 RetailRocket `events.csv` 上完成一次复现，关键结果如下：

- 事件量：2,756,101
- 用户数：1,407,580
- 商品数：235,061
- 时间范围：2015-05-03 至 2015-09-18
- 浏览到加购转化率：2.32%
- 加购到购买转化率：30.71%
- 浏览到购买转化率：0.89%
- 21/7 滚动窗口样本正样本率约 0.022%
- 测试集最佳模型：Random Forest
- 测试集 AUC：0.9042
- 测试集 PR-AUC：0.0100
- 测试集 Top 10% Recall：0.7566
- 测试集 Top 20% Recall：0.8464

由于购买样本极度稀疏，本项目将 Top-K Recall 和 PR-AUC 作为更关键的业务评价指标，而不是 Accuracy。

## 模型方案

模型对比：

- Logistic Regression：可解释 baseline
- Random Forest：非线性 baseline
- LightGBM：主模型，若环境未安装会自动跳过

评价指标：

- AUC
- PR-AUC
- Recall
- Precision
- F1
- Top 10% Recall
- Top 20% Recall

购买行为通常极度稀疏，因此不使用 Accuracy 作为主指标。

## 业务策略

基于模型评分和解释结果，项目输出以下增长策略：

- 对 Top 10% 高意向未购买用户做加购提醒或召回。
- 对加购未购买用户进行差异化优惠触达。
- 对高热度、高转化商品提高召回和排序权重。
- 对高频浏览低转化用户做替代商品推荐或价格敏感型触达。

A/B Test 设计：

- 实验对象：模型圈选的高意向未购买用户
- 实验组：接收加购提醒、优惠触达或排序加权策略
- 对照组：保持原策略
- 核心指标：购买转化率
- 辅助指标：加购转化率、触达响应率、复购率

## GitHub 交付

推荐提交顺序：

```bash
git add .
git commit -m "init project structure and data pipeline"
git commit -m "add funnel analysis, purchase model, and report"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

推送前请确认：

- `data/raw/` 不包含在 Git 提交中。
- `data/processed/` 不包含在 Git 提交中。
- 简历和 README 只写真实 RetailRocket 数据跑出的指标。
