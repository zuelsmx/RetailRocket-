-- Example SQL口径：visitor-item 粒度的 view -> addtocart -> transaction 漏斗。
-- 可在 DuckDB / SQLite 临时表 events 上执行；真实项目代码中使用 Pandas 复现同一口径。

WITH pair_flags AS (
    SELECT
        visitorid,
        itemid,
        MAX(CASE WHEN event = 'view' THEN 1 ELSE 0 END) AS has_view,
        MAX(CASE WHEN event = 'addtocart' THEN 1 ELSE 0 END) AS has_addtocart,
        MAX(CASE WHEN event = 'transaction' THEN 1 ELSE 0 END) AS has_transaction
    FROM events
    WHERE event IN ('view', 'addtocart', 'transaction')
    GROUP BY visitorid, itemid
)
SELECT
    SUM(has_view) AS view_pairs,
    SUM(has_addtocart) AS addtocart_pairs,
    SUM(has_transaction) AS transaction_pairs,
    1.0 * SUM(CASE WHEN has_view = 1 AND has_addtocart = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(has_view), 0) AS view_to_addtocart_rate,
    1.0 * SUM(CASE WHEN has_addtocart = 1 AND has_transaction = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(has_addtocart), 0) AS addtocart_to_transaction_rate,
    1.0 * SUM(CASE WHEN has_view = 1 AND has_transaction = 1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(has_view), 0) AS view_to_transaction_rate
FROM pair_flags;
