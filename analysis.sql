-- ============================================================
-- ChurnSense — SQL Customer Churn Analysis
-- Author : Aditya Mahale
-- Dataset: IBM Telco Customer Churn
-- Purpose: Segment-wise churn metrics to support marketing
--          and retention strategy decisions.
-- ============================================================

-- ------------------------------------------------------------
-- 1. Overall churn rate
-- ------------------------------------------------------------
SELECT
    COUNT(*)                                          AS total_customers,
    SUM(Churn)                                        AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)          AS churn_rate_pct
FROM customers;


-- ------------------------------------------------------------
-- 2. Churn rate by contract type
--    Insight: Month-to-month contracts have the highest churn
-- ------------------------------------------------------------
SELECT
    Contract,
    COUNT(*)                                          AS total,
    SUM(Churn)                                        AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)          AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- 3. Churn rate by internet service type
--    Insight: Fiber optic users churn at ~42%
-- ------------------------------------------------------------
SELECT
    InternetService,
    COUNT(*)                                          AS total,
    SUM(Churn)                                        AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)          AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- 4. Churn rate across tenure lifecycle stages
--    Buckets match the tenure_group feature in the ML model
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0-1yr'
        WHEN tenure BETWEEN 13 AND 24 THEN '1-2yr'
        WHEN tenure BETWEEN 25 AND 48 THEN '2-4yr'
        WHEN tenure BETWEEN 49 AND 72 THEN '4-6yr'
        ELSE '6+yr'
    END                                               AS tenure_group,
    COUNT(*)                                          AS total,
    SUM(Churn)                                        AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)          AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- 5. Average monthly charges — churned vs retained
--    Insight: Churned customers tend to pay more per month
-- ------------------------------------------------------------
SELECT
    CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    ROUND(AVG(MonthlyCharges), 2)                          AS avg_monthly_charges,
    ROUND(AVG(TotalCharges),   2)                          AS avg_total_charges,
    COUNT(*)                                               AS n
FROM customers
GROUP BY Churn;


-- ------------------------------------------------------------
-- 6. High-risk churn segments (top combinations)
--    Used to target retention campaigns
-- ------------------------------------------------------------
SELECT
    Contract,
    InternetService,
    TechSupport,
    COUNT(*)                                           AS total,
    SUM(Churn)                                         AS churned,
    ROUND(100.0 * SUM(Churn) / COUNT(*), 2)           AS churn_rate_pct
FROM customers
GROUP BY Contract, InternetService, TechSupport
HAVING COUNT(*) > 20
ORDER BY churn_rate_pct DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 7. Retention rate by payment method
-- ------------------------------------------------------------
SELECT
    PaymentMethod,
    COUNT(*)                                           AS total,
    SUM(CASE WHEN Churn = 0 THEN 1 ELSE 0 END)        AS retained,
    ROUND(100.0 * SUM(CASE WHEN Churn = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS retention_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY retention_rate_pct DESC;


-- ------------------------------------------------------------
-- 8. Customers at highest churn risk (for CRM targeting)
--    Month-to-month + short tenure + no tech support
-- ------------------------------------------------------------
SELECT
    customerID,
    tenure,
    Contract,
    MonthlyCharges,
    InternetService,
    TechSupport
FROM customers
WHERE
    Contract     = 'Month-to-month'
    AND tenure   <= 12
    AND TechSupport = 'No'
    AND Churn    = 0          -- not yet churned — proactive retention targets
ORDER BY MonthlyCharges DESC
LIMIT 50;
