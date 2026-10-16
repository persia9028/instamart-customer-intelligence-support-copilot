# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements the four-part capstone using the fictional deterministic data created by `generate_data.py` and the local offline model simulator in `mock_llm.py`. The two supplied files remain unchanged. No hosted model, network service, API key, image, or chart is used. Monetary values are in INR.

The pipeline cleans the raw exports into CSV and SQLite, segments customers and predicts lapse, triages privacy-redacted tickets, and answers policy questions while simulating safe refund actions.

## Reproduce

Run these commands from the repository root in a POSIX shell. Python 3.11 or newer is required.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a.py`
6. `python validate_part_a.py`
7. `python part_b.py`
8. `python part_c.py`
9. `python part_d.py`

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` and then use the same `python` commands. The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The project scripts create their output directories and overwrite reproducible derived files. `part_c.py` prints retry lines while it runs.

## Part A

### Raw audit

The audit read customers (1,200 × 5), products (40 × 4), and orders (13,084 × 9). Empty strings were treated as missing for the missing-value report; for tickets, `keep_default_na=False` preserves empty text values.

| Table | Column | pandas dtype | Missing count | Missing % |
|---|---|---:|---:|---:|
| customers | customer_id | string | 0 | 0.00% |
| customers | city | string | 0 | 0.00% |
| customers | signup_date | string | 0 | 0.00% |
| customers | age | float64 | 61 | 5.08% |
| customers | gender | string | 30 | 2.50% |
| products | product_id | string | 0 | 0.00% |
| products | product_name | string | 0 | 0.00% |
| products | category | string | 0 | 0.00% |
| products | price_inr | int64 | 0 | 0.00% |
| orders | order_id | string | 0 | 0.00% |
| orders | customer_id | string | 0 | 0.00% |
| orders | product_id | string | 0 | 0.00% |
| orders | quantity | int64 | 0 | 0.00% |
| orders | order_ts | string | 0 | 0.00% |
| orders | status | string | 0 | 0.00% |
| orders | discount_pct | int64 | 0 | 0.00% |
| orders | order_amount | string | 53 | 0.41% |
| orders | delivery_minutes | float64 | 2,541 | 19.42% |

The raw order audit found 119 exact duplicate rows, 656 pending rows, 12,309 distinct order IDs, 120 non-numeric order amounts, 106 negative delivery times, and 2,541 blank delivery times. Customers had 61 blank ages, 6 ages outside 10–100, 24 raw city spellings, 10 nonblank raw gender spellings, and 30 blank genders.

### Cleaning and validation

| Rule | Values/rows changed |
|---|---:|
| Standardize city by trimming and title-casing | 476 city values |
| Map gender variants and blanks to male/female/unknown | 950 values |
| Fill blank ages by city median; replace impossible ages before filling | 61 blank + 6 impossible ages |
| Remove exact order duplicates | 119 rows |
| Keep latest row per order ID after exact de-duplication | 656 older/pending rows |
| Convert invalid amounts to missing and calculate from price, quantity, discount | 112 amounts imputed after order de-duplication |
| Set negative delivery times missing | 105 remaining negative values |
| Fill missing completed-order delivery time with completed product-category median | 774 values |
| Keep cancelled delivery times missing | 1,199 values |

Cities were normalized to six values. Counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, and Mumbai 184. Gender counts are female 610, male 560, and unknown 30. Ages are complete and range from 18 to 60. Signup dates were parsed as dates.

The clean orders contain 12,309 rows: 11,110 completed and 1,199 cancelled, with no pending rows. There are no missing order amounts, no missing completed-order delivery times, and exactly 1,199 missing cancelled-order delivery times. Completed-order revenue is INR 2,998,668.80. The clean CSVs are in `data/clean/`; the SQLite database at `data/instamart.db` contains `customers`, `products`, and `orders`. The validation script checks the cleaning checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL and pandas analysis

The seven queries in `sql/queries.sql` were executed through `pd.read_sql_query`:

| Query | Result |
|---|---|
| Q1: completed orders over INR 500, customer spend over INR 8,000 | 30 customers |
| Q2: revenue by city and product category | 30 cells; total INR 2,998,668.80; largest is Bengaluru / Electronics at INR 233,909.45 |
| Q3: products above average catalogue price | 11 products |
| Q4: customers without completed orders | 25 customers |
| Q5: customers above average completed spend | 550 customers; average across spending customers INR 2,552.06 |
| Q6: spend ranking within city | Top: Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10. Eight rows have RANK different from ROW_NUMBER because tied spend shares a rank while row numbers still advance. |
| Q7: daily revenue and cumulative total | 183 days; final cumulative revenue INR 2,998,668.80 |

A pandas pivot table independently totals INR 2,998,668.80. Completed-order city aggregation:

| City | Sum (INR) | Mean (INR) | Orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order amount has mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, and Q3 INR 297.00. The 1.5×IQR fences are −INR 283.50 and INR 645.30; there are no lower outliers and 897 upper outliers.

- **What is this data about?** Synthetic completed grocery, personal-care, snack, and electronics purchases across six cities.
- **What do the statistics look like?** Spend is strongly right-skewed: the mean INR 269.91 is twice the median INR 135.00, with a long upper tail.
- **Are there outliers or anomalies?** There are 897 amounts above the upper IQR fence; the source also contains planted invalid amounts, duplicate orders, and impossible delivery times, which the cleaning rules address.
- **What relationships exist?** Amount and discount have the largest absolute correlation, about −0.095; this small correlation does not establish causation.

## Part B

### RFM segmentation

RFM is measured over completed orders through 2025-09-30, at snapshot 2025-10-01. There are 1,175 customers; recency median 19 days (range 0–180), frequency mean 9.46 (maximum 33), monetary mean INR 2,552.06 (maximum INR 14,766.55).

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected: it is at the elbow, has the strongest silhouette among the candidates, and yields distinct segments the retention team can act on. Inertia always falls as K increases because more centroids can fit the observations more closely; that decrease alone rewards adding clusters and does not account for interpretability or over-segmentation.

| Segment ID | Business name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Steady Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Send replenishment reminders and low-cost basket offers. |
| 1 | Premium Basket | 138 | 24.12 | 6.77 | 7,129.70 | Offer early access and personalized high-value bundles. |
| 2 | Frequent Essentials | 419 | 8.77 | 19.40 | 3,410.02 | Protect loyalty with reliable essentials availability. |
| 3 | At-Risk Light Buyers | 182 | 109.04 | 2.20 | 685.99 | Try a small win-back incentive and monitor response. |

The highest-mean-monetary group, segment 1, is Premium. Cluster IDs are arbitrary labels, not a rank or ordered scale. PCA on scaled RFM explains 63.43% and 21.80% in its two components, 85.23% total. `outputs/customer_segments.csv` contains the RFM fields, segment ID/name, and Premium flag.

### Lapse prediction

The cohort includes customers with at least one completed order before 2025-09-01. Lapsed means no completed order from 2025-09-01 through 2025-09-30. There are 1,156 customers and 443 lapsed (38.32%); the stratified split has 924 training and 232 test rows. The always-not-lapsed baseline has 61.6% accuracy and lapsed-class F1 of 0. Features use only pre-cutoff orders plus the clean customer table; task 3 segment labels were excluded. No encoder, scaler, or resampler saw test rows, and no order dated 2025-09-01 or later entered the features. Every feature column was checked for missing values.

| Model | Grid and best parameters | Best CV F1 | Test confusion matrix (true rows 0,1; predicted columns 0,1) | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C ∈ {0.01, 0.1, 1, 10, 100}; best C=100 | 0.651 | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | depth ∈ {3,5,8}, leaf ∈ {5,10,20}; best depth=8, leaf=10 | 0.681 | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

The frequency scaler mean is 8.115 on training rows, versus 8.029 for the full cohort, because the stratified sample is a subset and its realized mean differs slightly. Logistic regression was selected because its explanation is simpler and its test F1 (0.648) is close to the forest’s (0.655). Missing a lapsing customer loses a chance to retain them, while wrongly targeting a loyal customer wastes incentive budget. The test set does not encode those business costs, so the retention team should choose a decision threshold using the cost of a missed lapse and an unnecessary offer.

Oversampling used training rows only, preserving the original training rows and adding lapsed duplicates to balance both classes at 570. On the unchanged test set, logistic regression precision/recall/F1 changed from 0.644/0.652/0.648 to 0.589/0.854/0.697. Recall moved most, rising by about 0.202, while precision fell slightly; this finds more likely lapsers at the cost of more false-positive campaigns.

The random forest’s five strongest features were frequency (0.280), monetary (0.271), recency (0.115), average delivery (0.061), and cancellation rate (0.052). Logistic coefficients were negative for frequency (−1.698), positive for recency (+0.040), and positive for monetary (+0.285): more frequent customers are less likely to lapse, while longer inactivity and historical spend raise predicted lapse odds in this fit. The saved `outputs/lapse_model.joblib` reloads and exactly reproduces test predictions. The Premium test-set retention file contains 10 customers, of whom 3 actually lapsed. The retention list is illustrative and not a leakage-free evaluation because the Premium flag is based on whole-window RFM, including September.

## Part C

Input guardrails skipped 4 whitespace-empty tickets and flagged 4 tickets over the 400-token limit; those 8 never reached the model. Redaction ran email, phone, then order-ID patterns and replaced 55 emails, 91 phone numbers, and 132 order IDs (278 substitutions total). Assertions reapply all patterns to the redacted prompt text and confirm no match remains; the model user message contains only the redacted ticket text.

The zero-shot and few-shot system prompts both contain `CLASSIFY`, enumerate billing/delivery/technical/fraud, and require JSON-only output. The few-shot prompt adds four labelled examples, including fraud. Every eligible call is retried up to three times with exponential jitter; each independent run recovered all 19 transient failures. Replies are parsed as JSON first; otherwise exactly one allowed category word is accepted from lowercase `[a-z]+` extraction. Unsupported JSON categories and all other malformed replies go to human review.

| Run | classified_json | classified_keyword | flagged_invalid_output | Accuracy on 147 accepted | Total tokens | Tokens / accepted ticket |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 135 | 12 | 9 | 85.7% (126/147) | 10,336 | 70.31 |
| Few-shot | 135 | 12 | 9 | 100% (147/147) | 20,320 | 138.23 |

Zero-shot confusion table (true category by predicted category; accepted outputs only):

| True \ Predicted | billing | delivery | fraud | technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Few-shot used 20,320 tokens versus 10,336 for zero-shot, an extra 9,984 tokens or about 68 more per accepted ticket. It corrected all 21 zero-shot errors, raising accuracy from 85.7% (126/147) to 100% (147/147); those errors were 11 fraud and 10 technical tickets labeled billing. The extra cost is worth paying when the cost of misrouting those fraud and technical cases exceeds about 68 additional tokens per accepted ticket. Invalid replies, empty/long tickets, and budget-exhausted tickets are rule-routed to human review; accepted fraud cases go to the security team, Premium customers to the premium desk, and other cases to support.

The zero-shot budget demonstration used 6,000 tokens and stopped 65 tickets as `budget_exhausted` after the running total reached 6,036; the budget is checked before each call, so the last accepted call can cross the threshold. `outputs/triage_results.csv` is the 164-row few-shot run with budget 100,000 and segment-based routing. `outputs/triage_budget_demo.csv` contains the budget demonstration.

## Part D

### Policy RAG

The five policy files produce 15 chunks with 40 words per chunk and 10-word overlap: refund 4, delivery 3, escalation 3, payment 3, privacy 2. Retrieval uses a fitted `TfidfVectorizer` and cosine similarity; “refund policy late delivery” retrieves three refund-policy chunks. Term frequency measures how often a query term occurs in a chunk, while inverse document frequency reduces the weight of terms common across chunks. TF-IDF can miss equivalent policy wording when a question uses different words from the document.

| Chunking | Expected-document retrieval | Expected fact in answer | Unanswerable fallback |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| No overlap | 12/12 | 10/12 | 2/2 |

Without overlap, “What delivery time makes a completed order eligible for a full refund?” fails to return the exact fact `more than 45 minutes` because the answer spans adjacent chunks; retrieval returns the sentence fragment “receives a full refund of the order amount.”

### Refund agent

The agent exposes five tools through `TOOLS_SPEC` and `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. Refunds apply only to completed orders delivered over 45 minutes; eligible amounts above INR 1,000 are escalated instead of issued. The ledger tools recheck order eligibility and the requested amount; automatic issue blocks amounts above INR 1,000, while escalation records eligible amounts above that limit. Seven supplied scenarios:

| Scenario | Status | Refund amount (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 789 |
| S2 | REFUND | 135.00 | 5 | 789 |
| S3 | ESCALATE | 1,999.00 | 5 | 776 |
| S4 | DENIED | 0.00 | 4 | 531 |
| S5 | DENIED | 0.00 | 3 | 296 |
| S6 | NOT_FOUND | 0.00 | 2 | 70 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,655 |

The S1 token-limit demonstration with limit 200 ended `FALLBACK_TOKEN_LIMIT`. The timeout demonstration with `time_limit_seconds=0` ended `FALLBACK_TIMEOUT` before its first call. For crash recovery, S1 checkpointed after the second tool and resumed with this exact sequence: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`; the final status was REFUND for INR 128.25 with no repeated tool execution. Checkpoints are stored in `state/`, runs in `outputs/agent_runs.jsonl`, and ledger events in `outputs/refund_ledger.jsonl`.

ReAct used 5 calls and 789 tokens for S1, 3 calls and 296 tokens for cancelled S5, and 2 calls and 70 tokens for missing-order S6. A plan-and-execute agent could ask for the complete action plan once and use fewer calls on the normal path, though a large plan can increase the first response and still needs validation. ReAct gets each tool observation before choosing the next action, so it can stop cleanly on a cancelled order or NOT_FOUND result; a one-shot plan must be checked carefully before execution or replanned after an unexpected result. For refunds I would choose ReAct because the direct order and policy observations control whether money is issued or escalated, and the extra calls are modest for this small workflow.

## Limitations and Next Steps

All customer and policy data are synthetic. The deterministic mock is useful for reproducible evaluation but does not establish real-world language-model quality. The lapse model uses one temporal cutoff and a small feature set; production use would need time-based validation, monitoring, and cost-based campaign thresholds. The retention list is illustrative and not a leakage-free evaluation because its Premium flag comes from whole-window RFM including the September label period. Refund actions write only to a simulated local ledger; no payment system is contacted. The public submission must be pushed to the single `main` branch and linked directly to that branch.
