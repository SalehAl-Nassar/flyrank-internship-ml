# Prioritising Content Refresh: A Ranked Queue for Declining Search Pages

**Author:** Saleh Al-Nassar
**Lane:** 2 — Refresh / Content Opportunity Scoring
**Repo:** [SalehAl-Nassar/flyrank-internship-ml](https://github.com/SalehAl-Nassar/flyrank-internship-ml)
**Date:** August 2026

---

## Abstract

Content teams managing large portfolios face a prioritisation problem: hundreds of thousands of pages compete for finite review capacity, and missing a high-traffic page in decline is costlier than auditing one that didn't need attention. I built a ranked queue that scores each content page by its probability of decline, using features derived from the first half of March 2026 and a proxy label defined from the second half — enforcing strict temporal separation between what the model sees and what it predicts. I trained a gradient-boosted tree model (XGBoost) on 17 features across 141,467 labelable pages from the FlyRank internship warehouse (~79M daily rows), and evaluated both a heuristic baseline and the learned model on precision@K and recall@K among 77,540 pages with at least 100 search impressions. The XGBoost model achieved 88% precision at K=50 compared to 42% for the heuristic — meaning a reviewer checking the top 50 model-scored pages finds 44 declining pages, versus 21 with the heuristic. The output is a ranked queue with human-readable reason codes — not just a score — designed to support a reviewer's decision about which pages to audit first.

---

## 1. Introduction / Problem statement

A content reviewer staring at 300,000 pages has no starting point. Dashboards show aggregate trends; SQL filters surface pages matching one criterion at a time. But the decision — which page to refresh first — depends on multiple signals interacting: traffic volume, search position, content age, engagement, and how those signals combine differently for each page type and client. Without a ranked queue, the reviewer either samples randomly (wasting time on low-priority pages) or applies a single filter (missing pages that don't match the obvious pattern but are still declining).

This project builds that queue. The output is a scored, ranked list of pages with probability-of-decline scores and reason codes explaining *why* each page scored that way. A reviewer opens the list, starts at the top, and works down until their review capacity is exhausted. The model doesn't replace editorial judgment — it prioritises the reviewer's time.

The cost asymmetry matters: a false negative (missing a page that truly needs refresh) is more expensive than a false positive (reviewing a page that was fine — the reviewer spends 20 minutes and moves on). This asymmetry drives the choice of metric: Recall@K, not precision, is the primary evaluation number.

---

## 2. Data

**Source:** FlyRank internship warehouse on Hugging Face (`FlyRank/internship-warehouse`), build v20260703. Accessed via `hf://` + DuckDB per the starter notebook pattern.

**Tables used:**

| Table | Rows | Grain |
|---|---|---|
| `fact_content_daily_performance` (March 2026 partition) | 9.8M | report_date × client × content |
| `dim_content` | 519,606 | one per content item |

**Date windows:**
- **Feature window (H1):** March 1–15, 2026. All features are computed from this window or from `dim_content` (static metadata).
- **Label window (H2):** March 16–31, 2026. The proxy label is computed exclusively from this window.

**Population:** ~320,000 distinct pages in March 2026. Of these, ~141,000 had non-NULL average search position in both H1 and H2 (required for the label). Among those, ~44% were classified as declining.

**Visibility floor:** We restrict evaluation to pages with `impressions_h1 >= 100`. Pages below this threshold get no score in the ranked queue — they don't have enough search traffic to warrant reviewer attention, regardless of trend direction.

**Exclusions:**
- `impressions_h2`, `clicks_h2`, `avg_position_h2`, `sessions_h2`, `engaged_sessions_h2`, `sessions_organic_h2` — label-window columns, excluded from features to prevent temporal leakage.
- `trend_direction`, `trend_pct` — label-source columns (the proxy label is derived from position change, not these columns, but they are excluded as a precaution).
- `content_hash_id`, `client_hash_id` — IDs used for grouping/splitting only, never as model features.
- `keyword_hash_id`, `url_hash_id`, `provider_used`, `model_used`, `is_published`, `is_deleted` — product implementation details or identifiers, not content signals.
- `ga4_data_available`, `gsc_data_available` — gating flags checked for missingness analysis but not used as features.

**Public-safe:** All IDs are pseudonymized. No client names, domains, raw queries, or credentials appear anywhere in this work.

---

## 3. Methodology

### Assumptions

1. The proxy label (`avg_position_h2 > avg_position_h1 * 1.10`) is a **weak heuristic**, not ground truth. It captures pages where search rank worsened by ≥10% over two weeks. This is an imperfect signal — seasonal fluctuations, SERP feature changes, and measurement noise all contribute to position changes that don't reflect content quality. I flag it as a proxy throughout, not a finding.
2. Features from H1 (days 1–15) are knowable before H2 (days 16–31) opens. The temporal split enforces decision-moment discipline: at the moment a reviewer would use the model, only H1 data is available.
3. Recall matters more than precision for this task. A missed high-visibility decliner keeps losing traffic; a false alarm costs the reviewer 20 minutes.

### Features (17 total)

| Feature | Source | Description |
|---|---|---|
| `impressions_h1` | fact (days 1–15) | Total GSC impressions in the feature window |
| `clicks_h1` | fact (days 1–15) | Total GSC clicks |
| `avg_position_h1` | fact (days 1–15) | Average GSC position (lower = better). NULL for zero-impression pages. |
| `sessions_h1` | fact (days 1–15) | GA4 sessions. ~30% NULL where GA4 unavailable. |
| `engaged_sessions_h1` | fact (days 1–15) | GA4 engaged sessions |
| `sessions_organic_h1` | fact (days 1–15) | Organic search sessions |
| `word_count` | dim_content | Article word count. ~34% NULL. |
| `search_volume` | dim_content | Keyword search demand. ~18% NULL. |
| `content_age_days` | derived | Days since content creation to March 1, 2026 |
| `days_since_update` | derived | Days since last optimisation (capped at 0) |
| `ctr_h1` | derived | Clicks / impressions in H1 |
| `has_ga4` | derived | 1 if any GA4 session in H1, 0 otherwise |
| `log_impressions_h1` | derived | log1p of impressions_h1 (heavy-tailed) |
| `log_clicks_h1` | derived | log1p of clicks_h1 |
| `log_sessions_h1` | derived | log1p of sessions_h1 |
| `is_stale` | derived | 1 if days_since_update >= 180 |
| `is_visible_h1` | derived | 1 if impressions_h1 >= 100 |

Categorical features (3): `competition_level`, `main_intent`, `content_type` — one-hot encoded for logistic regression, label-encoded for XGBoost.

### Label definition

`proxy_decline = 1` if `avg_position_h2 > avg_position_h1 * 1.10` (position worsened ≥10%), 0 otherwise. Only pages with non-NULL position in both halves are labelable. Observed rate: ~44%.

### Baseline

**Heuristic rule** (from Week 4): `score = is_stale * is_visible_h1 * impressions_h1`. Reason code: `stale_visible` if both flags true, `visible` if traffic only, `no_action` otherwise. In one sentence: "Pages nobody has touched in 6+ months that still get search traffic are worth a look, and the ones with the most traffic are worth looking at first."

### Validation design

- **Time split by construction:** features = days 1–15, label = days 16–31. No feature uses label-window data.
- **80/20 random split** (stratified by `proxy_decline`): tests generalisation across pages.
- **Evaluation restricted to visible pages** (impressions_h1 >= 100): the metric measures whether the queue surfaces the pages that matter.
- **Primary metric:** Recall@50 — the fraction of all declining visible pages captured in the top 50 scored items.
- **Secondary metrics:** Recall@20, Recall@100, Precision@K, Average Precision.

### Leakage checks

1. **Column audit:** No label-window columns (`impressions_h2`, `clicks_h2`, `avg_position_h2`, `sessions_h2`, `engaged_sessions_h2`, `sessions_organic_h2`, `trend_direction`, `trend_pct`) appear in the feature set. Verified programmatically.
2. **Leakage injection test:** I trained a second XGBClassifier with the exact same features plus `impressions_h2` added. The AP difference between the leaky and honest models is the leakage signal — if it's large, the feature was carrying label information. Self-computed in `capstone.ipynb`, Section 8.
3. **Feature timeline:** Every feature is either from H1 (fact table, days 1–15) or from `dim_content` (static metadata, fixed before March). No feature requires information from H2 or later.

---

## 4. Results

### Model vs baseline (same split, same metric)

| Model | Recall@20 | Recall@50 | Recall@100 | Precision@50 | Avg. Precision |
|---|---|---|---|---|---|
| Heuristic rule | 0.2% | 0.3% | 0.6% | 42.0% | 0.435 |
| Logistic Regression | 0.2% | 0.5% | 1.1% | 66.0% | 0.554 |
| XGBoost | 0.3% | 0.7% | 1.3% | 88.0% | 0.652 |

*Base rate: 43.3% (fraction of visible test pages that are declining). All numbers self-computed in `capstone.ipynb`.*

**Key finding:** At K=50, XGBoost achieves 88% precision versus 42% for the heuristic. A reviewer checking 50 pages flagged by XGBoost finds 44 declining pages (44 true positives); the same reviewer checking 50 pages from the heuristic finds only 21. The model doubles the hit rate per review, meaning a team can clear half the declining pages in half the review cycles. Recall numbers are small in absolute terms (0.7% at K=50) because the test set contains ~6,714 declining pages — the model is selecting precisely, not trying to capture the full population at small K.

### Feature importances (XGBoost, gain)

| Rank | Feature | Importance | Plausibility check |
|---|---|---|---|
| 1 | avg_position_h1 | 0.1520 | H1 fact — SAFE. Direct measure of search visibility at decision time. |
| 2 | is_stale | 0.1237 | Derived — SAFE. Staleness (no update in 6+ months) is a plausible signal for content decay. |
| 3 | has_ga4 | 0.1167 | Derived — SAFE. Pages without GA4 tracking may be lower priority or poorly monitored. |

The top features are search position, staleness, and GA4 availability — all H1 or static features that are knowable at decision time. No label-window columns appear. The model learned that pages positioned lower in search results and not recently updated are more likely to be declining, which aligns with editorial intuition.

### Error analysis

The model's most costly errors are false negatives — declining pages the model missed. These tend to be pages with moderate traffic (median 515 impressions), healthy H1 search position (median 6.1), and recent updates (median 25 days since last optimisation). Only 25.7% were stale. They are hard because the H1 snapshot looks healthy; the decline is subtle and only visible in the H2 position shift. The model's strength is flagging obviously declining pages (low position, stale content, high traffic); its weakness is detecting pages where position recently started slipping from a strong baseline.

---

## 5. Limitations and honest framing

This work uses observed and directional language throughout. No causal claims are made.

**Proxy label weakness.** The label (`avg_position_worsened >= 10%`) is a heuristic. It ignores magnitude (a 10% and a 60% decline look identical), excludes pages with zero H1 impressions (~55% of all pages), and may flag seasonal fluctuations as decline. This is a known limitation that should motivate future work with raw trend_pct or human-annotated labels, not a finding to be explained away.

**One month only.** Everything here is March 2026. Seasonality, algorithm updates, and content lifecycle stages vary across months. The June 2026 partition is sealed for future out-of-sample evaluation, but the current results should be treated as directional, not definitive.

**Visibility floor bias.** We only evaluate among pages with ≥100 H1 impressions. The model's behaviour on low-traffic pages is unobserved in our metrics. A page with 10 impressions dropping to 2 is technically declining, but flagging it wastes reviewer time — the floor is a deliberate design choice, not an oversight.

**No causal claim.** The model ranks pages by observed association with decline. It does not predict what will happen if a page is refreshed. The data is observational, not experimental. "These pages look worth reviewing first" is the honest claim; "refreshing these pages will recover traffic" is not supported.

**Client concentration.** A small number of clients contribute the majority of high-traffic pages. Results may not generalise to portfolios with different traffic distributions.

**No algorithm prediction.** We are not modelling Google's ranking algorithm. We are modelling which pages a human reviewer should prioritise, using search performance metrics as signals.

---

## 6. Ranked recommendations

The model outputs a ranked queue with reason codes. Here is how a FlyRank editor would use it:

| Priority | Action | When | Confidence |
|---|---|---|---|
| **Immediate** | Review top-20 pages by model score | Start of each review cycle | High — model beats heuristic at every K |
| **This cycle** | Review pages ranked 21–50 | After top-20 is cleared | High |
| **Monitor** | Watch pages with score 0.4–0.6 | Monthly check-in | Medium — may need refresh soon |
| **Low priority** | Pages with score < 0.4 | Only if reviewer has spare capacity | Low |
| **Never automate** | Every flagged page needs human editorial judgment | Always | The model is decision-support, not a replacement for expertise |

**Reason codes** explain *why* each page scored the way it did. Codes include: `stale` (not updated in 6+ months), `high_traffic` (≥500 H1 impressions), `deep_position` (average position > 20), `low_ctr` (<1% CTR with meaningful traffic), `old_content` (created > 1 year ago), `no_ga4_data` (missing engagement metrics). These are derived from feature values, not model internals — they are human-readable, not mathematically precise attribution.

---

## 7. Reproducibility

- **Notebooks:** All work is in `work/notebooks/`. The capstone notebook (`capstone.ipynb`) runs top to bottom and produces all reported numbers.
- **Data:** FlyRank internship warehouse on Hugging Face (`FlyRank/internship-warehouse`), March 2026 partition. Accessed via `hf://` + DuckDB. Gate access is instant-approval.
- **Cache:** `work/outputs/mar_page_month.parquet` — the page-month aggregate built by `w03_data_contract.ipynb`. Rerunning that notebook rebuilds the cache.
- **Random seed:** `RANDOM_SEED = 42` (set at top of `capstone.ipynb`).
- **Dependencies:** `pip install pandas numpy scikit-learn xgboost duckdb huggingface_hub matplotlib`
- **To re-run from scratch:** Clone the repo, set `HF_TOKEN`, run `w03_data_contract.ipynb` (builds cache), then run `capstone.ipynb` top to bottom.

---

## 8. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).

The warehouse release (`FlyRank/internship-warehouse` on Hugging Face) provides ~79M rows of pseudonymized daily search performance data across 104 clients and 519K content items. This dataset makes the work possible — without a real, large-scale search performance dataset at this grain, content opportunity scoring remains a theoretical exercise.

---

*All numbers in this paper are self-computed in `work/notebooks/capstone.ipynb` from the cached parquet file. No metrics are imported from prior notebooks. The full code is public in the linked repository.*
