# Project C.O.C.O. (Chief of Operations

# Conut Optimizer)

## Implementation Plan & Architecture Document

**Hackathon:** Conut AI Engineering Challenge
**Course:** AI Engineering
**Focus:** AI Systems, Full MLOps Pipeline, Business Feasibility, OpenClaw Integration

## 1. Executive Summary & Design Philosophy

Project C.O.C.O. is a production-ready, AI-driven Chief of Operations Agent for Conut. In a
12-hour sprint, the primary risk is "over-engineering" models that fail to converge or cannot be
deployed.
Our philosophy is **Maximum Business Value through Pragmatic AI**. We prioritize robust,
explainable algorithms (Graph Clustering and Gradient Boosted Trees) over highly
experimental deep learning. The result is a highly reliable decision-support system integrated
natively with OpenClaw. C.O.C.O. goes beyond simple predictions by translating mathematical
outputs into relative business impact scores, ensuring Conut executives can make immediate,
data-backed decisions.

## 2. System Architecture & "Stand Out" Engineering

Our architecture isolates data processing from the agent interaction layer to ensure fast
response times, safe execution, and high availability.
● **Data Ingestion Pipeline (The "Data Janitor"):** Automated Python scripts utilizing Regex
and Pandas to programmatically strip repetitive pagination headers from the raw CSV
exports before pushing them to our relational database.
● **Data Layer (Supabase/PostgreSQL):** Cleaned transactional and attendance data
ingested into indexed tables for sub-second aggregations.
● **Model Layer (XGBoost & NetworkX):** Pre-trained lightweight models stored as .pkl files
for instant inference.
● **Agent Gateway (FastAPI):** A high-performance REST API acting as a "Private Skills
Registry" for OpenClaw. **Crucially, this layer includes Graceful Fallbacks** : if an ML model
fails, the system automatically routes to heuristic moving averages rather than crashing
the chat interface.
● **Interface (OpenClaw):** Executives interact via a natural language chat interface powered
by OpenClaw.


## 3. Addressing the 5 Business Objectives

### Objective 1: Combo Optimization (Graph Clustering)

```
● Data Source: REP_S_00502.csv (Sales by customer/receipt).
● Approach: We model transactional data as a weighted graph (nodes = menu items, edges
= co-purchases). We apply the Louvain community detection algorithm to identify
distinct item clusters.
● Business Translation: The agent doesn't just suggest combos; it attaches an "Attach Rate
Probability" based on the edge weight, explaining why the combo makes financial sense.
```
### Objective 2: Demand Forecasting (Gradient Boosted Trees)

```
● Data Source: rep_s_00334_1_SMRY.csv and temporal features.
● Approach: Train an XGBoost regressor using lagged sales, day-of-week, and
time-of-day features.
● Why XGBoost? It handles tabular, non-linear operational data faster and better than deep
learning approaches, requiring minimal hyperparameter tuning during a 12-hour sprint.
```
### Objective 3: Expansion Feasibility (Spatial Proxies)

```
● Data Source: Internal branch metrics mapped to external OpenStreetMap (OSM) data.
● Approach: Profile the highest-performing Conut branch (e.g., Conut Jnah) to find its
"spatial signature" (density of universities, main roads, offices). C.O.C.O. uses a cosine
similarity score to rank new candidate locations against this signature.
```
### Objective 4: Shift Staffing Estimation (Unified Pipeline)

```
● Data Source: REP_S_00461.csv (Time & Attendance) merged with time-stamped sales.
● Approach: Calculate the historical "Throughput per Employee" (sales volume divided by
active staff hours). Our pipeline takes the predicted demand from Objective 2 and divides
it by this throughput metric.
● Business Translation: C.O.C.O warns of "Expected Lost Volume" if the recommended
staffing levels are ignored.
```
### Objective 5: Coffee & Milkshake Growth (Agentic Strategy)

```
● Data Source: rep_s_00191_SMRY.csv (Sales by Items).
● Approach: A specialized tool pulls bottom-quartile branches for these categories and
cross-references them with Objective 1 (Combos). The LLM synthesizes this to generate
highly specific, targeted marketing interventions (e.g., "Implement a morning combo at
Branch X to bridge the gap between their high pastry sales and low coffee attach rate").
```
## 4. The Explainable AI (XAI) Strategy

Business leaders do not trust black boxes. Project C.O.C.O. implements a low-latency "Hacker's
XAI" layer directly into the OpenClaw prompt:


1. **Native Feature Importance:** During inference, the API extracts
    model.feature_importances_ from the XGBoost trees.
2. **Strict Prompt Grounding:** The OpenClaw system prompt dictates that the agent _must_
    cite these specific metrics when generating a response.
3. **Example Output:** _"Expect a 15% demand spike tomorrow at Jnah. Primary drivers are the_
    _Weekend_Flag (42% importance) and Previous_Day_Volume (28% importance)."_

## 5. Reproducibility & OpenClaw Integration

To ensure the highest system engineering quality, the repository includes a Makefile for
absolute reproducibility.

1. **Private Skills Registry:** The FastAPI app exposes endpoints like /tools/predict_demand
    and /tools/get_combos.
2. **The Handshake:** OpenClaw routes specific user intents as JSON payloads to our FastAPI
    tools. FastAPI runs the model, returning the raw data and XAI context.
3. **One-Command Setup:** Reviewers can run make build-pipeline to clean the data, train the
    models, and start the C.O.C.O. agent server locally in under two minutes.

## 6. Execution Sprint Timeline

```
● Hour 1-3: Data Janitor script execution (regex cleaning of CSV reports) and database
population.
● Hour 4-5: NetworkX graph construction and Louvain clustering for combos.
● Hour 6-7: Temporal feature engineering, XGBoost training, and XAI weight extraction.
● Hour 8-10: Wrapping .pkl models into FastAPI routes, constructing the OpenClaw
manifest, and rigorous error-handling/fallback testing.
● Hour 11-12: Drafting the Executive Summary PDF and recording the OpenClaw demo
video.
```

