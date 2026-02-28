# Project C.O.C.O. (Chief of Operations Conut Optimizer)

> An AI-driven decision-support system for **Conut** bakery operations, integrated with OpenClaw.

## 🎯 Business Problem

Conut is a growing sweets and beverages business that needs data-driven operational intelligence. Project C.O.C.O. transforms raw POS data into actionable business decisions across five objectives:

1. **Combo Optimization** — Graph-based co-purchase analysis to find optimal product bundles
2. **Demand Forecasting** — XGBoost time-series predictions by branch *(Modeling Duo)*
3. **Expansion Feasibility** — Cosine-similarity scoring against the top-performing branch profile
4. **Shift Staffing** — Throughput-based staffing estimation from demand forecasts *(Modeling Duo)*
5. **Coffee & Milkshake Growth** — Percentile-ranked branch analysis with targeted interventions

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Raw CSV Data   │ ──▶ │  Data Pipeline   │ ──▶ │  Parquet Files │
│  (POS Exports)  │     │  (clean_data.py) │     │  (cleaned/)   │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                        ┌────────────────────────────────┘
                        ▼
                ┌───────────────┐
                │  Model Layer  │
                │  NetworkX     │ ──▶ combo_optimizer.pkl
                │  Cosine Sim   │ ──▶ expansion_scorer.pkl
                │  Percentile   │ ──▶ growth_strategy.pkl
                │  XGBoost*     │ ──▶ demand_model.pkl*
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐     ┌───────────────┐
                │   FastAPI     │ ◀──▶│   OpenClaw    │
                │  Port 8000    │     │   (Agent)     │
                │  /tools/*     │     └───────────────┘
                │  /skills      │
                └───────────────┘

* = Modeling Duo responsibility
```

## 🚀 Quick Start

```bash
# 1. Create venv and install dependencies
py -3.11 -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 2. Clean the raw data
python pipeline/clean_data.py

# 3. Train models
python models/train_all.py

# 4. Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or do it all at once:
make build-pipeline
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | System health check |
| `/tools/get_combos` | POST | Combo recommendations (Obj. 1) |
| `/tools/predict_demand` | POST | Demand forecasting (Obj. 2) |
| `/tools/expansion_feasibility` | POST | Expansion scoring (Obj. 3) |
| `/tools/estimate_staffing` | POST | Staffing estimation (Obj. 4) |
| `/tools/growth_strategy` | POST | Coffee/milkshake strategy (Obj. 5) |
| `/tools/combo_stats` | GET | Graph statistics |
| `/tools/branch_rankings` | GET | Branch similarity rankings |
| `/skills` | GET | OpenClaw skills manifest |

### Example: Combo Recommendations

```bash
curl -X POST http://localhost:8000/tools/get_combos \
  -H "Content-Type: application/json" \
  -d '{"target_item": "CHIMNEY THE ONE", "top_n": 3}'
```

### Example: Growth Strategy

```bash
curl -X POST http://localhost:8000/tools/growth_strategy \
  -H "Content-Type: application/json" \
  -d '{"branch_name": "Conut - Tyre"}'
```

## 🔬 Technical Approach

### Objective 1: Combo Optimization
- **Method:** NetworkX weighted co-purchase graph + Louvain community detection
- **Metric:** Attach Rate Probability = edge_weight / max(degree(a), degree(b))
- **XAI:** Community membership, co-purchase count, cross-community upsell identification

### Objective 3: Expansion Feasibility
- **Method:** Cosine similarity of branch feature vectors (product mix ratios, item diversity)
- **Reference:** Highest-revenue branch profiled as "gold standard"
- **Output:** Similarity score with gap analysis and actionable recommendations

### Objective 5: Coffee & Milkshake Growth
- **Method:** Percentile ranking of coffee/shake revenue ratios across all branches
- **Output:** Bottom-quartile identification with severity-graded interventions
- **Cross-reference:** Combo graph data for upsell opportunities

## 📁 Project Structure

```
Project C.O.C.O/
├── app/
│   ├── main.py              # FastAPI application
│   └── schemas.py           # Pydantic models
├── models/
│   ├── combo_optimizer.py   # Objective 1
│   ├── expansion_scorer.py  # Objective 3
│   ├── growth_strategy.py   # Objective 5
│   └── train_all.py         # Training orchestrator
├── pipeline/
│   └── clean_data.py        # Data cleaning pipeline
├── cleaned/                  # Cleaned parquet outputs
├── Conut bakery Scaled Data/ # Raw CSV data
├── Makefile
├── requirements.txt
└── README.md
```

## 👥 Team

- **Systems & Analytics Lead (Hafez):** Infrastructure, Objectives 1/3/5, FastAPI, OpenClaw
- **Modeling Duo:** Objectives 2/4 (XGBoost demand forecasting, staffing estimation)

## 📊 Key Findings

*(To be completed after full pipeline run)*

---

**Course:** AI Engineering — American University of Beirut  
**Professor:** Ammar Mohanna
