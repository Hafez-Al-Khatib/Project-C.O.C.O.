# Project C.O.C.O. (Chief of Operations Copilot)

> A high-performance Agentic AI Decision-Support System for bakery operations, built with **LangGraph** and probabilistic ML.

## 🎯 Business Problem

Conut is a growing retail business that requires rapid operational intelligence. Project C.O.C.O. transforms raw POS data into actionable business decisions across five core objectives:

1.  **Combo Optimization** — Graph-based co-purchase analysis (Louvain) to identify high-lift product bundles.
2.  **Demand Forecasting** — V3 Ratio-based **Gaussian Process Regression (GPR)** with rigorous uncertainty quantification.
3.  **Expansion Feasibility** — OSM-integrated similarity scoring against the bakery's "Gold Standard" branch profile.
4.  **Shift Staffing** — Risk-bounded staffing estimation based on 95% upper-bound confidence intervals.
5.  **Coffee & Milkshake Growth** — Strategic gap analysis with targeted, agent-led interventions.

## 🏗️ Architecture: The ReAct Agent

C.O.C.O. is built on a **Reason + Act (ReAct)** framework. The agent autonomously orchestrates multiple ML tools to solve complex, multi-step business queries.

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Raw CSV Data   │ ──▶ │  Data Pipeline   │ ──▶ │  Parquet Files │
│  (POS Exports)  │     │  (clean_data.py) │     │  (cleaned/)   │
└─────────────────┘     └──────────────────┘ └───────┬───────┘
                                                         │
                        ┌────────────────────────────────┘
                        ▼
                ┌───────────────┐      ┌─────────────────────────────┐
                │  Model Layer  │      │      LangGraph ReAct Agent  │
                │  (Prob ML)    │ ◀──▶ │ (Thought -> Action -> Obs)  │
                └───────┬───────┘      └───────────────┬─────────────┘
                        │                              │
                        ▼                              ▼
                ┌───────────────┐              ┌───────────────┐
                │   FastAPI     │ ◀──────────▶ │   Frontend    │
                │  Port 8000    │              │   (Svelte)    │
                └───────────────┘              └───────────────┘
```

## 🚀 Quick Start

```bash
# 1. Setup Environment
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Run Pipeline & Train V3 Models
python pipeline/clean_data.py
python models/demand_forecaster.py  # Trains V3 GPR Model

# 3. Start Intelligence Server
uvicorn app.main:app --reload --port 8000

# 4. Launch Strategic Dashboard
cd frontend && npm run dev
```

## 📡 Agentic API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/openclaw` | POST | **Primary Agent Interface.** Accepts natural language business queries. |
| `/predict_demand` | POST | Probabilistic forecast with 95% confidence bands. |
| `/estimate_staffing`| POST | Throughput-decapsulated labor requirements. |
| `/get_combos` | POST | Market basket analysis & community detection. |
| `/generate_bi_plot`| POST | Dynamic Matplotlib chart generation for executives. |

## 🔬 Probabilistic Intelligence

### Objective 2: V3 Demand Forecasting
*   **Method:** Gaussian Process Regression (GPR) + Bayesian Ridge.
*   **Innovation:** Ratio-based target engineering (Growth Multiplier) to handle volatile holiday spikes.
*   **Risk Management:** Native standard deviation output used to warn executives of "Out-of-Distribution" market events.

### Objectives 1 & 4: Staffing & Bundling
*   **Staffing:** Uses a deterministic "Throughput Physics Engine" derived from December labor benchmarks to calculate required headcount from demand forecasts.
*   **Combos:** NetworkX co-purchase graphs identify item "communities," allowing for intelligent cross-sell recommendations with predicted sales lift.

## 👥 Team: Systems & Analytics Lead

**Lead Developer: Hafez Al Khatib**
*Project C.O.C.O. was developed for the AI Engineering Hackathon at AUB.*

---

**Course:** AI Engineering — American University of Beirut
**Professor:** Ammar Mohanna
**Date:** February 2026
