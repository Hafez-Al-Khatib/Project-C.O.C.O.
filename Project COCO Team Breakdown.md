# Project C.O.C.O. - 12-Hour Sprint

# Division of Labor

**Rule #1:** Work in parallel. Do not block each other.
**Rule #2:** Agree on the "Data Contract" immediately.
**Rule #3:** Push to GitHub frequently.

## 👥 The Operations Modeling Duo

**Focus:** Time-series forecasting and labor operations (Objectives 2 & 4).
Since two students are tackling this, you can divide the data cleaning and modeling perfectly in
half to guarantee a high-quality XGBoost model.
● **Hours 1-3 (Data Janitors):** * **Student A:** Clean rep_s_00334_1_SMRY.csv (Monthly Sales).
Strip out the "Page 1 of 2" headers and aggregate into a clean time-series DataFrame.
○ **Student B:** Clean REP_S_00461.csv (Time & Attendance). Convert punch-in/out times
to total labor hours per shift.
● **Hours 4-7 (Modeling & Math):**
○ **Student A:** Train the XGBoost model on the sales data to forecast future demand.
**Crucial:** Extract model.feature_importances_ and save them alongside predictions for
the XAI layer.
○ **Student B:** Create the mathematical logic that divides Predicted Demand by Historical
Labor Hours to output the Staffing Requirement.
● **Hours 8-10 (Packaging):**
○ Combine your work into two clean functions: predict_demand(branch_name, date)
and estimate_staffing(branch_name, date).
○ Hand these functions (and the saved .pkl model) to the Systems Lead.
● **Hours 11-12:** Write the Executive Brief PDF (Problem framing, top findings, impact).

## 󰭉 The Systems & Analytics Lead

**Focus:** Complex relationships, API infrastructure, LLM Prompt Engineering, and OpenClaw
(Objectives 1, 3, 5).
You are taking on a heavy, highly-integrated workload. Lean on your AI agent heavily for
scaffolding, boilerplate code, and API routing.
● **Hours 1-3 (Infrastructure & Graph Data):**
○ Initialize the GitHub repo, set up the Makefile, and scaffold a basic FastAPI application.
Ensure it runs locally on port 8000.
○ Tackle REP_S_00502.csv (Sales by customer/receipt). Group items by Receipt Number


```
to create a list of co-purchased items for your graph.
● Hours 4-7 (Graph Modeling & Agentic Strategy):
○ Build the NetworkX graph and run Louvain clustering to find Combo Communities
(Objective 1).
○ Mock up the Spatial Signature scoring function using cosine similarity for Conut Jnah
(Objective 3).
○ Parse rep_s_00191_SMRY.csv to identify branches struggling with coffee/milkshakes,
and draft the LLM System Prompt for marketing interventions (Objective 5).
● Hours 8-10 (The OpenClaw Convergence):
○ Wrap your own graph/spatial functions into FastAPI endpoints.
○ Receive the .pkl files and functions from the Modeling Duo and wrap them into
endpoints (e.g., @app.post("/tools/predict_demand")).
○ Construct the OpenClaw /skills JSON manifest.
○ Crucial: Implement Error Fallbacks (e.g., try/except blocks returning default values) so
the API never crashes OpenClaw.
● Hours 11-12: Connect OpenClaw to the local server. Test the chat interface rigorously.
Record the Demo Video of OpenClaw successfully invoking the tools.
```
## 🤝 The Data Contract (To be agreed upon at Hour 0)

To ensure you can build the API without waiting for the Duo to finish their models, agree to
these mock outputs immediately:
**The Modeling Duo will eventually provide a function that returns:**
{
"branch": "Conut Jnah",
"predicted_volume": 1250,
"recommended_staff": 6,
"xai_drivers": {"weekend": "45%", "prev_day_sales": "30%"}
}
**Your own Graph functions will eventually return:**
{
"target_item": "Latte",
"recommended_combo": "Croissant",
"confidence_weight": "28%",
"business_reason": "High co-occurrence in Morning cluster"
}
You can hardcode these JSON responses into the FastAPI routes during Hours 1-7 to test


OpenClaw. When Hours 8-10 arrive, simply replace the hardcoded JSON with the actual
function calls from your team.


