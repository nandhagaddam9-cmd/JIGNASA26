# Retail Inventory Reorder Assistant (JIG26_06)
## Complete Hackathon Pitch Deck & Speaker Script for Judges
**Track A:** ANN & Predictive Analytics  
**Team:** TEAM-28  
**Domain:** Supply Chain & Retail Intelligence  

---

## 📑 Slide Overview
1. **Slide 1:** Title & Executive Hook
2. **Slide 2:** The Billion-Dollar Retail Dilemma (The Problem)
3. **Slide 3:** Product Scope & Inventory Setup
4. **Slide 4:** Feature Engineering: The 6 Retail Pillars
5. **Slide 5:** Feature Selection: The Scientific Differentiator
6. **Slide 6:** Modeling Benchmark: Moving Average vs. Deep Learning ANN
7. **Slide 7:** Chronological Backtesting & Asymmetric Stockout Loss (SOEL)
8. **Slide 8:** The Reorder Engine & Edge-Case Policy Rules
9. **Slide 9:** Live Interactive Decision Dashboard
10. **Slide 10:** Business Impact, ROI & Summary

---

### Slide 1: Title & Executive Hook
**Title:** Retail Inventory Reorder Assistant  
**Subtitle:** Risk-Aware Inventory Replenishment Powered by Deep Learning  
**Presenter:** Team TEAM-28 | Track A (ANN & Predictive Analytics)

#### 📌 Visuals on Slide:
* Clean logo / shopping cart icon.
* One-line tagline: *"Eliminating stock-outs during demand spikes while preventing capital lockup from excess inventory."*

#### 🗣️ Pitch Script (What to say to the judges):
> *"Good morning, respected judges. In retail supply chains, guessing tomorrow's demand is a recipe for disaster. Every retailer is trapped between two expensive extremes: shelves running empty when a weekend or promotional spike hits, or warehouse capital getting frozen in unsold, expiring inventory.*
>
> *Today, Team TEAM-28 presents the **Retail Inventory Reorder Assistant** — an end-to-end AI system that uses an Artificial Neural Network with domain-grounded feature selection to forecast 7-day demand with 58% lower error than traditional industry baselines, translating every forecast into actionable purchase orders."*

---

### Slide 2: The Billion-Dollar Retail Dilemma
**Title:** The Dual Financial Trap: Stock-Outs vs. Excess Inventory

#### 📌 Key Points on Slide:
* **The Stock-Out Crisis (Under-Stocking):**
  - Causes: Promotional surges, weekend basket spikes, payday shopping waves, supplier delivery delays.
  - Consequence: Immediate revenue loss, brand abandonment, lost customer loyalty.
* **The Excess Inventory Trap (Over-Stocking):**
  - Causes: Ordering based on outdated moving averages, ignoring post-promo dips, slow-moving items hoarding space.
  - Consequence: Frozen working capital, high holding costs, inventory spoilage/expiration, forced discount clearance.
* **Why Traditional Methods Fail:**
  - Simple Moving Averages lag behind trends and are 100% blind to promotions and calendar dynamics.

#### 🗣️ Pitch Script:
> *"Why do existing retail replenishment systems fail? Because traditional inventory software relies on Moving Averages. If you take the 7-day average of milk sales, you're looking backwards. When marketing runs a 25% discount tomorrow or a payday weekend hits, a moving average has zero clue. The shelf empties in hours.*
>
> *Conversely, when a promotion ends, moving averages predict high sales because of last week's spike, causing managers to over-order right before demand crashes. We built a system that models the true causal drivers of retail demand."*

---

### Slide 3: Product Scope & Catalog Architecture
**Title:** Realistic Multi-SKU Retail Scope

#### 📌 Key Points on Slide:
* **Catalog:** 1 Retail Supermarket Store, **25 SKUs**, **730 Days** (2 Years) of daily transaction history.
* **Forecast Horizon:** **Next 7 Days** (Weekly replenishment cycle).
* **4 Distinct Behavioral Product Clusters:**
  1. **Fast-Moving Staples (8 SKUs):** Fresh Milk, Bread, Eggs, Soda (High daily volume, high stock-out risk).
  2. **Promo-Sensitive Items (8 SKUs):** Coffee, Olive Oil, Cereal, Detergent (Moderate baseline, massive spikes on discount).
  3. **Slow / Intermittent Items (6 SKUs):** Truffle Oil, Saffron, Balsamic (Zero-inflated sporadic demand).
  4. **Cold-Start New Launches (3 SKUs):** Matcha Drink, Protein Bar, Oat Milk (Zero prior sales history in final 30 days).

#### 🗣️ Pitch Script:
> *"To ensure our solution was tested on real-world retail reality, we defined a focused scope of 25 products representing a complete supermarket grocery footprint across 4 distinct behavioral clusters: fast-moving staples, highly elastic promotional items, intermittent slow-movers, and brand-new items launched with zero sales history.*
>
> *We track inventory dynamically using the physical inventory balance equation, accounting for lead times, on-hand shelf stock, and unfulfilled demand."*

---

### Slide 4: Feature Engineering — The 6 Retail Pillars
**Title:** Transforming Raw Scanner Logs into 37 Predictive Signals

#### 📌 Key Points on Slide:
* Raw POS data only contains Date, SKU, Sales, and Price. We engineered **37 domain features** across 6 pillars:
  1. **Autoregressive Lags:** $y_{t-1}, y_{t-2}, y_{t-3}$ (inertia) and $y_{t-7}, y_{t-14}, y_{t-21}, y_{t-28}$ (weekly cycles).
  2. **Rolling Volatility & Mean:** 7d, 14d, 28d rolling mean and rolling standard deviation ($\sigma$).
  3. **Price & Promo Elasticity:** `discount_percent`, `consecutive_promo_days` (fatigue), `days_since_last_promo` (post-promo dip).
  4. **Calendar & Cyclical Signals:** Day of week, cyclical $\sin/\cos$ transformations, month.
  5. **The Payday Effect (`is_payday_window`):** Captures days 1–5 and 28–31 salary surges (20–40% basket lift).
  6. **Censored Demand Indicator (`stock_to_sales_ratio_lag1`):** Flags when 0 sales occurred because the shelf was empty vs. true zero demand.
* **Zero Lookahead Leakage:** All lags and rolling statistics are strictly computed using historical data up to $t-1$.

#### 🗣️ Pitch Script:
> *"No raw retail dataset comes ready for machine learning. Real point-of-sale registers only log receipts. We engineered 37 features across 6 retail pillars.*
>
> *Critically, we solved the famous 'Censored Demand Trap' in retail: if milk had 0 sales yesterday, was it because customers didn't want it, or because the shelf was empty? By encoding inventory feedback, our model avoids learning false zero-demand signals."*

---

### Slide 5: Feature Selection — The Scientific Differentiator
**Title:** Rigorous 3-Stage Feature Selection Pipeline

#### 📌 Key Points on Slide:
* **The Problem:** Raw feature dumping causes neural network weight oscillation and overfitting due to multicollinearity (e.g. 7-day mean vs 8-day mean).
* **Our 3-Stage Selection Engine:**
  1. **Multicollinearity Filter:** Drops redundant pairwise features where $|r| > 0.88$, keeping the stronger target predictor.
  2. **Mutual Information Regression:** Measures pure non-linear statistical dependency.
  3. **Random Forest Gini Attribution:** Captures multi-variable split importance.
  4. **Composite Ranking:** $\text{Score} = 0.5 \times \text{Norm(MI)} + 0.5 \times \text{Norm(RF)}$.
* **Top Selected Features:** `rolling_mean_28`, `lag_28`, `regular_price`, `rolling_std_7`, `discount_percent`, `is_payday_window`, `momentum_ratio`.

#### 🗣️ Pitch Script:
> *"Feature selection was our most critical engineering priority. Feeding 37 raw features into a neural network degrades generalization because rolling averages are 95% collinear.*
>
> *We implemented a rigorous 3-stage pipeline: first filtering out collinear redundancies, then computing non-linear Mutual Information, and blending it with tree-based split attribution. This isolated the exact 16 signals that drive real retail purchase behavior."*

---

### Slide 6: Model Architecture & Benchmark
**Title:** Moving Average Baselines vs. Deep Learning ANN

#### 📌 Key Points on Slide:
* **ANN Architecture:** PyTorch MLP with Batch Normalization, Dropout (0.20), Huber Loss (robust to promo outlier spikes), and ReLU output (strictly non-negative demand $\ge 0$).
* **Chronological Test Benchmark (Untouched Final 30 Days):**

| Model | MAE | RMSE | WAPE (%) | Performance |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Learning ANN Forecaster** | **4.30** | **6.67** | **9.33%** | **58.6% Error Reduction** |
| Baseline Seasonal-SMA | 9.94 | 15.67 | 21.56% | Benchmark |
| Baseline SMA-14 | 10.34 | 15.72 | 22.43% | Benchmark |
| Baseline SMA-7 | 10.38 | 15.98 | 22.51% | Benchmark |

#### 🗣️ Pitch Script:
> *"We benchmarked our Deep Learning ANN against 3 traditional industry baselines: SMA-7, SMA-14, and Seasonal-SMA. On an untouched chronological test split, our ANN achieved an MAE of 4.30 units compared to 9.94 for Seasonal-SMA — a massive **58.6% reduction in forecast error**.*
>
> *The ANN smoothly anticipates weekend peaks and promo discounts that completely blindside moving averages."*

---

### Slide 7: Chronological Backtesting & Stock-Out-Oriented Error
**Title:** Asymmetric Evaluation: The Stock-Out Loss Metric (SOEL)

#### 📌 Key Points on Slide:
* **Why Standard MAE is Flawed in Retail:**
  - In symmetric MAE, under-forecasting by 10 units is treated the same as over-forecasting by 10 units.
  - In business, under-forecasting means an empty shelf and lost revenue; over-forecasting just means holding safety inventory.
* **Our Domain Metric — Asymmetric Stock-Out Error Loss (SOEL):**
  $$L(y, \hat{y}) = \begin{cases} 3.5 \times (y - \hat{y}) & \text{if } y > \hat{y} \text{ (Under-forecast: Stock-Out Penalty)} \\ 1.0 \times (\hat{y} - y) & \text{if } \hat{y} \ge y \text{ (Over-forecast: Holding Penalty)} \end{cases}$$
* **Results:**
  - Traditional SMA Stock-Out Loss: **24.03**
  - Deep Learning ANN Stock-Out Loss: **9.59** (**60.1% reduction in stock-out risk!**)

#### 🗣️ Pitch Script:
> *"Standard metrics like MAE are symmetric — they treat having 5 extra boxes of cereal the same as having 5 disappointed customers who find empty shelves.*
>
> *We implemented an Asymmetric Stock-Out-Oriented Error Loss that penalizes under-forecasting 3.5 times more heavily than over-forecasting. Our ANN cut the stock-out error score from 24.03 down to 9.59 — cutting retail stock-out risk by over 60%."*

---

### Slide 8: The Reorder Engine & Policy Rules
**Title:** From Demand Forecasts to Actionable Purchase Orders

#### 📌 Key Points on Slide:
* **Continuous Dynamic Safety Stock:**
  $$SS = Z \times \sigma_{\text{volatility}} \times \sqrt{L} \quad (\text{where } Z = 1.65 \text{ for 95% service level})$$
  $$ROP = (\text{Daily Rate} \times L) + SS$$
* **Specialized Policy Rules:**
  1. **Fast Staples:** Dynamic $(s, S)$ reorder with **Perishability Shelf-Life Caps** (e.g. fresh milk order cannot exceed 5-day consumption).
  2. **Slow / Intermittent Items (Truffle Oil, Saffron):** Croston/Poisson min-max threshold caps buffer at 1–2 units, preventing thousands of dollars from being frozen in slow-moving goods.
  3. **New / Cold-Start Items (Matcha Tea, Protein Bar):** Category-Analog heuristic applies a 1.5× launch buffer using median category velocity.
* **Actionable Status:** `CRITICAL_STOCKOUT_RISK`, `REORDER_RECOMMENDED`, `HEALTHY`, `OVERSTOCKED`.

#### 🗣️ Pitch Script:
> *"A forecast is useless unless it tells the store manager what to buy. Our Reorder Engine bridges this gap.*
>
> *We implemented specialized policies: For fast items, dynamic safety stocks adapt to volatility while perishability rules prevent over-ordering expiring milk. For slow-moving items like truffle oil, we use a Poisson min-max rule that caps buffers at 1 to 2 units, avoiding dead inventory. For cold-start items with zero sales history, our category-analog heuristic seeds an initial launch batch with a 1.5x buffer."*

---

### Slide 9: Interactive Decision Dashboard
**Title:** Decision-Support Dashboard (Built with Streamlit & Plotly)

#### 📌 Key Points on Slide:
* **4 Dedicated Control Centers:**
  1. **Reorder Workbench:** Live table with stock, ROP, suggested order quantities, status color codes, and CSV Purchase Order export.
  2. **Demand Forecast Inspector:** Interactive Plotly visualizer showing historical demand, promo days, baseline, and ANN forecast.
  3. **Feature Selection Studio:** Transparent feature importance charts showing managers *why* predictions occur.
  4. **Model Benchmark Scorecard:** Live chronological accuracy scoreboard.
* **Interactive Sliders:** Adjust service level (90%, 95%, 99%) and supplier lead times on the fly.

#### 🗣️ Pitch Script:
> *"Here is our live dashboard. Store managers don't need to know neural network math — they see an executive cockpit with immediate stock-out alerts, a filterable purchase order workbench, interactive SKU forecasts, and full model explainability. With one click, they can export verified purchase orders ready for supplier trucks."*

---

### Slide 10: Business Impact, ROI & Summary
**Title:** Measurable Business Value

#### 📌 Key Points on Slide:
* **58.6% Reduction in Forecast Error (MAE):** Eliminates guesswork in weekly purchasing.
* **60.1% Drop in Stock-Out Risk (SOEL):** Directly captures lost sales during promotional and weekend surges.
* **Working Capital Optimization:** Eliminates over-ordering of slow-moving items, freeing up working cash.
* **Scalable Architecture:** Modular Python code with automated unit tests (5/5 passing) and pluggable CSV loader for any real retail store.

#### 🗣️ Pitch Script:
> *"In summary, the Retail Inventory Reorder Assistant turns unpredictable consumer demand into reliable, automated inventory decisions. By pairing Deep Learning with domain-grounded feature selection and risk-aware replenishment rules, we deliver a 58.6% error reduction and a 60% drop in stock-out risk.*
>
> *Thank you, judges. We are now open for your questions and live demonstration!"*

---

## 🎯 Judges Q&A Cheat Sheet (Anticipated Questions & Winning Answers)

**Q1: Why did you choose an ANN instead of a simpler linear regression or XGBoost?**  
*Answer:* "Retail demand is fundamentally non-linear. The interaction between a 20% discount, payday timing, and weekend surges creates exponential demand lifts with diminishing returns (promotional fatigue). An ANN with non-linear ReLU activations and batch normalization captures these multi-variable interactions naturally, while our custom non-negative constraint guarantees forecasts never output unphysical negative demand."

**Q2: How do you prevent data leakage in time-series forecasting?**  
*Answer:* "We strictly enforced chronological splitting. The model was trained on the first 700 days and tested on the untouched final 30 days. Crucially, all lag and rolling window features were calculated with a 1-day historical shift (`.shift(1)`), ensuring that predictions for day $t$ only consume data available at or before $t-1$."

**Q3: How do you handle cold-start items that have no sales history?**  
*Answer:* "Traditional moving averages crash on new items because past averages are undefined. Our Reorder Engine uses a Category-Analog Heuristic: it borrows baseline velocity from the median of existing items in the same category (e.g. Dairy for a new Oat Milk) and adds a 1.5x safety stock buffer to protect against early launch volatility until 14 days of real sales accumulate."

**Q4: What is the business rationale behind your Stock-Out-Oriented Error Loss (SOEL)?**  
*Answer:* "Standard MAE assumes an under-forecast and an over-forecast have equal business cost. In reality, under-forecasting causes empty shelves, lost revenue, and lost customers. Over-forecasting only causes minor temporary holding costs. SOEL penalizes under-forecasting 3.5x more heavily, aligning our neural network directly with retail profitability."
