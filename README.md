# Retail Inventory Reorder Assistant (JIG26_06)
**Track A — ANN & Predictive Analytics | Team: TEAM-28**  
*Domain: Supply Chain & Retail Intelligence*

An end-to-end AI-powered inventory replenishment system that forecasts customer demand using Deep Learning (ANN), dramatically outperforms traditional moving average baselines, and translates predictions into risk-aware reorder suggestions across fast, promo-sensitive, slow-moving, and new cold-start items.

---

## 🏆 Key Achievements & Benchmarks
Tested on an untouched 30-day chronological backtest split (no future lookahead bias):

| Model | MAE | RMSE | WAPE (%) | Stock-Out Loss (SOEL) | Performance vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Learning ANN Forecaster** | **4.30** | **6.67** | **9.33%** | **9.59** | **58.6% error reduction** |
| Baseline Seasonal-SMA | 9.94 | 15.67 | 21.56% | 24.03 | Benchmark |
| Baseline SMA-14 | 10.34 | 15.72 | 22.43% | 22.64 | Benchmark |
| Baseline SMA-7 | 10.38 | 15.98 | 22.51% | 22.82 | Benchmark |

> **Stock-Out-Oriented Error Loss (SOEL)** penalizes under-forecasting **3.5× more** than over-forecasting, reflecting real-world retail economics where running out of stock destroys revenue and customer loyalty. The ANN reduces stock-out loss by **60.1%**!

---

## 🛠️ Architecture & Pipeline

```
JIG26/
├── data/
│   ├── retail_store_sales.csv         # 25 SKUs over 730 days (fast, promo, slow, cold-start)
│   ├── retail_engineered_features.csv # 37 engineered features across 6 pillars
│   ├── feature_rankings.csv           # MI & Random Forest composite attribution scores
│   ├── model_scorecard.csv            # Chronological evaluation metrics
│   ├── ann_model.pt                   # Trained PyTorch ANN model checkpoint
│   └── latest_reorder_manifest.csv    # Current SKU reorder recommendations
├── src/
│   ├── data_generator.py              # Retail simulator with price elasticity & inventory balance
│   ├── feature_engineering.py         # 37 zero-leakage features
│   ├── feature_selection.py           # Collinearity filter + Mutual Info + Tree attribution
│   ├── baseline_models.py             # SMA-7, SMA-14, Seasonal-SMA with cold-start fallback
│   ├── ann_model.py                   # PyTorch MLP with BatchNorm, Dropout, Huber loss
│   ├── evaluation.py                  # Chronological backtesting & asymmetric loss (SOEL)
│   └── reorder_engine.py              # Dynamic SS, ROP, Croston-Poisson, Cold-start heuristics
├── tests/
│   ├── test_features.py               # Zero lookahead leakage & finite data assertions
│   ├── test_models.py                 # Predictor shape & non-negative demand tests
│   └── test_reorder.py                # Reorder triggers & edge-case rule tests
├── app.py                             # Interactive Streamlit Decision-Support Dashboard
├── run_pipeline.py                    # End-to-end headless pipeline runner
└── requirements.txt                   # Dependencies
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Executes data generation, feature engineering, feature selection, model training, and reorder manifest generation:
```bash
python run_pipeline.py
```

### 3. Run Automated Tests
```bash
pytest tests/
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```

---

## 🧠 Feature Selection Methodology
Raw retail POS data lacks model-ready signals. We engineer 37 candidate features across **6 retail pillars** and filter them through a 3-stage selection pipeline:
1. **Multicollinearity Filter**: Drops pairwise $|r| > 0.88$ features to prevent neural weight oscillation.
2. **Mutual Information Regression**: Extracts non-linear dependencies with customer demand.
3. **Random Forest Gini Attribution**: Measures decision tree split importance.
4. **Curated Top Features**:
   - `rolling_mean_28` & `lag_28`: Underlying baseline velocity.
   - `discount_percent` & `consecutive_promo_days`: Non-linear promotional elasticity & promo fatigue.
   - `rolling_std_7`: Demand volatility informing dynamic safety stock.
   - `is_payday_window`: Salary surge on days 1–5 and 28–31.
   - `stock_to_sales_ratio_lag1`: Distinguishes zero sales from empty shelves (censored demand).

---

## 📦 Reorder Engine Policies
1. **Fast & Medium Velocity**: Dynamic Continuous $(s, S)$ policy with Safety Stock $SS = Z \times \sigma \times \sqrt{L}$.
2. **Slow / Intermittent Demand**: Poisson/Croston min-max threshold capping safety buffers at 1–2 units to prevent frozen capital.
3. **Cold-Start New Items**: Category-analog velocity heuristic with a 1.5× safety stock multiplier until 14 days of history accumulate.
4. **Perishability Constraints**: Shelf-life upper bound capping order sizes on short-shelf-life goods (milk, bread).
