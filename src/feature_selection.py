"""
Feature Selection Engine for Retail Demand Forecasting
Performs rigorous feature selection to eliminate noise and multicollinearity:
1. Pairwise Correlation & Multicollinearity Filtering (removes |r| > 0.88 redundancy)
2. Mutual Information Regression (non-linear statistical dependency)
3. Tree-Based Importance (Random Forest feature attribution)
4. Composite Ranking & Selection of the Top-K Predictive Features
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


def remove_collinear_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.88,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Identifies pairs of features with correlation > threshold.
    For each collinear pair, retains the feature with higher correlation to target y.
    Returns the list of retained feature names and a dictionary of dropped features with reasons.
    """
    corr_matrix = X.corr().abs().fillna(0.0)
    y_corrs = X.corrwith(y).abs().fillna(0.0)

    dropped: Dict[str, str] = {}
    retained: List[str] = list(X.columns)

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_a = corr_matrix.columns[i]
            col_b = corr_matrix.columns[j]

            if col_a in retained and col_b in retained:
                corr_val = corr_matrix.loc[col_a, col_b]
                if corr_val > threshold:
                    # Drop the one with lower correlation to y
                    if y_corrs[col_a] >= y_corrs[col_b]:
                        retained.remove(col_b)
                        dropped[col_b] = f"Collinear with {col_a} (r={corr_val:.2f})"
                    else:
                        retained.remove(col_a)
                        dropped[col_a] = f"Collinear with {col_b} (r={corr_val:.2f})"

    return retained, dropped


def rank_and_select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 16,
    collinear_threshold: float = 0.88,
    random_state: int = 42,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Ranks candidate features using a dual-attribution pipeline:
    1. Filter multicollinear features.
    2. Mutual Information Regression (non-linear statistical dependencies).
    3. Random Forest Gini Importance (decision tree split attribution).
    4. Composite Score = 0.5 * Norm(MI) + 0.5 * Norm(RF).
    Returns the selected top_k feature names and the full feature importance ranking DataFrame.
    """
    # Sanitize input features
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Step 1: Remove collinear features
    retained_cols, dropped_info = remove_collinear_features(
        X_train_clean, y_train, threshold=collinear_threshold
    )

    X_filtered = X_train_clean[retained_cols].copy()

    # Step 2: Mutual Information
    print("Computing Mutual Information scores...")
    mi_scores = mutual_info_regression(X_filtered, y_train, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=retained_cols)

    # Step 3: Random Forest Feature Importance
    print("Fitting Random Forest feature ranker...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1)
    rf.fit(X_filtered, y_train)
    rf_series = pd.Series(rf.feature_importances_, index=retained_cols)

    # Step 4: Normalize and compute Composite Score
    mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-6)
    rf_norm = (rf_series - rf_series.min()) / (rf_series.max() - rf_series.min() + 1e-6)

    composite_score = 0.5 * mi_norm + 0.5 * rf_norm

    ranking_df = pd.DataFrame({
        "feature": retained_cols,
        "composite_score": composite_score,
        "mutual_info": mi_series,
        "rf_importance": rf_series,
    }).sort_values(by="composite_score", ascending=False).reset_index(drop=True)

    selected_features = ranking_df.head(top_k)["feature"].tolist()

    # Ensure critical retail anchors are present (e.g. promo, is_weekend, payday)
    must_have_anchors = ["is_promo", "is_weekend", "is_payday_window", "lag_1", "rolling_mean_7"]
    for anchor in must_have_anchors:
        if anchor in retained_cols and anchor not in selected_features:
            # Replace the lowest ranked feature with the anchor
            selected_features[-1] = anchor

    return selected_features, ranking_df


if __name__ == "__main__":
    from feature_engineering import build_retail_features, get_candidate_feature_names

    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "retail_store_sales.csv")
    df_raw = pd.read_csv(raw_path)
    df_feat = build_retail_features(df_raw)

    # Chronological train split (all data except last 30 days)
    max_date = df_feat["date"].max()
    split_date = max_date - pd.Timedelta(days=30)
    train_df = df_feat[df_feat["date"] <= split_date].copy()

    candidate_cols = get_candidate_feature_names()
    X_train = train_df[candidate_cols]
    y_train = train_df["actual_sales"]

    selected_features, ranking_df = rank_and_select_features(X_train, y_train, top_k=16)

    print("\n--- Top Selected Features for ANN Demand Model ---")
    print(ranking_df.head(16)[["feature", "composite_score", "mutual_info", "rf_importance"]])
    print(f"\nFinal Selected Count: {len(selected_features)}")
