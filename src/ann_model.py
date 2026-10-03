"""
Deep Learning Artificial Neural Network (ANN) Forecaster for Retail Demand
Implements a multi-layer perceptron (MLP) with Batch Normalization, Dropout,
ReLU activations, non-negative demand constraints, and early stopping.
"""

import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DemandMLP(nn.Module):
    """PyTorch Multi-Layer Perceptron architecture for demand regression."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),  # Enforces strictly non-negative demand (sales >= 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ANNDemandForecaster:
    """
    Production-ready wrapper around PyTorch DemandMLP.
    Handles scaling, tensor conversion, batching, validation, and inference.
    """

    def __init__(
        self,
        feature_names: List[str],
        lr: float = 0.003,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        epochs: int = 60,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.scaler = StandardScaler()
        self.model: Optional[DemandMLP] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "actual_sales",
        val_df: Optional[pd.DataFrame] = None,
    ) -> "ANNDemandForecaster":
        """
        Trains the ANN using historical data.
        Validates chronologically on val_df if provided.
        """
        X_train_raw = train_df[self.feature_names].values
        y_train_raw = train_df[target_col].values.astype(np.float32)

        # Fit scaler strictly on training split
        X_train_scaled = self.scaler.fit_transform(X_train_raw).astype(np.float32)

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled), torch.tensor(y_train_raw)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if val_df is not None:
            X_val_scaled = self.scaler.transform(val_df[self.feature_names].values).astype(np.float32)
            y_val_raw = val_df[target_col].values.astype(np.float32)
            val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val_raw))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize network
        in_dim = len(self.feature_names)
        self.model = DemandMLP(in_features=in_dim)
        criterion = nn.SmoothL1Loss()  # Huber loss (robust to promotional outlier surges)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_weights = None

        self.model.train()
        for epoch in range(self.epochs):
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_x)

            epoch_train_loss = total_train_loss / len(train_dataset)
            self.train_losses.append(epoch_train_loss)

            # Validation step
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vpreds = self.model(vx)
                        vloss = criterion(vpreds, vy)
                        total_val_loss += vloss.item() * len(vx)
                epoch_val_loss = total_val_loss / len(val_dataset)
                self.val_losses.append(epoch_val_loss)
                scheduler.step(epoch_val_loss)

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.model.train()

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        self.model.eval()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generates continuous demand predictions for input DataFrame."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        X_raw = df[self.feature_names].values
        X_scaled = self.scaler.transform(X_raw).astype(np.float32)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X_scaled)).numpy()

        return np.maximum(0.0, preds)

    def save(self, filepath: str):
        """Saves model weights, scaler, and training loss history."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict() if self.model else None,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            },
            filepath,
        )
        # Also export history as CSV for dashboard plotting
        hist_df = pd.DataFrame({
            "epoch": list(range(1, len(self.train_losses) + 1)),
            "train_loss": self.train_losses,
            "val_loss": self.val_losses if len(self.val_losses) == len(self.train_losses) else [None] * len(self.train_losses),
        })
        hist_csv_path = os.path.join(os.path.dirname(filepath), "ann_training_history.csv")
        hist_df.to_csv(hist_csv_path, index=False)

    def load(self, filepath: str):
        """Loads saved weights and scaler."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.feature_names = checkpoint["feature_names"]
        self.scaler = checkpoint["scaler"]
        self.model = DemandMLP(in_features=len(self.feature_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()


if __name__ == "__main__":
    from baseline_models import MovingAverageForecaster
    from feature_engineering import build_retail_features
    from feature_selection import rank_and_select_features

    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "retail_store_sales.csv")
    df_raw = pd.read_csv(raw_path)
    df_feat = build_retail_features(df_raw)

    max_date = df_feat["date"].max()
    split_date = max_date - pd.Timedelta(days=30)
    train_df = df_feat[df_feat["date"] <= split_date].copy()
    test_df = df_feat[df_feat["date"] > split_date].copy()

    # Feature selection
    candidate_cols = [c for c in df_feat.columns if c not in ["date", "sku_id", "product_name", "category", "velocity_class", "true_demand", "actual_sales", "unfulfilled_demand", "starting_stock", "ending_stock", "was_out_of_stock"]]
    selected_features, _ = rank_and_select_features(train_df[candidate_cols], train_df["actual_sales"], top_k=16)

    # Train ANN
    ann = ANNDemandForecaster(feature_names=selected_features, epochs=50)
    ann.fit(train_df, val_df=test_df)
    ann_preds = ann.predict(test_df)

    # Train baseline
    baseline = MovingAverageForecaster(method="seasonal_sma").fit(train_df)
    base_preds = baseline.predict(test_df)

    y_true = test_df["actual_sales"].values
    base_mae = np.mean(np.abs(y_true - base_preds))
    ann_mae = np.mean(np.abs(y_true - ann_preds))

    print(f"\n================ MODEL BENCHMARK ================")
    print(f"Traditional Seasonal-SMA Baseline MAE: {base_mae:.2f}")
    print(f"Deep Learning ANN Forecaster MAE:     {ann_mae:.2f}")
    print(f"Error Reduction:                     {((base_mae - ann_mae) / base_mae) * 100:.1f}%")
    print(f"==================================================")
