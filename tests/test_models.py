"""Unit tests for models and non-negative demand constraint."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from baseline_models import MovingAverageForecaster
from ann_model import ANNDemandForecaster


def test_baseline_forecaster():
    """Verify MovingAverageForecaster handles standard lags and fallbacks."""
    df = pd.DataFrame({
        "category": ["Dairy", "Bakery"],
        "actual_sales": [50, 40],
        "rolling_mean_7": [45.0, 38.0],
        "rolling_mean_14": [44.0, 37.0],
        "lag_7": [40, 35],
        "lag_14": [42, 36],
        "lag_21": [41, 34],
        "lag_28": [43, 37],
    })

    model = MovingAverageForecaster(method="sma_7").fit(df)
    preds = model.predict(df)
    assert len(preds) == 2
    assert (preds >= 0).all()
    assert np.isclose(preds[0], 45.0)


def test_ann_non_negative_predictions():
    """Verify ANN always outputs non-negative predictions."""
    features = ["feat_1", "feat_2"]
    ann = ANNDemandForecaster(feature_names=features, epochs=2, batch_size=4)

    dummy_train = pd.DataFrame({
        "feat_1": [1.0, 2.0, 3.0, 4.0],
        "feat_2": [0.5, 1.5, 2.5, 3.5],
        "actual_sales": [10.0, 20.0, 30.0, 40.0],
    })
    ann.fit(dummy_train)
    preds = ann.predict(dummy_train)
    assert len(preds) == 4
    assert (preds >= 0.0).all()
