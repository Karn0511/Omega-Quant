"""Pipeline module for training and evaluating exactly the OMEGA-QUANT Hybrid strategies."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

# pylint: disable=import-error
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
import xgboost as xgb  # type: ignore
# pylint: enable=import-error

from omega_quant.models.dataset import DatasetArtifacts, build_sequences
from omega_quant.models.ensemble import DeepEnsemble
from omega_quant.models.meta_optimizer import MetaOptimizer
from omega_quant.models.trainer import Trainer
from omega_quant.utils.features import build_features, infer_feature_columns, normalize_features

LOGGER = logging.getLogger(__name__)


class UniversalEnsemble:
    """Wrapper class combining PyTorch DeepEnsemble and XGBoost tabular models."""

    def __init__(self, deep_model: DeepEnsemble, xgb_model: xgb.XGBClassifier):
        """Auto-docstring."""
        self.deep_model = deep_model
        self.xgb_model = xgb_model
        self.meta_optimizer = MetaOptimizer()

    def to(self, device: torch.device):
        """Map deep learning elements to designated device."""
        self.deep_model.to(device)
        return self

    def eval(self):
        """Set models to evaluation testing mode."""
        self.deep_model.eval()


def prepare_training_data(
    df: pd.DataFrame, config: Dict, fit_scaler: bool = True
) -> DatasetArtifacts:
    """Parses raw datasets into structured time-series formatted features."""
    fdf = build_features(df)
    feature_columns = infer_feature_columns(fdf)

    scaler_path = config["features"]["scaler_path"]
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)

    ndf = normalize_features(
        fdf,
        feature_columns,
        scaler_path=scaler_path,
        fit=fit_scaler
    )

    seq_len = config["features"]["sequence_length"]
    horizon = config["features"]["prediction_horizon"]
    return build_sequences(ndf, feature_columns, sequence_length=seq_len, horizon=horizon)


def build_model(config: Dict, input_dim: int) -> DeepEnsemble:
    """Instantiate the deep ensemble architecture."""
    return DeepEnsemble(input_dim, config)


def train_model(df: pd.DataFrame, config: Dict) -> Tuple[UniversalEnsemble, DatasetArtifacts]:
    """Execute end-to-end training over deep learning units and tree architectures."""
    artifacts = prepare_training_data(df, config, fit_scaler=True)
    deep_model = build_model(config, input_dim=len(artifacts.feature_columns))

    trainer = Trainer(deep_model, config)
    trainer.fit(artifacts.x, artifacts.y)

    LOGGER.info("Training XGBoost tabular model...")
    x_tabular = artifacts.x[:, -1, :]
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        tree_method="hist",
    )
    if torch.cuda.is_available():
        xgb_model.set_params(device="cuda")

    xgb_model.fit(x_tabular, artifacts.y)

    # Feature Importance Tracking
    importances = xgb_model.feature_importances_
    features = artifacts.feature_columns
    importance_dict = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    LOGGER.info("Top 10 Feature Importances (XGBoost):")
    for feat, imp in importance_dict[:10]:
        LOGGER.info("  %s: %.4f", feat, imp)

    pd.DataFrame(importance_dict, columns=["Feature", "Importance"]).to_csv(
        Path(config["training"]["model_path"]).parent / "feature_importance.csv", index=False
    )

    xgb_path = Path(config["training"]["model_path"]).parent / "xgboost_model.json"
    xgb_model.save_model(xgb_path)

    return UniversalEnsemble(deep_model, xgb_model), artifacts


def load_model(config: Dict, input_dim: int) -> UniversalEnsemble:
    """Restore models from path artifacts."""
    deep_model = build_model(config, input_dim)
    model_path = config["training"]["model_path"]
    state = torch.load(model_path, map_location="cpu")
    deep_model.load_state_dict(state)
    deep_model.eval()

    xgb_path = Path(model_path).parent / "xgboost_model.json"
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)

    return UniversalEnsemble(deep_model, xgb_model)


def predict_probabilities(model: UniversalEnsemble, input_x: np.ndarray) -> np.ndarray:
    """Generate explicitly weighted array outputs mapped against temperature values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Deep representations
    tensor = torch.from_numpy(input_x.astype(np.float32)).to(device)
    with torch.no_grad():
        dm = model.deep_model
        qcnn_out = dm.qcnn(tensor)
        lstm_out = dm.lstm(tensor)
        trans_out = dm.transformer(tensor)

        temp = dm.temperature

        prob_qcnn = torch.softmax(qcnn_out / temp, dim=1).cpu().numpy()
        prob_lstm = torch.softmax(lstm_out / temp, dim=1).cpu().numpy()
        prob_trans = torch.softmax(trans_out / temp, dim=1).cpu().numpy()

    # XGBoost tabular input
    x_tabular = input_x[:, -1, :]
    prob_xgb = model.xgb_model.predict_proba(x_tabular)

    # Phase 5: Dynamic weights guided by MetaOptimizer
    w_qcnn, w_lstm, w_xgb, w_trans = model.meta_optimizer.get_weights()

    final_probs = (w_qcnn * prob_qcnn) + (w_lstm * prob_lstm) + \
                  (w_xgb * prob_xgb) + (w_trans * prob_trans)
    return final_probs
