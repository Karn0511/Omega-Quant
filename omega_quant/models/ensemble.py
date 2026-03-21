# pylint: disable=import-error
import torch  # type: ignore
import torch.nn as nn  # type: ignore
# pylint: enable=import-error
from typing import Optional

from omega_quant.models.qcnn_hybrid import QCNNCNNLSTM

class TimeSeriesLSTM(nn.Module):
    """Class configuration auto-docstring."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        """Auto-docstring."""
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Auto-docstring."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class SequenceTransformer(nn.Module):
    """Class configuration auto-docstring."""
    def __init__(self, input_dim: int, num_heads: int = 4, hidden_dim: int = 64, num_classes: int = 3):
        """Auto-docstring."""
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Auto-docstring."""
        x = self.embedding(x)
        out = self.transformer(x)
        return self.fc(out.mean(dim=1))

class DeepEnsemble(nn.Module):
    """Class configuration auto-docstring."""
    def __init__(self, input_dim: int, config: dict):
        """Auto-docstring."""
        super().__init__()
        mcfg = config["model"]
        self.qcnn = QCNNCNNLSTM(
            input_dim=input_dim,
            num_qubits=mcfg.get("num_qubits", 4),
            q_layers=mcfg.get("q_layers", 2),
            conv_channels=mcfg.get("conv_channels", 32),
            lstm_hidden=mcfg.get("lstm_hidden", 64),
            dropout=mcfg.get("dropout", 0.2),
            num_classes=mcfg.get("num_classes", 3)
        )
        self.lstm = TimeSeriesLSTM(input_dim, hidden_dim=64, num_classes=3)
        self.transformer = SequenceTransformer(
            input_dim,
            num_heads=mcfg.get("num_heads", 4),
            hidden_dim=128,
            num_classes=mcfg.get("num_classes", 3)
        )

        # Confidence calibration (Softmax temperature scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Auto-docstring."""
        qcnn_out = self.qcnn(x)
        lstm_out = self.lstm(x)
        trans_out = self.transformer(x)

        # Simple average of logits
        ensemble_logits = (qcnn_out + lstm_out + trans_out) / 3.0

        # Scale by learned temperature for confidence calibration
        return ensemble_logits / self.temperature

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Auto-docstring."""
        self.eval()
        if device is not None:
            x = x.to(device)
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
