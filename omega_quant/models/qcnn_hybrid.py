from __future__ import annotations

import logging
from typing import Optional

import torch  # type: ignore
import torch.nn as nn  # type: ignore

LOGGER = logging.getLogger(__name__)

try:
    import pennylane as qml  # type: ignore[import-not-found]
except ImportError:
    qml = None


class QuantumConvLayer(nn.Module):
    """Class configuration auto-docstring."""
    def __init__(self, input_dim: int, num_qubits: int, q_layers: int):
        """Auto-docstring."""
        super().__init__()
        self.num_qubits = num_qubits
        self.input_proj = nn.Linear(input_dim, num_qubits)

        if qml is not None:
            qml_mod = qml
            dev = qml_mod.device("default.qubit", wires=num_qubits)

            @qml_mod.qnode(dev, interface="torch")
            def qnode(inputs, weights):
                """Auto-docstring."""
                qml_mod.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
                qml_mod.StronglyEntanglingLayers(weights, wires=range(num_qubits))
                return [qml_mod.expval(qml_mod.PauliZ(i)) for i in range(num_qubits)]

            weight_shapes = {"weights": (q_layers, num_qubits, 3)}
            self.quantum = qml_mod.qnn.TorchLayer(qnode, weight_shapes)
        else:
            LOGGER.warning("PennyLane not available. Falling back to classical projection layer.")
            self.quantum = nn.Sequential(
                nn.Linear(num_qubits, num_qubits),
                nn.Tanh(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Auto-docstring."""
        projected = self.input_proj(x)
        return self.quantum(projected)


class QCNNCNNLSTM(nn.Module):
    """Class configuration auto-docstring."""
    def __init__(
        self,
        input_dim: int,
        num_qubits: int = 4,
        q_layers: int = 2,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.quantum_layer = QuantumConvLayer(input_dim=input_dim, num_qubits=num_qubits, q_layers=q_layers)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + num_qubits, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Auto-docstring."""
        # x shape: [batch, sequence, features]
        conv_in = x.transpose(1, 2)
        conv_out = self.conv(conv_in).transpose(1, 2)

        lstm_out, _ = self.lstm(conv_out)
        temporal_repr = lstm_out[:, -1, :]

        quantum_in = x[:, -1, :]
        q_repr = self.quantum_layer(quantum_in)

        fused = torch.cat([temporal_repr, q_repr], dim=1)
        logits = self.head(fused)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Auto-docstring."""
        self.eval()
        if device is not None:
            x = x.to(device)
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
