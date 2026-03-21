from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

# pylint: disable=import-error
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader, random_split  # type: ignore
from tqdm import tqdm
# pylint: enable=import-error

from omega_quant.models.dataset import MarketSequenceDataset

LOGGER = logging.getLogger(__name__)


class Trainer:
    """Class configuration auto-docstring."""
    def __init__(self, model: nn.Module, config: Dict):
        """Auto-docstring."""
        self.model = model
        self.config = config
        train_cfg = config["training"]
        self.initial_batch_size = train_cfg["batch_size"]
        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.train_split = train_cfg["train_split"]

        self.checkpoint_dir = Path(train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = Path(train_cfg["model_path"])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Hardware Acceleration Optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            LOGGER.info("CUDA DETECTED: Engaging CUDNN Hardware Acceleration.")
            
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # Mixed precision training
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def _split_dataset(self, dataset: MarketSequenceDataset) -> Tuple[MarketSequenceDataset, MarketSequenceDataset]:
        """Auto-docstring."""
        train_size = int(len(dataset) * self.train_split)
        valid_size = len(dataset) - train_size
        train_ds, valid_ds = random_split(dataset, [train_size, valid_size])
        return train_ds, valid_ds

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Auto-docstring."""
        dataset = MarketSequenceDataset(x, y)
        train_ds, valid_ds = self._split_dataset(dataset)

        current_batch_size = self.initial_batch_size

        # Dynamic Batch Sizing - loop to catch OOM
        while current_batch_size >= 1:
            try:
                self._train_loop(train_ds, valid_ds, current_batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_batch_size > 1:
                    LOGGER.warning("CUDA OOM detected. Halving batch size from %d to %d.", current_batch_size, current_batch_size // 2)
                    torch.cuda.empty_cache()
                    current_batch_size //= 2
                    # Reset model and optimizer state before retrying
                    self.model.to(self.device)
                else:
                    raise e

        LOGGER.info("Saved trained model to %s", self.model_path)

    def _train_loop(self, train_ds, valid_ds, batch_size):
        """Auto-docstring."""
        num_workers = 4 if self.device.type == "cuda" else 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            bytes_processed = 0
            started = time.time()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}", unit="batch")
            for xb, yb in pbar:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                # Mixed Precision Training
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(xb)
                        loss = self.criterion(logits, yb)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

                bytes_processed += xb.numel() * xb.element_size() + yb.numel() * yb.element_size()
                elapsed = max(1e-6, time.time() - started)
                mb_s = (bytes_processed / (1024 * 1024)) / elapsed
                pbar.set_postfix(loss=loss.item(), mb_s=f"{mb_s:.2f}")

            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)
            valid_loss, valid_acc = self.evaluate(valid_loader)

            LOGGER.info(
                "Epoch %s: train_loss=%.4f train_acc=%.4f valid_loss=%.4f valid_acc=%.4f",
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
            )

            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            torch.save(self.model.state_dict(), self.model_path)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Auto-docstring."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)
            else:
                logits = self.model(xb)
                loss = self.criterion(logits, yb)

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        return running_loss / max(1, total), correct / max(1, total)
