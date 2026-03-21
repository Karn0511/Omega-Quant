"""Phase 6: Multi-Asset Intelligence."""
import logging
# pylint: disable=import-error
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
# pylint: enable=import-error
from typing import List, Dict

LOGGER = logging.getLogger(__name__)

class PortfolioAllocator:
    """Manages multi-asset correlation matrices to prevent overexposure."""

    def __init__(self, target_assets: List[str]):
        """Auto-docstring."""
        self.assets = target_assets
        self.correlation_matrix = pd.DataFrame()
        self.allocations = {asset: 1.0 / len(target_assets) for asset in target_assets}

    def update_correlations(self, price_histories: Dict[str, pd.DataFrame]):
        """Compute rolling correlation matrix across given assets."""
        df_close = pd.DataFrame()
        for asset, df in price_histories.items():
            if not df.empty and "close" in df.columns:
                df_close[asset] = df["close"]

        self.correlation_matrix = df_close.pct_change().corr()
        LOGGER.info("Portfolio Allocator: Correlation matrix updated.")

    def get_dynamic_allocation(self, asset: str) -> float:
        """Reduce capital exposure dynamically if highly correlated assets are volatile."""
        if self.correlation_matrix.empty or asset not in self.correlation_matrix.columns:
            return self.allocations.get(asset, 0.0)

        corrs = self.correlation_matrix[asset].drop(asset)
        high_corr = corrs[corrs > 0.8]

        penalty = len(high_corr) * 0.15 # Penalize allocation by 15% for every highly correlated companion

        base_alloc = self.allocations.get(asset, 0.0)
        final_alloc = max(0.05, base_alloc - penalty) # Floor at 5% alloc minimum
        return final_alloc

    def get_total_exposure(self) -> float:
        """Calculate aggregate exposure across target assets."""
        return float(sum(self.allocations.values()))
