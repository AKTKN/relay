"""
DecodeResult dataclass - decoder-agnostic result container.
This module is kept separate to avoid circular imports.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class DecodeResult:
    predictions: np.ndarray # final prediction
    iterations: Optional[np.ndarray] = None # Number of iterations taken by RelayBP decoder
    converged: Optional[np.ndarray] = None  # Whether RelayBP decoder converged
    mean_iterations: Optional[np.ndarray] = None  # Mean iterations from ensemble RelayBP decoder
    std_iterations: Optional[np.ndarray] = None   # Std of iterations from ensemble RelayBP decoder
    logical_gaps: Optional[np.ndarray] = None   # Logical gaps from RelayBP decoder
    iter_deltas: Optional[np.ndarray] = None    # Change in iterations compared to second mostlikely coset
    vote_deltas: Optional[np.ndarray] = None    # Change in votes compared the second most coset
    switch_count: Optional[int] = None   # Number of shots switched to Tesseract
    switch_resason: Optional[str] = None # Trigger code (1: NoConv, 2: Gap, 3 : MeanIter, 4: StdIter), if mulpile reasons, wrtie as "2434" including all reasons for all shots.   
    correction_change_count: Optional[int] = None # Number of bits changed in correction when switching to Tesseract
    converged_count: Optional[np.ndarray] = None  # Number of child decoders that converged per shot
    correction_hammingweight: Optional[np.ndarray] = None  # Hamming weight of the estimated error (correction) per shot
    correction_weight: Optional[np.ndarray] = None  # Total LLR of the estimated error (correction) per shot

    def has_detailed_stats(self) -> bool:
        return self.iterations is not None
    
    def has_ensemble_stats(self) -> bool:
        return self.mean_iterations is not None or self.std_iterations is not None

    @property
    def num_shots(self) -> int:
        return self.predictions.shape[0]