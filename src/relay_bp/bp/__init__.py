# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Base imports/exports for relay bp rust bindings."""

from __future__ import annotations

__all__ = [
    "RelayDecoderF32",
    "RelayDecoderF64",
    "RelayDecoderI32",
    "RelayDecoderI64",
    "LRBPDecoderF32",
    "LRBPDecoderF64",
    "LRBPDecoderI32",
    "LRBPDecoderI64",
    "SLGMBPDecoderF64",
    "DisorderedBPDecoderF64",
    "DualRelayDecoderF64",
    "AdaptiveRelayDecoderF64",
    "MinSumBPDecoderF32",
    "MinSumBPDecoderF64",
    "MinSumBPDecoderI8",
    "MinSumBPDecoderI16",
    "MinSumBPDecoderI32",
    "MinSumBPDecoderI64",
    "MinSumBPDecoderFixed",
    "LBFDecoder",
]

from .._relay_bp import _bp  # pylint: disable=E0611
RelayDecoderF32 = _bp.RelayDecoderF32
RelayDecoderF64 = _bp.RelayDecoderF64
RelayDecoderI32 = _bp.RelayDecoderI32
RelayDecoderI64 = _bp.RelayDecoderI64
LRBPDecoderF32 = _bp.LRBPDecoderF32
LRBPDecoderF64 = _bp.LRBPDecoderF64
LRBPDecoderI32 = _bp.LRBPDecoderI32
LRBPDecoderI64 = _bp.LRBPDecoderI64
SLGMBPDecoderF64 = _bp.SLGMBPDecoderF64
DisorderedBPDecoderF64 = _bp.DisorderedBPDecoderF64
DualRelayDecoderF64 = _bp.DualRelayDecoderF64
AdaptiveRelayDecoderF64 = _bp.AdaptiveRelayDecoderF64
MinSumBPDecoderF32 = _bp.MinSumBPDecoderF32
MinSumBPDecoderF64 = _bp.MinSumBPDecoderF64
MinSumBPDecoderI8 = _bp.MinSumBPDecoderI8
MinSumBPDecoderI16 = _bp.MinSumBPDecoderI16
MinSumBPDecoderI32 = _bp.MinSumBPDecoderI32
MinSumBPDecoderI64 = _bp.MinSumBPDecoderI64
MinSumBPDecoderFixed = _bp.MinSumBPDecoderFixed
LBFDecoder = _bp.LBFDecoder
