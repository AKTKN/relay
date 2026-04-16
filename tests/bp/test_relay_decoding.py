# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import numpy as np

import relay_bp


def test_decode_detailed(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
    )

    detectors = np.array([1, 1], dtype=np.uint8)

    result = decoder.decode_detailed(detectors)
    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.iterations <= 120
    assert result.max_iter == 120


def test_decode_detailed_batch(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.RelayDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
    )

    detectors = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)

    results = decoder.decode_detailed_batch(detectors)

    result0 = results[0]
    assert result0.success
    assert np.all(result0.decoding == np.array([1, 0, 0]))

    result1 = results[1]
    assert result1.success
    assert np.all(result1.decoding == np.array([0, 1, 0]))

    result2 = results[2]
    assert result2.success
    assert np.all(result2.decoding == np.array([0, 0, 1]))


def test_dual_relay_decode_detailed(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.DualRelayDecoderF64(
        **repetition_code_config,
        pre_iter=80,
        maximum_leg=20,
        iteration_per_leg=40,
        initial_gamma_slow=0.125,
        initial_gamma_fast=0.125,
        gamma_interval_slow=(0.1, 0.66),
        gamma_interval_fast=(0.1, 0.66),
        mix_mode="naive_average",
        eta=0.5,
        delta=1.0,
        n_solutions=1,
        stop_nconv=1,
        stopping_criterion="nconv",
        seed=11,
    )

    detectors = np.array([1, 1], dtype=np.uint8)
    result = decoder.decode_detailed(detectors)

    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.max_iter == 80 + 19 * 40
    assert result.dual_relay_trace is not None


def test_dual_relay_decode_detailed_batch(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.DualRelayDecoderF64(
        **repetition_code_config,
        pre_iter=80,
        maximum_leg=20,
        iteration_per_leg=40,
        initial_gamma_slow=0.125,
        initial_gamma_fast=0.125,
        gamma_interval_slow=(0.1, 0.66),
        gamma_interval_fast=(0.1, 0.66),
        mix_mode="weighted_fast",
        eta=0.5,
        delta=1.0,
        n_solutions=1,
        stop_nconv=1,
        stopping_criterion="nconv",
        seed=11,
    )

    detectors = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    results = decoder.decode_detailed_batch(detectors)

    assert results[0].success
    assert np.all(results[0].decoding == np.array([1, 0, 0]))
    assert results[1].success
    assert np.all(results[1].decoding == np.array([0, 1, 0]))
    assert results[2].success
    assert np.all(results[2].decoding == np.array([0, 0, 1]))


def test_dual_relay_with_previous_message(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.DualRelayDecoderF64(
        **repetition_code_config,
        pre_iter=80,
        maximum_leg=20,
        iteration_per_leg=40,
        initial_gamma_slow=0.125,
        initial_gamma_fast=0.125,
        gamma_interval_slow=(0.1, 0.66),
        gamma_interval_fast=(0.1, 0.66),
        mix_mode="naive_average",
        eta=0.5,
        delta=1.0,
        use_previous_message=True,
        beta=0.3,
        n_solutions=1,
        stop_nconv=1,
        stopping_criterion="nconv",
        seed=11,
    )

    detectors = np.array([1, 1], dtype=np.uint8)
    result = decoder.decode_detailed(detectors)

    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.dual_relay_trace is not None


def test_dual_relay_ensemble_mode(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.DualRelayDecoderF64(
        **repetition_code_config,
        pre_iter=40,
        maximum_leg=10,
        iteration_per_leg=20,
        ensemble_mode=True,
        ensemble_size=3,
        ensemble_gamma_interval=(0.1, 0.5),
        num_pre_iteration_instance=2,
        initial_gamma=[0.125, 0.2],
        use_previous_message=True,
        beta=0.2,
        n_solutions=1,
        stop_nconv=1,
        stopping_criterion="nconv",
        seed=11,
    )

    detectors = np.array([1, 1], dtype=np.uint8)
    result = decoder.decode_detailed(detectors)

    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.dual_relay_trace is not None
