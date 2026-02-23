import numpy as np

import relay_bp


def test_decode_detailed_lrbp(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.LRBPDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
        osc_window=5,
        friction_slope=2.0,
        friction_shift=0.0,
        tau=0.2,
        seed=11,
    )

    detectors = np.array([1, 1], dtype=np.uint8)
    result = decoder.decode_detailed(detectors)

    assert result.success
    assert np.all(result.decoding == np.array([0, 1, 0]))
    assert result.iterations <= 120 + 40 * 60
    assert result.max_iter == 120


def test_decode_detailed_batch_lrbp(repetition_code_config):
    repetition_code_config.pop("max_iter", None)
    decoder = relay_bp.LRBPDecoderF32(
        **repetition_code_config,
        pre_iter=120,
        num_sets=40,
        set_max_iter=60,
        gamma_dist_interval=(-0.24, 0.66),
        explicit_gammas=None,
        stop_nconv=3,
        osc_window=5,
        friction_slope=2.0,
        friction_shift=0.0,
        tau=0.2,
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
