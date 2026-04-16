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
import pytest

import relay_bp


def _check_satisfies_syndrome(check_matrix, syndrome: np.ndarray, decoding: np.ndarray) -> bool:
    got = (check_matrix.dot(decoding.astype(np.uint8)) % 2).astype(np.uint8)
    return np.array_equal(got, syndrome.astype(np.uint8))


def test_lbf_decode_and_decode_detailed(repetition_code_sparse_csr, repetition_code_error_priors):
    decoder = relay_bp.LBFDecoder(
        repetition_code_sparse_csr,
        error_priors=repetition_code_error_priors,
        max_iter=20,
        weight=1000,
        k_step=2,
    )

    syndrome = np.array([1, 1], dtype=np.uint8)

    decoding = decoder.decode(syndrome)
    assert decoding.dtype == np.uint8
    assert decoding.shape == (3,)
    assert _check_satisfies_syndrome(repetition_code_sparse_csr, syndrome, decoding)

    detailed = decoder.decode_detailed(syndrome)
    assert detailed.success
    assert detailed.iterations <= 20
    assert detailed.max_iter == 20
    assert np.array_equal(detailed.decoded_detectors, syndrome)


def test_lbf_batch_decode(repetition_code_sparse_csr, repetition_code_error_priors):
    decoder = relay_bp.LBFDecoder(
        repetition_code_sparse_csr,
        error_priors=repetition_code_error_priors,
        max_iter=20,
        weight=1000,
        k_step=2,
    )

    syndromes = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    decodings = decoder.decode_batch(syndromes)
    assert decodings.shape == (3, 3)
    for i in range(3):
        assert _check_satisfies_syndrome(repetition_code_sparse_csr, syndromes[i], decodings[i])


def test_lbf_k_step_must_be_even(repetition_code_sparse_csr, repetition_code_error_priors):
    with pytest.raises(ValueError):
        relay_bp.LBFDecoder(
            repetition_code_sparse_csr,
            error_priors=repetition_code_error_priors,
            max_iter=20,
            weight=1000,
            k_step=3,
        )


def test_lbf_weight_warning_when_recommended_condition_violated(
    repetition_code_sparse_csr, repetition_code_error_priors
):
    # repetition code has d_max=2, so recommended condition is W>1.
    with pytest.warns(UserWarning):
        relay_bp.LBFDecoder(
            repetition_code_sparse_csr,
            error_priors=repetition_code_error_priors,
            max_iter=5,
            weight=1,
            k_step=2,
        )
