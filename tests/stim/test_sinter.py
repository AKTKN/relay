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
import pathlib
import pytest
import sinter
import stim
import tempfile

_STIM_IMPORT_ERROR: Exception | None = None
try:
    from relay_bp.stim import (
        SinterDecoder_RelayBP,
        SinterDecoder_AdaptiveRelay,
        SinterDecoder_DualRelay,
        SinterDecoder_LBF,
        SinterDecoder_SLGMBP,
        sinter_decoders,
        CheckMatrices,
    )
except Exception as exc:  # pragma: no cover
    _STIM_IMPORT_ERROR = exc
    SinterDecoder_RelayBP = None  # type: ignore[assignment]
    SinterDecoder_AdaptiveRelay = None  # type: ignore[assignment]
    SinterDecoder_DualRelay = None  # type: ignore[assignment]
    SinterDecoder_LBF = None  # type: ignore[assignment]
    SinterDecoder_SLGMBP = None  # type: ignore[assignment]
    sinter_decoders = None  # type: ignore[assignment]
    CheckMatrices = None  # type: ignore[assignment]


def _require_relay_bp_stim() -> None:
    if _STIM_IMPORT_ERROR is not None:
        pytest.skip(
            "relay_bp.stim import failed (likely beliefmatching/ldpc environment issue): "
            + str(_STIM_IMPORT_ERROR)
        )


def _require_testdata():
    _require_relay_bp_stim()
    try:
        from testdata import (
            get_test_circuit,
            get_all_test_circuits,
            filter_detectors_by_basis,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(
            "tests/testdata import failed (likely relay_bp.stim import dependency issue): "
            + str(exc)
        )
    return get_test_circuit, get_all_test_circuits, filter_detectors_by_basis


def test_check_matrix_pruning():
    _require_relay_bp_stim()
    """Test decoding of the surface code via files."""
    circuit = stim.Circuit.generated(
        rounds=11,
        distance=11,
        after_clifford_depolarization=0.003,
        code_task=f"surface_code:rotated_memory_z",
    )
    dem = circuit.detector_error_model(decompose_errors=True)

    check_matrices = CheckMatrices.from_dem(
        dem, decomposed_hyperedges=True, prune_decided_errors=False
    )
    check_matrices_pruned = CheckMatrices.from_dem(
        dem, decomposed_hyperedges=True, prune_decided_errors=True
    )

    assert (
        check_matrices.check_matrix.shape[0]
        == check_matrices_pruned.check_matrix.shape[0]
    )
    assert (
        check_matrices.check_matrix.shape[1]
        > check_matrices_pruned.check_matrix.shape[1]
    )
    assert (
        check_matrices.observables_matrix.shape[0]
        == check_matrices_pruned.observables_matrix.shape[0]
    )
    assert (
        check_matrices.observables_matrix.shape[1]
        > check_matrices_pruned.observables_matrix.shape[1]
    )


def test_sinter_relay_bp_decoder_integration():
    _require_relay_bp_stim()
    """Test decoding of the surface code with sinter."""

    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["relay-bp"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "relay-bp"
    assert samples[0].errors <= 5
    assert samples[0].shots == 100


def test_sinter_msl_bp_decoder_integration():
    _require_relay_bp_stim()
    """Test decoding of the surface code with sinter."""

    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["msl-bp"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "msl-bp"
    assert samples[0].errors <= 10
    assert samples[0].shots == 100


def test_sinter_lbf_decoder_integration():
    _require_relay_bp_stim()
    """Smoke-test decoding via sinter using the LBF decoder wrapper."""

    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    # Ensure the symbol is importable and the decoder runs through sinter.
    assert SinterDecoder_LBF is not None

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["lbf"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "lbf"
    assert samples[0].shots == 100


def test_sinter_lbf_decoder_with_detailed_stats_aux_payload() -> None:
    _require_relay_bp_stim()

    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task="surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    compiled = SinterDecoder_LBF(get_detailed_stats=True).compile_decoder_for_dem(dem=dem)

    dets, _ = circuit.compile_detector_sampler().sample(
        shots=16,
        bit_packed=True,
        separate_observables=True,
    )
    _ = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)

    pop_aux = getattr(compiled, "pop_last_decode_aux", None)
    assert callable(pop_aux)
    aux = pop_aux()
    assert isinstance(aux, dict)
    assert "iterations" in aux
    assert "converged" in aux
    assert len(aux["iterations"]) == 16
    assert len(aux["converged"]) == 16



def test_sinter_mem_bp_decoder_integration():
    _require_relay_bp_stim()
    """Test decoding of the surface code with sinter."""

    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    # Collect the samples (takes a few minutes).
    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["mem-bp"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "mem-bp"
    assert samples[0].errors <= 20
    assert samples[0].shots == 100


def test_sinter_lrbp_decoder_integration():
    _require_relay_bp_stim()
    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["lr-bp"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "lr-bp"
    assert samples[0].errors <= 20
    assert samples[0].shots == 100


def test_sinter_adaptive_relay_decoder_integration():
    _require_relay_bp_stim()
    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["adaptive-relay"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "adaptive-relay"
    assert samples[0].errors <= 20
    assert samples[0].shots == 100


def test_sinter_dual_relay_decoder_integration():
    _require_relay_bp_stim()
    def generate_example_tasks():
        for p in [0.0001]:
            for d in [3]:
                yield sinter.Task(
                    circuit=stim.Circuit.generated(
                        rounds=d,
                        distance=d,
                        after_clifford_depolarization=p,
                        code_task=f"surface_code:rotated_memory_x",
                    ),
                    json_metadata={
                        "p": p,
                        "d": d,
                    },
                )

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=generate_example_tasks(),
        decoders=["dual-relay"],
        custom_decoders=sinter_decoders(),
    )
    assert samples[0].decoder == "dual-relay"
    assert samples[0].errors <= 20
    assert samples[0].shots == 100


def test_dual_relay_decoder_build_observable_decoder():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_DualRelay(
        pre_iter=40,
        maximum_leg=10,
        iteration_per_leg=20,
        mix_mode="weighted_fast",
        eta=0.5,
        delta=1.0,
        n_solutions=1,
    )
    observable_decoder = decoder.build_observable_decoder(check_matrices)
    assert observable_decoder is not None


def test_adaptive_relay_decoder_accepts_posterior_clamp_options():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_AdaptiveRelay(
        posterior_marginal_clamp_mode="abs_threshold_clamp",
        posterior_marginal_abs_threshold=1e10,
    )
    observable_decoder = decoder.build_observable_decoder(check_matrices)

    assert observable_decoder is not None


def test_sinter_decode_via_files():
    _require_relay_bp_stim()
    """Test decoding of the surface code via files."""
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model()

    with tempfile.TemporaryDirectory() as d:
        testdir = pathlib.Path(d)

        circuit.compile_detector_sampler().sample_write(
            shots=100,
            filepath=testdir / "detectors.b8",
            format="b8",
        )

        dem.to_file(testdir / "dem.dem")

        SinterDecoder_RelayBP(parallel=True).decode_via_files(
            num_shots=10,
            num_dets=dem.num_detectors,
            num_obs=dem.num_observables,
            dem_path=testdir / "dem.dem",
            dets_b8_in_path=testdir / "detectors.b8",
            obs_predictions_b8_out_path=testdir / "observable_predictions.b8",
            tmp_dir=testdir,
        )

        predictions = stim.read_shot_data_file(
            path=testdir / "observable_predictions.b8",
            format="b8",
            num_observables=dem.num_observables,
        )
        assert np.sum(predictions) <= 5


def test_slg_mbp_decoder_accepts_drop_params():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_SLGMBP(drop_p=0.5, drop_llr_threshold=1.0)
    observable_decoder = decoder.build_observable_decoder(check_matrices)

    assert observable_decoder is not None


def test_slg_mbp_decoder_accepts_adaptive_perturbation_sign_probability():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_SLGMBP(
        adaptive_perturbation=True,
        adaptive_perturbation_sign_mode="random",
        adaptive_perturbation_positive_sign_prob=0.8,
    )
    observable_decoder = decoder.build_observable_decoder(check_matrices)

    assert observable_decoder is not None


def test_slg_mbp_decoder_accepts_adaptive_perturbation_target_modes():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    for mode in ("prior", "posterior", "both", "memory_strength"):
        decoder = SinterDecoder_SLGMBP(
            adaptive_perturbation=True,
            adaptive_perturbation_target=mode,
        )
        observable_decoder = decoder.build_observable_decoder(check_matrices)
        assert observable_decoder is not None


def test_slg_mbp_decoder_accepts_dynamic_adaptive_perturbation_threshold_params():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_SLGMBP(
        adaptive_perturbation=True,
        adaptive_perturbation_llr_threshold_mode="generation_log10_iter",
        adaptive_perturbation_llr_threshold=0.5,
        adaptive_perturbation_llr_threshold_min=0.5,
        adaptive_perturbation_llr_threshold_max=2.0,
        adaptive_perturbation_llr_threshold_factor=0.25,
    )
    observable_decoder = decoder.build_observable_decoder(check_matrices)

    assert observable_decoder is not None


def test_slg_mbp_decoder_accepts_adaptive_perturbation_factor_and_bias_modes():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_SLGMBP(
        adaptive_perturbation=True,
        adaptive_perturbation_factor_mode="uniform_per_variable",
        adaptive_perturbation_factor_interval=(0.2, 0.8),
        adaptive_perturbation_bias_mode="scale",
    )
    observable_decoder = decoder.build_observable_decoder(check_matrices)

    assert observable_decoder is not None


def test_slg_mbp_decoder_accepts_adaptive_prior_carry_and_reset_options():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_SLGMBP(
        adaptive_perturbation=True,
        adaptive_perturbation_target="prior",
        adaptive_perturbation_prior_base_mode="previous_biased",
        adaptive_perturbation_reset_on_threshold_exit=True,
    )
    observable_decoder = decoder.build_observable_decoder(check_matrices)

    assert observable_decoder is not None


def test_slg_mbp_decoder_accepts_first_leg_fixed_adaptive_variable_mode():
    _require_relay_bp_stim()
    circuit = stim.Circuit.generated(
        rounds=3,
        distance=3,
        after_clifford_depolarization=0.0001,
        code_task=f"surface_code:rotated_memory_x",
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    check_matrices = CheckMatrices.from_dem(dem, decomposed_hyperedges=True)

    decoder = SinterDecoder_SLGMBP(
        adaptive_perturbation=True,
        adaptive_perturbation_target="prior",
        adaptive_perturbation_variable_base_mode="first_leg_posterior_fixed",
        adaptive_perturbation_sign_mode="random",
    )
    observable_decoder = decoder.build_observable_decoder(check_matrices)

    assert observable_decoder is not None


def test_get_testdata_circuit():
    _require_relay_bp_stim()
    """Test getting test circuit and decoding."""
    get_test_circuit, _, _ = _require_testdata()
    circuit = get_test_circuit("bicycle_bivariate_18_4_3_memory_Z", 0.001)
    tasks = [sinter.Task(circuit=circuit)]

    samples = sinter.collect(
        num_workers=2,
        max_shots=1_00,
        tasks=tasks,
        decoders=["relay-bp"],
        custom_decoders=sinter_decoders(),
    )

    assert samples[0].decoder == "relay-bp"
    assert samples[0].errors <= 10
    assert samples[0].shots == 100


def test_get_all_testdata_circuit():
    _require_relay_bp_stim()
    """Test getting test circuit and decoding."""
    _, get_all_test_circuits, _ = _require_testdata()
    circuits = get_all_test_circuits("*", 0.001)
    for name, circuit in circuits.items():
        assert isinstance(name, str)
        assert isinstance(circuit, stim.Circuit)

    assert len(circuits) > 1


def test_filter_detectors_by_basis():
    _require_relay_bp_stim()
    """Test getting test circuit and decoding."""
    get_test_circuit, _, filter_detectors_by_basis = _require_testdata()
    circuit = get_test_circuit("bicycle_bivariate_18_4_3_memory_Z", 0.001)

    dem = circuit.detector_error_model()
    check_matrices = CheckMatrices.from_dem(dem)

    assert check_matrices.check_matrix.shape == (54, 1800)

    z_circuit = filter_detectors_by_basis(circuit, "Z")
    z_dem = z_circuit.detector_error_model()
    z_check_matrices = CheckMatrices.from_dem(z_dem)

    assert z_check_matrices.check_matrix.shape == (36, 288)
