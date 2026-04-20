# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import pathlib
from typing import Iterable

from sinter import Decoder, CompiledDecoder
import numpy as np

import stim

import relay_bp

from .check_matrices import CheckMatrices


class SinterCompiledDecoder_BP(CompiledDecoder):
    def __init__(
        self,
        observable_decoder: relay_bp.ObservableDecoderRunner,
        check_matrices: CheckMatrices,
        parallel: bool = False,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        collect_iteration_metric: bool = False,
    ):
        self.observable_decoder = observable_decoder
        self.parallel = parallel
        self.check_matrices = check_matrices
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish
        self.collect_iteration_metric = collect_iteration_metric
        self._last_decode_aux: dict[str, np.ndarray] | None = None

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: "np.ndarray",
    ) -> "np.ndarray":
        self._last_decode_aux = None

        syndromes = np.unpackbits(
            bit_packed_detection_event_data, bitorder="little", axis=1
        ).astype(np.uint8)
        # Packed shot rows are byte-aligned; drop trailing padding bits beyond real detectors.
        num_detectors = int(self.check_matrices.check_matrix.shape[0])
        if syndromes.shape[1] < num_detectors:
            raise ValueError(
                "Decoded syndrome width is smaller than the number of detectors "
                f"({syndromes.shape[1]} < {num_detectors})."
            )
        syndromes = syndromes[:, :num_detectors]

        if self.check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + self.check_matrices.syndrome_bias) % 2

        if self.collect_iteration_metric:
            detailed = self.observable_decoder.decode_observables_detailed_batch(
                syndromes,
                parallel=self.parallel,
                progress_bar=self.show_progress,
                leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            )
            predictions = np.asarray([entry.observables for entry in detailed], dtype=np.uint8)
            self._last_decode_aux = {
                "iterations": np.asarray(
                    [int(entry.iterations) for entry in detailed], dtype=np.int64
                ),
                "converged": np.asarray(
                    [bool(entry.converged) for entry in detailed], dtype=np.bool_
                ),
                "force_logical_error": np.asarray(
                    [bool(entry.force_logical_error) for entry in detailed], dtype=np.bool_
                ),
                "confidence_score_token": np.asarray(
                    [
                        str(entry.confidence_score_token)
                        if entry.confidence_score_token is not None
                        else "inf"
                        for entry in detailed
                    ],
                    dtype=object,
                ),
            }
        else:
            predictions = self.observable_decoder.decode_observables_batch(
                syndromes,
                parallel=self.parallel,
                progress_bar=self.show_progress,
                leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            )

        if self.check_matrices.observables_bias is not None:
            predictions = (predictions + self.check_matrices.observables_bias) % 2

        outputs = np.packbits(predictions, axis=1, bitorder="little")
        return outputs

    def pop_last_decode_aux(self) -> dict[str, np.ndarray] | None:
        aux = self._last_decode_aux
        self._last_decode_aux = None
        return aux


class SinterDecoder_BaseBP(Decoder):
    def __init__(
        self,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        collect_iteration_metric: bool = False,
    ):
        f"""Class for decoding stim circuits with sinter and relay-bp."""
        self.parallel = parallel
        self.decomposed_hyperedges = decomposed_hyperedges
        self.prune_decided_errors = prune_decided_errors
        self.threshold = threshold
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish
        self.collect_iteration_metric = collect_iteration_metric

    def build_observable_decoder(
        self, dem: stim.DetectorErrorModel
    ) -> relay_bp.ObservableDecoderRunner:
        raise NotImplementedError("Not yet implemented")

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledDecoder:
        check_matrices = CheckMatrices.from_dem(
            dem,
            decomposed_hyperedges=self.decomposed_hyperedges,
            prune_decided_errors=self.prune_decided_errors,
            threshold=self.threshold,
        )
        observable_decoder_runner = self.build_observable_decoder(check_matrices)
        return SinterCompiledDecoder_BP(
            observable_decoder_runner,
            check_matrices=check_matrices,
            parallel=self.parallel,
            show_progress=self.show_progress,
            leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            collect_iteration_metric=self.collect_iteration_metric,
        )

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:

        dem = stim.DetectorErrorModel.from_file(dem_path)
        check_matrices = CheckMatrices.from_dem(
            dem,
            decomposed_hyperedges=self.decomposed_hyperedges,
            prune_decided_errors=self.prune_decided_errors,
            threshold=self.threshold,
        )

        observable_decoder = self.build_observable_decoder(check_matrices)

        syndromes = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format="b8",
            num_detectors=dem.num_detectors,
            bit_packed=False,
        ).astype(np.uint8)

        if check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + check_matrices.syndrome_bias) % 2

        predictions = observable_decoder.decode_observables_batch(
            syndromes,
            parallel=self.parallel,
            progress_bar=self.show_progress,
            leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
        )

        if check_matrices.observables_bias is not None:
            predictions = (predictions + check_matrices.observables_bias) % 2

        stim.write_shot_data_file(
            data=np.packbits(predictions, axis=1, bitorder="little"),
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=dem.num_observables,
        )


class SinterDecoder_RelayBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        alpha: float | None = None,
        gamma0: float = 0.1,
        pre_iter: int = 60,
        num_sets: int = 60,
        set_max_iter: int = 60,
        gamma_dist_interval: tuple[float, float] = (-0.24, 0.66),
        explicit_gammas: np.ndarray | None = None,
        stop_nconv: int = 5,
        stopping_criterion: str = "nconv",
        logging=False,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        f"""Class for decoding stim circuits with sinter and relay-bp."""
        self.alpha = alpha
        self.gamma0 = gamma0
        self.pre_iter = pre_iter
        self.num_sets = num_sets
        self.set_max_iter = set_max_iter
        self.gamma_dist_interval = tuple(gamma_dist_interval)
        self.explicit_gammas = explicit_gammas
        self.stop_nconv = stop_nconv
        self.stopping_criterion = stopping_criterion
        self.logging = logging
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:

        decoder = relay_bp.RelayDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if self.alpha == 0.0 else self.alpha,
            gamma0=self.gamma0,
            pre_iter=self.pre_iter,
            num_sets=self.num_sets,
            set_max_iter=self.set_max_iter,
            gamma_dist_interval=self.gamma_dist_interval,
            explicit_gammas=self.explicit_gammas,
            stop_nconv=self.stop_nconv,
            stopping_criterion=self.stopping_criterion,
            logging=self.logging,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_AdaptiveRelay(SinterDecoder_BaseBP):
    def __init__(
        self,
        alpha: float | None = None,
        initial_gamma: float = 0.125,
        gamma_min: float = -0.24,
        gamma_max: float = 0.66,
        tau: float = 1.0,
        beta: float = 0.9,
        pre_decoding: bool = False,
        pre_iteration: int = 80,
        maximum_iteration: int = 600,
        iter_per_leg: int = 60,
        update_mode: str = "per-iteration",
        perturbation_mode: str = "uniform",
        perturbation_interval: tuple[float, float] = (0.0, 0.0),
        perturbation_sigma: float = 0.0,
        ensemble_size: int = 1,
        carry_marginal_between_legs: bool = True,
        posterior_marginal_clamp_mode: str = "no_clamp",
        posterior_marginal_abs_threshold: float = 1e10,
        seed: int = 0,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        self.alpha = alpha
        self.initial_gamma = initial_gamma
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau = tau
        self.beta = beta
        self.pre_decoding = pre_decoding
        self.pre_iteration = pre_iteration
        self.maximum_iteration = maximum_iteration
        self.iter_per_leg = iter_per_leg
        self.update_mode = update_mode
        self.perturbation_mode = perturbation_mode
        self.perturbation_interval = tuple(perturbation_interval)
        self.perturbation_sigma = perturbation_sigma
        self.ensemble_size = ensemble_size
        self.carry_marginal_between_legs = carry_marginal_between_legs
        self.posterior_marginal_clamp_mode = posterior_marginal_clamp_mode
        self.posterior_marginal_abs_threshold = float(posterior_marginal_abs_threshold)
        self.seed = seed
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.AdaptiveRelayDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if self.alpha == 0.0 else self.alpha,
            initial_gamma=self.initial_gamma,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
            tau=self.tau,
            beta=self.beta,
            pre_decoding=self.pre_decoding,
            pre_iteration=self.pre_iteration,
            maximum_iteration=self.maximum_iteration,
            iter_per_leg=self.iter_per_leg,
            update_mode=self.update_mode,
            perturbation_mode=self.perturbation_mode,
            perturbation_interval=self.perturbation_interval,
            perturbation_sigma=self.perturbation_sigma,
            ensemble_size=self.ensemble_size,
            carry_marginal_between_legs=self.carry_marginal_between_legs,
            posterior_marginal_clamp_mode=self.posterior_marginal_clamp_mode,
            posterior_marginal_abs_threshold=self.posterior_marginal_abs_threshold,
            seed=self.seed,
            collect_iteration_metric=self.collect_iteration_metric,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_DisorderedBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        alpha: float | None = None,
        t_0: int = 80,
        maximum_leg: int = 100,
        iteration_per_leg: int = 60,
        initial_alpha: float = 0.625,
        alpha_mode: str = "interval_random",
        alpha_fixed: float = 0.625,
        alpha_interval: tuple[float, float] = (0.6, 0.7),
        initial_gamma: float = 0.125,
        gamma_mode: str = "interval_random",
        gamma_fixed: float = 0.125,
        gamma_interval: tuple[float, float] = (-0.24, 0.66),
        bias_mode: str = "fixed",
        bias_fixed: float = 0.0,
        bias_interval: tuple[float, float] = (0.0, 0.0),
        bias_apply_mode: str = "all",
        bias_filter_threshold: float = 1.0,
        negative_sign_prob: float = 0.0,
        carry_marginal_factor: float = 1.0,
        seed: int = 0,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        self.alpha = alpha
        self.t_0 = t_0
        self.maximum_leg = maximum_leg
        self.iteration_per_leg = iteration_per_leg
        self.initial_alpha = initial_alpha
        self.alpha_mode = alpha_mode
        self.alpha_fixed = alpha_fixed
        self.alpha_interval = tuple(alpha_interval)
        self.initial_gamma = initial_gamma
        self.gamma_mode = gamma_mode
        self.gamma_fixed = gamma_fixed
        self.gamma_interval = tuple(gamma_interval)
        self.bias_mode = bias_mode
        self.bias_fixed = bias_fixed
        self.bias_interval = tuple(bias_interval)
        self.bias_apply_mode = bias_apply_mode
        self.bias_filter_threshold = bias_filter_threshold
        if not 0.0 <= negative_sign_prob <= 1.0:
            raise ValueError("negative_sign_prob must be between 0.0 and 1.0")
        self.negative_sign_prob = negative_sign_prob
        if not 0.0 <= carry_marginal_factor <= 1.0:
            raise ValueError("carry_marginal_factor must be between 0.0 and 1.0")
        self.carry_marginal_factor = carry_marginal_factor
        self.seed = seed

        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.DisorderedBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if self.alpha == 0.0 else self.alpha,
            t_0=self.t_0,
            maximum_leg=self.maximum_leg,
            iteration_per_leg=self.iteration_per_leg,
            initial_alpha=self.initial_alpha,
            alpha_mode=self.alpha_mode,
            alpha_fixed=self.alpha_fixed,
            alpha_interval=self.alpha_interval,
            initial_gamma=self.initial_gamma,
            gamma_mode=self.gamma_mode,
            gamma_fixed=self.gamma_fixed,
            gamma_interval=self.gamma_interval,
            bias_mode=self.bias_mode,
            bias_fixed=self.bias_fixed,
            bias_interval=self.bias_interval,
            bias_apply_mode=self.bias_apply_mode,
            bias_filter_threshold=self.bias_filter_threshold,
            negative_sign_prob=self.negative_sign_prob,
            carry_marginal_factor=self.carry_marginal_factor,
            seed=self.seed,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_DualRelay(SinterDecoder_BaseBP):
    def __init__(
        self,
        alpha: float | None = None,
        alpha_iteration_scaling_factor: float = 1.0,
        gamma0: float = 0.125,
        pre_iter: int = 80,
        maximum_leg: int = 100,
        iteration_per_leg: int = 60,
        initial_gamma_slow: float = 0.125,
        initial_gamma_fast: float = 0.125,
        gamma_interval_slow: tuple[float, float] = (0.1, 0.66),
        gamma_interval_fast: tuple[float, float] = (0.1, 0.66),
        mix_mode: str = "naive_average",
        eta: float = 0.5,
        delta: float = 1.0,
        use_previous_message: bool = False,
        beta: float = 0.0,
        ensemble_mode: bool = False,
        ensemble_size: int = 2,
        ensemble_gamma_interval: tuple[float, float] = (0.1, 0.66),
        num_pre_iteration_instance: int = 1,
        initial_gamma: list[float] | tuple[float, ...] = (0.125,),
        n_solutions: int = 1,
        stop_nconv: int | None = None,
        stopping_criterion: str = "nconv",
        seed: int = 0,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = True,
    ):
        self.alpha = alpha
        self.alpha_iteration_scaling_factor = alpha_iteration_scaling_factor
        self.gamma0 = gamma0
        self.pre_iter = pre_iter
        self.maximum_leg = maximum_leg
        self.iteration_per_leg = iteration_per_leg
        self.initial_gamma_slow = initial_gamma_slow
        self.initial_gamma_fast = initial_gamma_fast
        self.gamma_interval_slow = tuple(gamma_interval_slow)
        self.gamma_interval_fast = tuple(gamma_interval_fast)
        self.mix_mode = mix_mode
        self.eta = eta
        self.delta = delta
        self.use_previous_message = bool(use_previous_message)
        self.beta = float(beta)
        self.ensemble_mode = bool(ensemble_mode)
        self.ensemble_size = max(1, int(ensemble_size))
        self.ensemble_gamma_interval = tuple(ensemble_gamma_interval)
        self.num_pre_iteration_instance = max(1, int(num_pre_iteration_instance))
        self.initial_gamma = [float(v) for v in initial_gamma]
        self.n_solutions = max(1, int(n_solutions))
        self.stop_nconv = int(stop_nconv) if stop_nconv is not None else self.n_solutions
        self.stopping_criterion = stopping_criterion
        self.seed = seed

        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.DualRelayDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if self.alpha == 0.0 else self.alpha,
            alpha_iteration_scaling_factor=self.alpha_iteration_scaling_factor,
            gamma0=self.gamma0,
            pre_iter=self.pre_iter,
            maximum_leg=self.maximum_leg,
            iteration_per_leg=self.iteration_per_leg,
            initial_gamma_slow=self.initial_gamma_slow,
            initial_gamma_fast=self.initial_gamma_fast,
            gamma_interval_slow=self.gamma_interval_slow,
            gamma_interval_fast=self.gamma_interval_fast,
            mix_mode=self.mix_mode,
            eta=self.eta,
            delta=self.delta,
            use_previous_message=self.use_previous_message,
            beta=self.beta,
            ensemble_mode=self.ensemble_mode,
            ensemble_size=self.ensemble_size,
            ensemble_gamma_interval=self.ensemble_gamma_interval,
            num_pre_iteration_instance=self.num_pre_iteration_instance,
            initial_gamma=self.initial_gamma,
            n_solutions=self.n_solutions,
            stop_nconv=self.stop_nconv,
            stopping_criterion=self.stopping_criterion,
            seed=self.seed,
            collect_iteration_metric=self.collect_iteration_metric,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_MemBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        max_iter: int = 100,
        alpha: float | None = None,
        gamma0: float = 0.1,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        f"""Class for decoding stim circuits with sinter and mem-bp."""
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma0 = gamma0
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:

        decoder = relay_bp.MinSumBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=self.max_iter,
            alpha=None if self.alpha == 0.0 else self.alpha,
            gamma0=self.gamma0,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_LBF(SinterDecoder_BaseBP):
    def __init__(
        self,
        max_iter: int = 200,
        weight: int = 1000,
        k_step: int = 2,
        get_detailed_stats: bool = False,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
    ):
        f"""Class for decoding stim circuits with sinter using LBF (local bit-flipping)."""
        self.max_iter = int(max_iter)
        self.weight = int(weight)
        self.k_step = int(k_step)
        self.get_detailed_stats = bool(get_detailed_stats)
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=self.get_detailed_stats,
        )

    def build_observable_decoder(
        self,
        check_matrices: CheckMatrices,
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.LBFDecoder(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=self.max_iter,
            weight=self.weight,
            k_step=self.k_step,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_ClipBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        min_llr: float = 0.3,
        sign_mode: str = "hysteresis",
        gamma_first: float = 0.125,
        gamma_center: float = 0.21,
        gamma_width: float = 0.9,
        max_legs: int = 100,
        max_iter_first: int = 80,
        max_iter: int = 60,
        max_solutions: int = 1,
        alpha: float | None = None,
        seed: int = 0,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        self.min_llr = min_llr
        self.sign_mode = sign_mode
        self.gamma_first = gamma_first
        self.gamma_center = gamma_center
        self.gamma_width = gamma_width
        self.max_legs = max_legs
        self.max_iter_first = max_iter_first
        self.max_iter = max_iter
        self.max_solutions = max_solutions
        self.alpha = alpha
        self.seed = seed
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self,
        check_matrices: CheckMatrices,
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.ClipBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            min_llr=self.min_llr,
            sign_mode=self.sign_mode,
            gamma_first=self.gamma_first,
            gamma_center=self.gamma_center,
            gamma_width=self.gamma_width,
            max_legs=self.max_legs,
            max_iter_first=self.max_iter_first,
            max_iter=self.max_iter,
            max_solutions=self.max_solutions,
            alpha=None if self.alpha == 0.0 else self.alpha,
            seed=self.seed,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_LRBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        alpha: float | None = None,
        gamma0: float = 0.1,
        pre_iter: int = 60,
        num_sets: int = 60,
        set_max_iter: int = 60,
        odd_leg_max_iter: int = 60,
        even_leg_uniform_gamma: float = 0.0,
        gamma_dist_interval: tuple[float, float] = (-0.24, 0.66),
        explicit_gammas: np.ndarray | None = None,
        stop_nconv: int = 5,
        stopping_criterion: str = "nconv",
        logging=False,
        seed: int = 0,
        osc_window: int = 5,
        friction_slope: float = 2.0,
        friction_shift: float = 0.0,
        tau: float = 0.2,
        r_dyn: int = 0,
        t_dyn: int = 0,
        dyn_mode: str = "EBP",
        gamma_dyn_penalty: float = 0.0,
        dyn_gamma_mode: str = "fixed",
        gamma_dyn_center: float = 0.0,
        gamma_dyn_min: float = -0.24,
        gamma_dyn_max: float = 0.66,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        self.alpha = alpha
        self.gamma0 = gamma0
        self.pre_iter = pre_iter
        self.num_sets = num_sets
        self.set_max_iter = set_max_iter
        self.odd_leg_max_iter = odd_leg_max_iter
        self.even_leg_uniform_gamma = even_leg_uniform_gamma
        self.gamma_dist_interval = tuple(gamma_dist_interval)
        self.explicit_gammas = explicit_gammas
        self.stop_nconv = stop_nconv
        self.stopping_criterion = stopping_criterion
        self.logging = logging
        self.seed = seed
        self.osc_window = osc_window
        self.friction_slope = friction_slope
        self.friction_shift = friction_shift
        self.tau = tau
        self.r_dyn = r_dyn
        self.t_dyn = t_dyn
        self.dyn_mode = dyn_mode
        self.gamma_dyn_penalty = gamma_dyn_penalty
        self.dyn_gamma_mode = dyn_gamma_mode
        self.gamma_dyn_center = gamma_dyn_center
        self.gamma_dyn_min = gamma_dyn_min
        self.gamma_dyn_max = gamma_dyn_max
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.LRBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if self.alpha == 0.0 else self.alpha,
            gamma0=self.gamma0,
            pre_iter=self.pre_iter,
            num_sets=self.num_sets,
            set_max_iter=self.set_max_iter,
            odd_leg_max_iter=self.odd_leg_max_iter,
            even_leg_uniform_gamma=self.even_leg_uniform_gamma,
            gamma_dist_interval=self.gamma_dist_interval,
            explicit_gammas=self.explicit_gammas,
            stop_nconv=self.stop_nconv,
            stopping_criterion=self.stopping_criterion,
            logging=self.logging,
            seed=self.seed,
            osc_window=self.osc_window,
            friction_slope=self.friction_slope,
            friction_shift=self.friction_shift,
            tau=self.tau,
            r_dyn=self.r_dyn,
            t_dyn=self.t_dyn,
            dyn_mode=self.dyn_mode,
            gamma_dyn_penalty=self.gamma_dyn_penalty,
            dyn_gamma_mode=self.dyn_gamma_mode,
            gamma_dyn_center=self.gamma_dyn_center,
            gamma_dyn_min=self.gamma_dyn_min,
            gamma_dyn_max=self.gamma_dyn_max,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_SLGMBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        alpha: float | None = None,
        ensemble_size: int = 64,
        t_ms: int = 16,
        t_mem: int = 16,
        g_max: int = 30,
        sigma2: float = 0.15,
        delta: float = 0.2,
        fitness_alpha: float = 1000.0,
        fitness_beta: float = 1.0,
        fitness_low_llr_mu: float = 0.0,
        fitness_low_llr_threshold: float = 0.0,
        eta: float = 1.0,
        mutation_rate: float = 0.02,
        mutation_llr_abs_threshold: float = 0.25,
        sequential_mc: bool = False,
        init_perturbation_mode: str = "gaussian",
        init_strategy: str = "min-sum",
        init_gamma: float = 0.125,
        initial_prior_bias: float = 0.0,
        adaptive_perturbation: bool = False,
        adaptive_perturbation_llr_threshold_mode: str = "constant",
        adaptive_perturbation_llr_threshold: float = 0.0,
        adaptive_perturbation_llr_threshold_min: float = 0.0,
        adaptive_perturbation_llr_threshold_max: float = 0.0,
        adaptive_perturbation_llr_threshold_factor: float = 0.0,
        adaptive_perturbation_factor: float = 1.0,
        adaptive_perturbation_factor_mode: str = "fixed",
        adaptive_perturbation_factor_interval: tuple[float, float] = (1.0, 1.0),
        adaptive_perturbation_bias_mode: str = "additive",
        adaptive_perturbation_sign_mode: str = "random",
        adaptive_perturbation_positive_sign_prob: float = 0.5,
        adaptive_perturbation_target: str = "prior",
        adaptive_perturbation_prior_base_mode: str = "initial",
        adaptive_perturbation_variable_base_mode: str = "previous_leg_posterior",
        adaptive_perturbation_reset_on_threshold_exit: bool = False,
        continue_perturbation: bool = False,
        perturbation_method: str = "fixed",
        reset_marginal: bool = False,
        marginal_carry_damping_factor: float = 1.0,
        marginal_carry_llr_abs_threshold: float = -1.0,
        drop_p: float = 0.0,
        drop_llr_threshold: float = 0.0,
        solution_collection_mode: str = "first",
        n_solutions: int = 1,
        final_solution_selection: str = "fastest",
        selection_mode: str = "weighted",
        weighted_selection_mode: str = "softmax",
        gamma_mode: str = "fixed",
        gamma_fixed: float = 0.125,
        gamma_interval: tuple[float, float] = (0.0, 0.25),
        gamma_random_sign_flip_prob: float = 0.0,
        adaptive_memory: bool = False,
        adaptive_memory_zeta: float = 1.0,
        adaptive_memory_adjacent_gamma_interval: tuple[float, float] = (0.0, 0.25),
        adaptive_memory_mode: str = "probabilistic_flip",
        biased_relay_mode: bool = False,
        switch_relay_leg: int | None = None,
        biased_relay_r_relay: int = 10,
        biased_relay_maximum_round: int = 10,
        biased_relay_t_0: int = 80,
        biased_relay_r_relay_iter: int = 60,
        relay_gamma_interval: tuple[float, float] = (0.0, 0.25),
        tournament_size: int = 3,
        elite_count: int = 2,
        seed: int = 0,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        self.alpha = alpha
        self.ensemble_size = ensemble_size
        self.t_ms = t_ms
        self.t_mem = t_mem
        self.g_max = g_max
        self.sigma2 = sigma2
        self.delta = delta
        self.fitness_alpha = fitness_alpha
        self.fitness_beta = fitness_beta
        self.fitness_low_llr_mu = fitness_low_llr_mu
        self.fitness_low_llr_threshold = fitness_low_llr_threshold
        self.eta = eta
        self.mutation_rate = mutation_rate
        self.mutation_llr_abs_threshold = mutation_llr_abs_threshold
        self.sequential_mc = sequential_mc
        self.init_perturbation_mode = init_perturbation_mode
        self.init_strategy = init_strategy
        self.init_gamma = init_gamma
        self.initial_prior_bias = initial_prior_bias
        self.adaptive_perturbation = adaptive_perturbation
        self.adaptive_perturbation_llr_threshold_mode = (
            adaptive_perturbation_llr_threshold_mode
        )
        self.adaptive_perturbation_llr_threshold = adaptive_perturbation_llr_threshold
        self.adaptive_perturbation_llr_threshold_min = (
            adaptive_perturbation_llr_threshold_min
        )
        self.adaptive_perturbation_llr_threshold_max = (
            adaptive_perturbation_llr_threshold_max
        )
        self.adaptive_perturbation_llr_threshold_factor = (
            adaptive_perturbation_llr_threshold_factor
        )
        self.adaptive_perturbation_factor = adaptive_perturbation_factor
        self.adaptive_perturbation_factor_mode = adaptive_perturbation_factor_mode
        self.adaptive_perturbation_factor_interval = tuple(
            adaptive_perturbation_factor_interval
        )
        self.adaptive_perturbation_bias_mode = adaptive_perturbation_bias_mode
        self.adaptive_perturbation_sign_mode = adaptive_perturbation_sign_mode
        if not 0.0 <= adaptive_perturbation_positive_sign_prob <= 1.0:
            raise ValueError(
                "adaptive_perturbation_positive_sign_prob must be between 0.0 and 1.0"
            )
        self.adaptive_perturbation_positive_sign_prob = (
            adaptive_perturbation_positive_sign_prob
        )
        self.adaptive_perturbation_target = adaptive_perturbation_target
        self.adaptive_perturbation_prior_base_mode = (
            adaptive_perturbation_prior_base_mode
        )
        self.adaptive_perturbation_variable_base_mode = (
            adaptive_perturbation_variable_base_mode
        )
        self.adaptive_perturbation_reset_on_threshold_exit = (
            adaptive_perturbation_reset_on_threshold_exit
        )
        self.continue_perturbation = continue_perturbation
        self.perturbation_method = perturbation_method
        self.reset_marginal = reset_marginal
        if not 0.0 <= marginal_carry_damping_factor <= 1.0:
            raise ValueError(
                "marginal_carry_damping_factor must be between 0.0 and 1.0"
            )
        if (
            marginal_carry_llr_abs_threshold < 0.0
            and marginal_carry_llr_abs_threshold != -1.0
        ):
            raise ValueError(
                "marginal_carry_llr_abs_threshold must be -1.0 (all variables) or >= 0.0"
            )
        self.marginal_carry_damping_factor = marginal_carry_damping_factor
        self.marginal_carry_llr_abs_threshold = marginal_carry_llr_abs_threshold
        self.drop_p = drop_p
        self.drop_llr_threshold = drop_llr_threshold
        self.solution_collection_mode = solution_collection_mode
        if n_solutions < 1:
            raise ValueError("n_solutions must be >= 1")
        self.n_solutions = n_solutions
        self.final_solution_selection = final_solution_selection
        self.selection_mode = selection_mode
        self.weighted_selection_mode = weighted_selection_mode
        self.gamma_mode = gamma_mode
        self.gamma_fixed = gamma_fixed
        self.gamma_interval = tuple(gamma_interval)
        if not 0.0 <= gamma_random_sign_flip_prob <= 1.0:
            raise ValueError("gamma_random_sign_flip_prob must be between 0.0 and 1.0")
        self.gamma_random_sign_flip_prob = gamma_random_sign_flip_prob
        self.adaptive_memory = adaptive_memory
        self.adaptive_memory_zeta = adaptive_memory_zeta
        self.adaptive_memory_adjacent_gamma_interval = tuple(
            adaptive_memory_adjacent_gamma_interval
        )
        self.adaptive_memory_mode = adaptive_memory_mode
        self.biased_relay_mode = biased_relay_mode
        self.switch_relay_leg = switch_relay_leg
        self.biased_relay_r_relay = biased_relay_r_relay
        self.biased_relay_maximum_round = biased_relay_maximum_round
        self.biased_relay_t_0 = biased_relay_t_0
        self.biased_relay_r_relay_iter = biased_relay_r_relay_iter
        self.relay_gamma_interval = tuple(relay_gamma_interval)
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.seed = seed
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        decoder = relay_bp.SLGMBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            alpha=None if self.alpha == 0.0 else self.alpha,
            ensemble_size=self.ensemble_size,
            t_ms=self.t_ms,
            t_mem=self.t_mem,
            g_max=self.g_max,
            sigma2=self.sigma2,
            delta=self.delta,
            fitness_alpha=self.fitness_alpha,
            fitness_beta=self.fitness_beta,
            fitness_low_llr_mu=self.fitness_low_llr_mu,
            fitness_low_llr_threshold=self.fitness_low_llr_threshold,
            eta=self.eta,
            mutation_rate=self.mutation_rate,
            mutation_llr_abs_threshold=self.mutation_llr_abs_threshold,
            sequential_mc=self.sequential_mc,
            init_perturbation_mode=self.init_perturbation_mode,
            init_strategy=self.init_strategy,
            init_gamma=self.init_gamma,
            initial_prior_bias=self.initial_prior_bias,
            adaptive_perturbation=self.adaptive_perturbation,
            adaptive_perturbation_llr_threshold_mode=self.adaptive_perturbation_llr_threshold_mode,
            adaptive_perturbation_llr_threshold=self.adaptive_perturbation_llr_threshold,
            adaptive_perturbation_llr_threshold_min=self.adaptive_perturbation_llr_threshold_min,
            adaptive_perturbation_llr_threshold_max=self.adaptive_perturbation_llr_threshold_max,
            adaptive_perturbation_llr_threshold_factor=self.adaptive_perturbation_llr_threshold_factor,
            adaptive_perturbation_factor=self.adaptive_perturbation_factor,
            adaptive_perturbation_factor_mode=self.adaptive_perturbation_factor_mode,
            adaptive_perturbation_factor_interval=self.adaptive_perturbation_factor_interval,
            adaptive_perturbation_bias_mode=self.adaptive_perturbation_bias_mode,
            adaptive_perturbation_sign_mode=self.adaptive_perturbation_sign_mode,
            adaptive_perturbation_positive_sign_prob=self.adaptive_perturbation_positive_sign_prob,
            adaptive_perturbation_target=self.adaptive_perturbation_target,
            adaptive_perturbation_prior_base_mode=self.adaptive_perturbation_prior_base_mode,
            adaptive_perturbation_variable_base_mode=self.adaptive_perturbation_variable_base_mode,
            adaptive_perturbation_reset_on_threshold_exit=self.adaptive_perturbation_reset_on_threshold_exit,
            continue_perturbation=self.continue_perturbation,
            perturbation_method=self.perturbation_method,
            reset_marginal=self.reset_marginal,
            marginal_carry_damping_factor=self.marginal_carry_damping_factor,
            marginal_carry_llr_abs_threshold=self.marginal_carry_llr_abs_threshold,
            drop_p=self.drop_p,
            drop_llr_threshold=self.drop_llr_threshold,
            solution_collection_mode=self.solution_collection_mode,
            n_solutions=self.n_solutions,
            final_solution_selection=self.final_solution_selection,
            selection_mode=self.selection_mode,
            weighted_selection_mode=self.weighted_selection_mode,
            gamma_mode=self.gamma_mode,
            gamma_fixed=self.gamma_fixed,
            gamma_interval=self.gamma_interval,
            gamma_random_sign_flip_prob=self.gamma_random_sign_flip_prob,
            adaptive_memory=self.adaptive_memory,
            adaptive_memory_zeta=self.adaptive_memory_zeta,
            adaptive_memory_adjacent_gamma_interval=self.adaptive_memory_adjacent_gamma_interval,
            adaptive_memory_mode=self.adaptive_memory_mode,
            biased_relay_mode=self.biased_relay_mode,
            switch_relay_leg=self.switch_relay_leg,
            biased_relay_r_relay=self.biased_relay_r_relay,
            biased_relay_maximum_round=self.biased_relay_maximum_round,
            biased_relay_t_0=self.biased_relay_t_0,
            biased_relay_r_relay_iter=self.biased_relay_r_relay_iter,
            relay_gamma_interval=self.relay_gamma_interval,
            tournament_size=self.tournament_size,
            elite_count=self.elite_count,
            seed=self.seed,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


class SinterDecoder_MSLBP(SinterDecoder_BaseBP):

    def __init__(
        self,
        max_iter: int = 100,
        alpha: float | None = None,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        collect_iteration_metric: bool = False,
    ):
        f"""Class for decoding stim circuits with sinter and relay-bp."""
        self.max_iter = max_iter
        self.alpha = alpha
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            collect_iteration_metric=collect_iteration_metric,
        )

    def build_observable_decoder(
        self,
        check_matrices: CheckMatrices,
    ) -> relay_bp.ObservableDecoderRunner:

        decoder = relay_bp.MinSumBPDecoderF64(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=self.max_iter,
            alpha=None if self.alpha == 0.0 else self.alpha,
            gamma0=None,
        )

        observable_decoder = relay_bp.ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=False,
        )
        return observable_decoder


def sinter_decoders(
    selected_decoders: Iterable[str] | None = None,
    **decoder_kwargs: dict,
) -> dict[str, Decoder]:
    relay_config = decoder_kwargs.copy()
    adaptive_relay_config = decoder_kwargs.copy()
    lrbp_config = decoder_kwargs.copy()
    slg_mbp_config = decoder_kwargs.copy()
    disordered_bp_config = decoder_kwargs.copy()
    dual_relay_config = decoder_kwargs.copy()

    lrbp_only_keys = [
        "odd_leg_max_iter",
        "even_leg_uniform_gamma",
        "osc_window",
        "friction_slope",
        "friction_shift",
        "tau",
        "seed",
        "r_dyn",
        "t_dyn",
        "dyn_mode",
        "gamma_dyn_penalty",
        "dyn_gamma_mode",
        "gamma_dyn_center",
        "gamma_dyn_min",
        "gamma_dyn_max",
    ]
    for key in lrbp_only_keys:
        relay_config.pop(key, None)
        adaptive_relay_config.pop(key, None)
        disordered_bp_config.pop(key, None)

    slg_mbp_only_keys = [
        "ensemble_size",
        "t_ms",
        "t_mem",
        "g_max",
        "sigma2",
        "delta",
        "fitness_alpha",
        "fitness_beta",
        "eta",
        "mutation_rate",
        "mutation_llr_abs_threshold",
        "sequential_mc",
        "init_perturbation_mode",
        "init_strategy",
        "init_gamma",
        "continue_perturbation",
        "perturbation_method",
        "solution_collection_mode",
        "n_solutions",
        "final_solution_selection",
        "selection_mode",
        "weighted_selection_mode",
        "gamma_mode",
        "gamma_fixed",
        "gamma_interval",
        "tournament_size",
        "elite_count",
        "seed",
    ]
    for key in slg_mbp_only_keys:
        relay_config.pop(key, None)
        adaptive_relay_config.pop(key, None)
        lrbp_config.pop(key, None)
        disordered_bp_config.pop(key, None)
        dual_relay_config.pop(key, None)

    relay_only_keys = [
        "gamma0",
        "pre_iter",
        "num_sets",
        "set_max_iter",
        "gamma_dist_interval",
        "explicit_gammas",
        "stop_nconv",
        "stopping_criterion",
        "logging",
    ]
    for key in relay_only_keys:
        adaptive_relay_config.pop(key, None)
        disordered_bp_config.pop(key, None)
        dual_relay_config.pop(key, None)

    adaptive_relay_only_keys = [
        "initial_gamma",
        "gamma_min",
        "gamma_max",
        "tau",
        "beta",
        "pre_decoding",
        "pre_iteration",
        "maximum_iteration",
        "iter_per_leg",
        "update_mode",
        "perturbation_mode",
        "perturbation_interval",
        "perturbation_sigma",
        "ensemble_size",
        "carry_marginal_between_legs",
        "posterior_marginal_clamp_mode",
        "posterior_marginal_abs_threshold",
    ]
    for key in adaptive_relay_only_keys:
        relay_config.pop(key, None)
        lrbp_config.pop(key, None)
        slg_mbp_config.pop(key, None)
        disordered_bp_config.pop(key, None)
        dual_relay_config.pop(key, None)

    disordered_bp_only_keys = [
        "t_0",
        "maximum_leg",
        "iteration_per_leg",
        "initial_alpha",
        "alpha_mode",
        "alpha_fixed",
        "alpha_interval",
        "initial_gamma",
        "gamma_mode",
        "gamma_fixed",
        "gamma_interval",
        "bias_mode",
        "bias_fixed",
        "bias_interval",
        "bias_apply_mode",
        "bias_filter_threshold",
        "negative_sign_prob",
        "carry_marginal_factor",
    ]
    for key in disordered_bp_only_keys:
        relay_config.pop(key, None)
        adaptive_relay_config.pop(key, None)
        lrbp_config.pop(key, None)
        slg_mbp_config.pop(key, None)
        dual_relay_config.pop(key, None)

    dual_relay_only_keys = [
        "maximum_leg",
        "iteration_per_leg",
        "initial_gamma_slow",
        "initial_gamma_fast",
        "gamma_interval_slow",
        "gamma_interval_fast",
        "mix_mode",
        "eta",
        "delta",
        "use_previous_message",
        "beta",
        "ensemble_mode",
        "ensemble_size",
        "ensemble_gamma_interval",
        "num_pre_iteration_instance",
        "initial_gamma",
        "n_solutions",
    ]
    for key in dual_relay_only_keys:
        relay_config.pop(key, None)
        adaptive_relay_config.pop(key, None)
        lrbp_config.pop(key, None)
        slg_mbp_config.pop(key, None)
        disordered_bp_config.pop(key, None)

    msl_config = {}

    if max_iter := decoder_kwargs.get("max_iter"):
        msl_config["max_iter"] = max_iter
        relay_config.pop("max_iter", None)
        lrbp_config.pop("max_iter", None)

    if alpha := decoder_kwargs.get("alpha"):
        msl_config["alpha"] = alpha

    membp_config = msl_config.copy()

    if gamma0 := decoder_kwargs.get("gamma0"):
        membp_config["gamma0"] = gamma0

    lbf_config = {
        k: decoder_kwargs[k]
        for k in (
            "max_iter",
            "weight",
            "k_step",
            "parallel",
            "decomposed_hyperedges",
            "prune_decided_errors",
            "threshold",
            "collect_iteration_metric",
        )
        if k in decoder_kwargs
    }

    decoders = {
        "relay-bp": SinterDecoder_RelayBP(**relay_config),  # type: ignore
        "adaptive-relay": SinterDecoder_AdaptiveRelay(**adaptive_relay_config),  # type: ignore
        "disordered-bp": SinterDecoder_DisorderedBP(**disordered_bp_config),  # type: ignore
        "dual-relay": SinterDecoder_DualRelay(**dual_relay_config),  # type: ignore
        "lr-bp": SinterDecoder_LRBP(**lrbp_config),  # type: ignore
        "slg-mbp": SinterDecoder_SLGMBP(**slg_mbp_config),  # type: ignore
        "mem-bp": SinterDecoder_MemBP(**membp_config),  # type: ignore
        "msl-bp": SinterDecoder_MSLBP(**msl_config),  # type: ignore
        "lbf": SinterDecoder_LBF(**lbf_config),  # type: ignore
    }

    if selected_decoders is None:
        return decoders

    selected = tuple(selected_decoders)
    unknown = sorted(set(selected) - set(decoders))
    if unknown:
        raise ValueError(
            "Unknown decoder(s) requested in selected_decoders: "
            + ", ".join(unknown)
        )

    return {name: decoders[name] for name in selected}
