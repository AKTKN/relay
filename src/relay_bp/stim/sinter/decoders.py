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
        eta: float = 1.0,
        mutation_rate: float = 0.02,
        mutation_llr_abs_threshold: float = 0.25,
        init_perturbation_mode: str = "gaussian",
        init_strategy: str = "min-sum",
        init_gamma: float = 0.125,
        selection_mode: str = "weighted",
        weighted_selection_mode: str = "softmax",
        gamma_mode: str = "fixed",
        gamma_fixed: float = 0.125,
        gamma_interval: tuple[float, float] = (0.0, 0.25),
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
        self.eta = eta
        self.mutation_rate = mutation_rate
        self.mutation_llr_abs_threshold = mutation_llr_abs_threshold
        self.init_perturbation_mode = init_perturbation_mode
        self.init_strategy = init_strategy
        self.init_gamma = init_gamma
        self.selection_mode = selection_mode
        self.weighted_selection_mode = weighted_selection_mode
        self.gamma_mode = gamma_mode
        self.gamma_fixed = gamma_fixed
        self.gamma_interval = tuple(gamma_interval)
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
            eta=self.eta,
            mutation_rate=self.mutation_rate,
            mutation_llr_abs_threshold=self.mutation_llr_abs_threshold,
            init_perturbation_mode=self.init_perturbation_mode,
            init_strategy=self.init_strategy,
            init_gamma=self.init_gamma,
            selection_mode=self.selection_mode,
            weighted_selection_mode=self.weighted_selection_mode,
            gamma_mode=self.gamma_mode,
            gamma_fixed=self.gamma_fixed,
            gamma_interval=self.gamma_interval,
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
    lrbp_config = decoder_kwargs.copy()
    slg_mbp_config = decoder_kwargs.copy()

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
        "init_perturbation_mode",
        "init_strategy",
        "init_gamma",
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
        lrbp_config.pop(key, None)

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

    decoders = {
        "relay-bp": SinterDecoder_RelayBP(**relay_config),  # type: ignore
        "lr-bp": SinterDecoder_LRBP(**lrbp_config),  # type: ignore
        "slg-mbp": SinterDecoder_SLGMBP(**slg_mbp_config),  # type: ignore
        "mem-bp": SinterDecoder_MemBP(**membp_config),  # type: ignore
        "msl-bp": SinterDecoder_MSLBP(**msl_config),  # type: ignore
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
