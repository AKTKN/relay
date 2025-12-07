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

from sinter import Decoder, CompiledDecoder
import numpy as np
import numpy.typing as npt

import scipy.sparse as sparse
from autdec.igraph_auts import random_vertex_graph_auts_from_bliss

import stim

import relay_bp

from .check_matrices import CheckMatrices

from typing import TYPE_CHECKING, Optional

from ldpc.sinter_decoders import SinterBpOsdDecoder
from ldpc.sinter_decoders.sinter_lsd_decoder import SinterLsdDecoder
from tesseract_decoder import make_tesseract_sinter_decoders_dict, TesseractSinterDecoder
import tesseract_decoder

class SinterCompiledDecoder_BP(CompiledDecoder):
    def __init__(
        self,
        observable_decoder: relay_bp.ObservableDecoderRunner,
        check_matrices: CheckMatrices,
        parallel: bool = False,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        get_detail: bool = False,
        save_detail_path: Optional[pathlib.Path] = None,
    ):
        self.observable_decoder = observable_decoder
        self.parallel = parallel
        self.check_matrices = check_matrices
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish
        self.get_detail = get_detail
        
        # 詳細情報を蓄積するためのバッファ
        self.accumulated_details = {
            'converged': [],
            'logical_gaps': [],
            'selected_coset_avg_iter': [],
            'runner_up_coset_avg_iter': [],
            'selected_coset_votes': [],
            'runner_up_coset_votes': [],
        }

    def _save_accumulated_details(self):
        if self.save_detail_path is None or not self.accumulated_details['converged']:
            return
        
        # リストをNumPy配列に変換
        data_to_save = {}
        for key, values in self.accumulated_details.items():
            # Noneを含む可能性があるため、object型配列として保存
            data_to_save[key] = np.array(values, dtype=object)
        
        # NPZ形式で圧縮保存
        self.save_detail_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.save_detail_path, **data_to_save)
        print(f"Detailed metrics saved to: {self.save_detail_path}")


    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: "np.ndarray",
    ) -> "np.ndarray":
        syndromes = np.unpackbits(
            bit_packed_detection_event_data, bitorder="little", axis=1
        ).astype(np.uint8)

        if self.check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + self.check_matrices.syndrome_bias) % 2

        # In harmonized decoder, it calculates permutation of syndromes. 
        # At that time, the dimension of checks and syndromes must be the same. 
        # In decode_shots_bit_packed, the syndrome data is bit-packed, so its shape is a multiple of 8 (byte).
        # Therefore, when the dimension of check matrix (row) is not a multiple of 8, 
        # we need to slice the syndrome data to match the dimension.
        num_checks = self.check_matrices.check_matrix.shape[0]
        if syndromes.shape[1] > num_checks:
            syndromes = syndromes[:, :num_checks]

        iterations = None
        converged = None
        logical_gaps = None
        iter_deltas = None  
        vote_deltas = None

        if self.get_detail:
            results = self.observable_decoder.decode_observables_detailed_batch(
                syndromes,
                parallel=self.parallel,
                progress_bar=self.show_progress,
                leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            )
            predictions = np.array([res.observables for res in results])
            converged = np.array([res.converged for res in results])
            logical_gaps = np.array([res.logical_gap for res in results])

            # 差分メトリックの計算
            iter_deltas_list = []
            vote_deltas_list = []

            for res in results:
                extra = res.extra
                if extra is not None:
                    selected_iter = extra.get('selected_coset_avg_iter')
                    runner_up_iter = extra.get('runner_up_coset_avg_iter')
                    selected_votes = extra.get('selected_coset_votes')
                    runner_up_votes = extra.get('runner_up_coset_votes')

                    # print(f"Debug: selected_iter={selected_iter}, runner_up_iter={runner_up_iter}, selected_votes={selected_votes}, runner_up_votes={runner_up_votes}")
                    
                    # 差分を計算 (runner_up - selected)
                    if runner_up_iter is not None and selected_iter is not None:
                        iter_delta = runner_up_iter - selected_iter
                    else:
                        iter_delta = None
                    
                    if runner_up_votes is not None and selected_votes is not None:
                        vote_delta = runner_up_votes - selected_votes
                    else:
                        vote_delta = None
                    
                    iter_deltas_list.append(iter_delta)
                    vote_deltas_list.append(vote_delta)
                else:
                    # print("Warning: Extra result is None, cannot compute deltas.")
                    iter_deltas_list.append(None)
                    vote_deltas_list.append(None)

            # NumPy配列に変換（Noneを含むためobject型）
            iter_deltas = np.array(iter_deltas_list, dtype=object)
            vote_deltas = np.array(vote_deltas_list, dtype=object)

            # Try to use effective_iterations from ensemble extra if available
            extra = results[0].extra
            if extra is not None and extra.get("effective_iterations") is not None:
                # print(f"Debug: Using effective_iterations from ensemble extra")
                # print(f"Debug: Sample effective_iterations values (first 5): {[res.extra.get('effective_iterations') if res.extra is not None else None for res in results[:5]]}")
                # Use effective_iterations from all results
                iterations = np.array([
                    res.extra["effective_iterations"] if res.extra is not None else float(res.iterations)
                    for res in results
                ], dtype=float)
            else:
                # Fallback to regular iterations with inf for non-converged
                iterations = np.array([res.iterations for res in results], dtype=float) # return type from rust is int, so we need to convert to float.
                iterations[~converged] = np.inf
                
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
        if self.get_detail:
            return outputs, iterations, converged, logical_gaps, iter_deltas, vote_deltas
        else:
            return outputs

    def __del__(self):
        """デストラクタで蓄積したデータを保存"""
        if hasattr(self, 'save_detail_path') and self.save_detail_path is not None:
            self._save_accumulated_details()


class SinterDecoder_BaseBP(Decoder):
    def __init__(
        self,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        get_detail_result: bool = False,
    ):
        f"""Class for decoding stim circuits with sinter and relay-bp."""
        self.parallel = parallel
        self.decomposed_hyperedges = decomposed_hyperedges
        self.prune_decided_errors = prune_decided_errors
        self.threshold = threshold
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish
        self.get_detail_result = get_detail_result

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
            get_detail=self.get_detail_result,
        )


    # relay-basedではこれは使われない。osdやlsdとの互換性の都合上、iterations_out_pathは現状コメントアウト。もしこれを考慮する場合、例外処理、及びOptional等でsinterを修正する必要がある。
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
        # iterations_out_path: Optional[pathlib.Path] = None, 
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



        if self.get_detail_result:
            results = observable_decoder.decode_observables_detailed_batch(
                syndromes,
                parallel=self.parallel,
                progress_bar=self.show_progress,
                leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
            )
            predictions = np.array([res.observables for res in results])
            iterations = np.array([res.iterations for res in results])

            # if iterations_out_path is not None:
            #     with open(iterations_out_path, 'wb') as f:
            #         iterations.tofile(f)
            #     print(f"Debug: iterations saved to {iterations_out_path}")
            # else:
            #     raise ValueError("iterations_out_path must be provided when get_detail_result is True")

        else:
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

class SinterDecoder_HarmonizedBP(SinterDecoder_BaseBP):
    def __init__(
        self,
        # --- relay-BP parameters ---
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
        seed:Optional[int] = None,        
        # --- New parameters for harmonization ---
        ensemble_size: int = 1,
        selection_strategy: str = "MostLikely",
        perturbation_min: float = 0.0,
        perturbation_max: float = 0.0,
        # --- For automorphism ---
        use_automorphism: bool = False,
        # --- For repulsive mode ---
        ensemble_mode = "normal",
        repulsive_size: int = 0,
        repulsive_gamma_dist: tuple[float, float] = None,
        abs_llr_threshold: float = None,
        pulse_per_leg: int = None,
        start_leg: int = None,
        # --- BaseBP parameters ---
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        get_detail: bool = False,    # if True, return detailed decoding info
    ):
        """Class for decoding stim circuits with the harmonized BP ensemble decoder."""
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
        self.ensemble_size = ensemble_size
        self.selection_strategy = selection_strategy
        self.perturbation_min = perturbation_min
        self.perturbation_max = perturbation_max
        self.use_automorphism = use_automorphism
        self.ensemble_mode = ensemble_mode
        self.repulsive_size = repulsive_size
        self.repulsive_gamma_dist = tuple(repulsive_gamma_dist) 
        self.abs_llr_threshold = abs_llr_threshold
        self.pulse_per_leg = pulse_per_leg
        self.start_leg = start_leg
        self.seed = np.random.randint(0, 2**32 - 1) if seed is None else seed
        self.get_detail = get_detail

        # 親クラスの__init__を呼び出す
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            get_detail_result=get_detail
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:
        
            
        col_perms = None
        row_perms = None
        current_ensemble_size = self.ensemble_size

        if self.use_automorphism:
            assert self.ensemble_size >= 1, "Warning: ensemble_size should be at least 1 when using automorphism. Used identity."

            col_perms = []
            row_perms = []

            if self.ensemble_size >= 2:
                bliss_cols, bliss_rows = random_vertex_graph_auts_from_bliss(
                    check_matrices.check_matrix, k=self.ensemble_size-1 # except identity
                )
                col_perms.extend(bliss_cols)
                row_perms.extend(bliss_rows)

            identity_col = sparse.identity(check_matrices.check_matrix.shape[1], dtype=int, format='csr')
            identity_row = sparse.identity(check_matrices.check_matrix.shape[0], dtype=int, format='csr')
            col_perms.insert(0, identity_col)
            row_perms.insert(0, identity_row)
            
            print(f"Debug: Found {len(col_perms)} automorphisms using bliss.")
            current_ensemble_size = len(col_perms)
            if current_ensemble_size < self.ensemble_size:
                print(f"Warning: Only found {current_ensemble_size} automorphisms, which is less than the requested ensemble_size of {self.ensemble_size}. Using {current_ensemble_size} instead.")
                self.ensemble_size = current_ensemble_size

        self.col_permutations = col_perms
        self.row_permutations = row_perms
    
        observable_decoder = relay_bp.ObservableDecoderRunner.with_ensemble_decoder(
            ensemble_size=self.ensemble_size,
            check_matrix=check_matrices.check_matrix,
            observable_matrix=check_matrices.observables_matrix,
            error_priors=check_matrices.error_priors,
            alpha=self.alpha,
            gamma0=self.gamma0,
            pre_iter=self.pre_iter,
            num_sets=self.num_sets,
            set_max_iter=self.set_max_iter,
            gamma_dist_interval=self.gamma_dist_interval,
            explicit_gammas=self.explicit_gammas,
            stop_nconv=self.stop_nconv,
            stopping_criterion=self.stopping_criterion,
            logging=self.logging,
            selection_strategy=self.selection_strategy,
            perturbation_min=self.perturbation_min,
            perturbation_max=self.perturbation_max,
            col_permutations=self.col_permutations,
            row_permutations=self.row_permutations,
            seed=self.seed,
            ensemble_mode=self.ensemble_mode,
            repulsive_size=self.repulsive_size,
            repulsive_gamma_dist=self.repulsive_gamma_dist,
            abs_llr_threshold=self.abs_llr_threshold,
            pulse_per_leg=self.pulse_per_leg,
            start_leg=self.start_leg,
        )


        return observable_decoder
    
    
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
        get_detail: bool = False,    # if True, return detailed decoding info
        seed: Optional[int] = None,
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
        self.get_detail = get_detail
        self.seed = np.random.randint(0, 2**32 - 1) if seed is None else seed
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            get_detail_result=get_detail
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
            seed=self.seed, 
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


class SinterDecoder_MSLBP(SinterDecoder_BaseBP):

    def __init__(
        self,
        max_iter: int = 100,
        alpha: float | None = None,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
    ):
        f"""Class for decoding stim circuits with sinter and relay-bp."""
        self.max_iter = max_iter
        self.alpha = alpha
        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
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


def sinter_decoders(**decoder_kwargs: dict) -> dict[str, Decoder]:
    msl_config = {}

    if max_iter := decoder_kwargs.get("max_iter"):
        msl_config["max_iter"] = max_iter
        decoder_kwargs.pop("max_iter", None)

    if alpha := decoder_kwargs.get("alpha"):
        msl_config["alpha"] = alpha

    membp_config = msl_config.copy()

    if gamma0 := decoder_kwargs.get("gamma0"):
        membp_config["gamma0"] = gamma0

    harmonized_config = decoder_kwargs.copy()

    relay_config = decoder_kwargs.copy()
    
    relay_config.pop("ensemble_size", None)
    relay_config.pop("selection_strategy", None)
    relay_config.pop("perturbation_min", None)
    relay_config.pop("perturbation_max", None)
    relay_config.pop("use_automorphism", None)
    relay_config.pop("ensemble_mode", None)
    relay_config.pop("repulsive_size", None)
    relay_config.pop("repulsive_gamma_dist", None)
    relay_config.pop("abs_llr_threshold", None)
    relay_config.pop("pulse_per_leg", None)
    relay_config.pop("start_leg", None)
    # relay_config.pop("seed", None)

    return {
        # 修正：relay_config を使用する
        "relay-bp": SinterDecoder_RelayBP(**relay_config),  # type: ignore
        "mem-bp": SinterDecoder_MemBP(**membp_config),  # type: ignore
        "msl-bp": SinterDecoder_MSLBP(**msl_config),  # type: ignore
        "harmonized-bp": SinterDecoder_HarmonizedBP(**harmonized_config),  # type: ignore
    }


def build_decoders(decoder_specs: list[dict]) -> dict[str, Decoder]:
    """
    decoder_specs: [{name: str, params: dict}]
    Supports base relay/harmonized/mem/msl plus extra: bplsd, bposd.
    """
    built: dict[str, Decoder] = {}
    for spec in decoder_specs:
        name = spec.get("name")
        params = spec.get("params", {}) or {}
        if name in {"relay-bp", "mem-bp", "msl-bp", "harmonized-bp"}:
            # Reuse sinter_decoders factory (single extraction)
            built[name] = sinter_decoders(**params)[name]
        elif name == "bposd":
            built[name] = SinterBpOsdDecoder(**params)
        elif name == "bplsd":
            if SinterLsdDecoder is None:
                raise ImportError("SinterLsdDecoder unavailable.")
            built[name] = SinterLsdDecoder(**params)
        elif name in ['tesseract', 'tesseract-long-beam', 'tesseract-short-beam']:
            tesseract_decoders_dict = make_tesseract_sinter_decoders_dict() # ccurrently, custom parameters for terrerasct are not supported.
            built[name] = tesseract_decoders_dict[name]
            
        else:
            raise ValueError(f"Unknown decoder name: {name}")
    return built

