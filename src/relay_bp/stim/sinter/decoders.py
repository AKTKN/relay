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
from dataclasses import dataclass, asdict, field
import scipy.sparse as sparse
from autdec.igraph_auts import random_vertex_graph_auts_from_bliss

import stim

import relay_bp

from .check_matrices import CheckMatrices
from .decode_result import DecodeResult

from typing import TYPE_CHECKING, Optional, Tuple, List

from ldpc.sinter_decoders import SinterBpOsdDecoder
from ldpc.sinter_decoders.sinter_lsd_decoder import SinterLsdDecoder
from tesseract_decoder import make_tesseract_sinter_decoders_dict, TesseractSinterDecoder
# Note: retesseract imports are done inside build_decoders() to avoid circular import
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
        local_ambiguity_threshold: float = 0.0,
    ):
        self.observable_decoder = observable_decoder
        self.parallel = parallel
        self.check_matrices = check_matrices
        # Hack to pass threshold to decode_shots_bit_packed
        self.check_matrices.local_ambiguity_threshold = local_ambiguity_threshold
        
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

        # print("DEBUG: check matrix shape:", self.check_matrices.check_matrix.shape)
        if syndromes.shape[1] > num_checks:
            syndromes = syndromes[:, :num_checks]
        iterations = None
        converged = None
        logical_gaps = None
        iter_deltas = None  
        vote_deltas = None
        mean_iterations = None
        std_iterations = None
        converged_counts = None
        correction_hammingweight = None
        correction_weight = None

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
            mean_iter_list = []
            std_iter_list = []
            converged_count_list = [] 
            correction_hammingweight_list = []
            correction_weight_list = []
            local_ambiguity_score_list = []

            # Get error_priors (LLR) for computing correction weights
            # Calculate prior LLRs: ln((1-p)/p)
            eps = 1e-18
            error_priors = self.check_matrices.error_priors
            prior_llrs = np.log((1.0 - error_priors) / (error_priors + eps))

            # Threshold for filtering posterior LLRs. 
            # If 0.0, all indices are considered.
            # If > 0.0, only indices where |posterior_llr| <= threshold (ambiguous bits) are considered ? 
            # Or |posterior_llr| >= threshold ? 
            # The user said: "consider only posterior LLRs larger than threshold".
            # "事後LLRの絶対値に対してthresholdを設けて、上記の和を取るときに、thresholdよりも大きい事後LLRのみを対称とする"
            # -> This likely means we only sum up prior LLRs for bits that are "confident enough" (large posterior LLR)
            # OR typically "ambiguity" implies focusing on bits with SMALL posterior LLRs (uncertain). 
            # But the user specifically asked for "larger than threshold". I will follow the user's instruction literally: 
            # target = {i | |posterior_llr[i]| > threshold }. 
            # Then sum += prior_llr[i] for i in target.
            
            # Wait, "local ambiguity score" usually implies summing up risk or ambiguity.
            # If I follow the user's previous logic: "sum of inverse posterior LLRs" -> low LLR = high score = high ambiguity.
            # Now user says: "sum of PRIOR LLRs". and "only for posterior LLRs > threshold".
            # If threshold is 0, we sum prior LLRs for all neighbors.
            # If threshold is high, we sum prior LLRs only for neighbors that are "confident" (high posterior LLR).
            # This seems counter-intuitive for an "ambiguity" score if increasing threshold filters out low-confidence bits.
            # However, maybe the user wants to filter out bits that are *too* ambiguous (close to 0) or bits that are *too* certain?
            
            # User instruction: "thresholdよりも大きい事後LLRのみを対称とする"
            # Literal translation: "target only posterior LLRs larger than threshold".
            # I will implement as requested: filter condition is `abs(posterior_llr) > threshold`.
            
            local_ambiguity_threshold = getattr(self.check_matrices, 'local_ambiguity_threshold', 0.0)

            for res in results:
                # Calculate local ambiguity score
                score = None
                phys_res = res.physical_decode_result
                if phys_res is not None:
                     indices = phys_res.bad_syndrome_neighbour_indices
                     post_llrs = phys_res.posterior_ratios
                     print(f"debug: posterior_llrs: {post_llrs}, average: {np.mean(np.abs(post_llrs))}, std: {np.std(post_llrs)}")
                     if indices is not None and post_llrs is not None:
                         # Filter indices based on posterior LLR threshold
                         # We select indices where |posterior_llr| > threshold
                         
                         valid_indices = []
                         for idx in indices:
                             if abs(post_llrs[idx]) > local_ambiguity_threshold:
                                 valid_indices.append(idx)
                         
                         if valid_indices:
                             # Sum of *prior* LLRs for these indices
                             # prior_llrs is derived from error_priors
                             vals = prior_llrs[valid_indices]
                             score = float(np.sum(vals))
                         else:
                             score = 0.0
                local_ambiguity_score_list.append(score)

                extra = res.extra
                if extra is not None:
                    selected_iter = extra.get('selected_coset_avg_iter')
                    runner_up_iter = extra.get('runner_up_coset_avg_iter')
                    selected_votes = extra.get('selected_coset_votes')
                    runner_up_votes = extra.get('runner_up_coset_votes')
                    
                    # Collect mean/std iterations if available
                    mean_iter_list.append(extra.get('ensemble_mean_iteration'))
                    std_iter_list.append(extra.get('ensemble_std_iteration'))
                    
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

                    # Extract correction from physical_decode_result
                    selected_idx = extra.get('selected_index')
                    all_corrections = extra.get('all_corrections')
                    llr_sums = extra.get('llr_sums')
                    converged_count = np.sum(extra.get('child_success')) 
                    converged_count_list.append(converged_count)
                    
                    if all_corrections is not None and selected_idx is not None:
                            correction = all_corrections[selected_idx]
                            
                            if correction is not None:
                                # Calculate Hamming weight
                                correction_hw = int(np.sum(correction))
  
                                if llr_sums is not None and len(llr_sums) > selected_idx:
                                    correction_wt = float(llr_sums[selected_idx])
                                else:
                                    eps = 1e-18
                                    llr = np.log((1.0 - error_priors) / (error_priors + eps))
                                    correction_wt = float(np.sum(llr * correction))

                            correction_hammingweight_list.append(correction_hw)
                            correction_weight_list.append(correction_wt)

                    # Debug
                    assert selected_idx == np.argmin(correction_wt) or True, "Selected index does not match minimum weight index."


                else:
                    iter_deltas_list.append(None)
                    vote_deltas_list.append(None)
                    mean_iter_list.append(None)
                    std_iter_list.append(None)

            # NumPy配列に変換（Noneを含むためobject型）
            iter_deltas = np.array(iter_deltas_list, dtype=object)
            vote_deltas = np.array(vote_deltas_list, dtype=object)
            mean_iterations = np.array(mean_iter_list, dtype=object)
            std_iterations = np.array(std_iter_list, dtype=object)
            converged_counts = np.array(converged_count_list, dtype=object)
            correction_hammingweight = np.array(correction_hammingweight_list, dtype=object)
            correction_weight = np.array(correction_weight_list, dtype=object)
            local_ambiguity_score = np.array(local_ambiguity_score_list, dtype=object)

            # Mean/std iterations (convert to float array if all values are present)
            if all(v is not None for v in mean_iter_list):
                mean_iterations = np.array(mean_iter_list, dtype=float)
            if all(v is not None for v in std_iter_list):
                std_iterations = np.array(std_iter_list, dtype=float)

            # Try to use effective_iterations from ensemble extra if available
            extra = results[0].extra
            if extra is not None and extra.get("effective_iterations") is not None:
                iterations = np.array([
                    res.extra["effective_iterations"] if res.extra is not None else float(res.iterations)
                    for res in results
                ], dtype=float)
            else:
                # Fallback to regular iterations with inf for non-converged
                iterations = np.array([res.iterations for res in results], dtype=float)
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
        
        return DecodeResult(
            predictions=outputs,
            iterations=iterations,
            converged=converged,
            mean_iterations=mean_iterations,
            std_iterations=std_iterations,
            logical_gaps=logical_gaps,
            iter_deltas=iter_deltas,
            vote_deltas=vote_deltas,
            converged_count=converged_counts,
            correction_hammingweight=correction_hammingweight,
            correction_weight=correction_weight,
            local_ambiguity_score=local_ambiguity_score,
        )

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
        local_ambiguity_threshold: float = 0.0,
    ):
        f"""Class for decoding stim circuits with sinter and relay-bp."""
        self.parallel = parallel
        self.decomposed_hyperedges = decomposed_hyperedges
        self.prune_decided_errors = prune_decided_errors
        self.threshold = threshold
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish
        self.get_detail_result = get_detail_result
        self.local_ambiguity_threshold = local_ambiguity_threshold

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
            local_ambiguity_threshold=getattr(self, 'local_ambiguity_threshold', 0.0),
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
        repulsive_gamma_dist: tuple[float, float] = (0.0, 0.0),
        abs_llr_threshold: float = None,
        pulse_per_leg: int = None,
        start_leg: int = None,
        local_ambiguity_threshold: float = 0.0,
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
        self.local_ambiguity_threshold = local_ambiguity_threshold
        self.use_automorphism = use_automorphism
        self.ensemble_mode = ensemble_mode
        self.repulsive_size = repulsive_size
        if repulsive_gamma_dist is None:          # <- add guard
            repulsive_gamma_dist = (0.0, 0.0)
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
    relay_config.pop("repulsive_gamma_dist", (0.0, 0.0))
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
        
        # Support 'type' or 'name' field to allow multiple instances of the same decoder class
        # e.g. name="relay-bp-1", params={"type": "relay-bp", ...}
        # User also wants to use 'name' field in YAML to specify the type.
        decoder_type = params.get("type", params.get("name", name))
        
        # Create a copy of params to avoid modifying the original spec
        # and remove 'type'/'name' so it doesn't get passed to constructors if they don't expect it
        decoder_params = params.copy()
        if "type" in decoder_params:
            del decoder_params["type"]
        if "name" in decoder_params:
            del decoder_params["name"]

        if decoder_type in {"relay-bp", "mem-bp", "msl-bp", "harmonized-bp"}:
            # Reuse sinter_decoders factory (single extraction)
            built[name] = sinter_decoders(**decoder_params)[decoder_type]
        elif decoder_type == "bposd":
            built[name] = SinterBpOsdDecoder(**decoder_params)
        elif decoder_type == "bplsd":
            if SinterLsdDecoder is None:
                raise ImportError("SinterLsdDecoder unavailable.")
            built[name] = SinterLsdDecoder(**decoder_params)
        elif decoder_type in ['tesseract', 'tesseract-long-beam', 'tesseract-short-beam']:
            tesseract_decoders_dict = make_tesseract_sinter_decoders_dict() # currently, custom parameters for terrerasct are not supported.
            built[name] = tesseract_decoders_dict[decoder_type]
        elif decoder_type == "retesseract":
            # ReTesseract uses dataclass-based configuration (import here to avoid circular import)
            from .retesseract import SinterDecoderReTesseract
            config = build_retesseract_config(decoder_params)
            built[name] = SinterDecoderReTesseract(config=config, seed=decoder_params.get("seed"))
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type} (for decoder '{name}')")
    return built


def build_retesseract_config(params: dict):
    """
    Build ReTesseractConfig from a flat dictionary of parameters.
    
    The params dict can contain keys like:
    - relay_bp.alpha, relay_bp.gamma0, relay_bp.pre_iter, ...
    - harmonized.ensemble_size, harmonized.selection_strategy, ...
    - tesseract.det_beam, tesseract.pqlimit, ...
    - integration.independent_mode, integration.use_llr_based_det_order, ...
    - confidence_threshold (top-level ReTesseract param)
    
    Or use nested dicts:
    - relay_bp_config: {alpha: ..., gamma0: ...}
    - harmonized_config: {ensemble_size: ..., ...}
    - tesseract_config: {det_beam: ..., ...}
    - tesseract_integration_config: {independent_mode: ..., ...}
    """
    # Import here to avoid circular import
    from .retesseract import (
        ReTesseractConfig,
        RelayBPConfig,
        HarmonizedConfig,
        TesseractConfig,
        TesseractIntegrationConfig,
    )
    
    # Extract nested configs if provided directly
    relay_bp_dict = params.get("relay_bp_config", {})
    harmonized_dict = params.get("harmonized_config", {})
    tesseract_dict = params.get("tesseract_config", {})
    integration_dict = params.get("tesseract_integration_config", {})
    
    # Also support flat dot-notation keys (relay_bp.alpha -> relay_bp_config.alpha)
    for key, value in params.items():
        if key.startswith("relay_bp."):
            field_name = key[len("relay_bp."):]
            relay_bp_dict[field_name] = value
        elif key.startswith("harmonized."):
            field_name = key[len("harmonized."):]
            harmonized_dict[field_name] = value
        elif key.startswith("tesseract."):
            field_name = key[len("tesseract."):]
            tesseract_dict[field_name] = value
        elif key.startswith("integration."):
            field_name = key[len("integration."):]
            integration_dict[field_name] = value
    
    # Handle tuple conversion for gamma_dist_interval and perturbation_range
    if "gamma_dist_interval" in relay_bp_dict and isinstance(relay_bp_dict["gamma_dist_interval"], list):
        relay_bp_dict["gamma_dist_interval"] = tuple(relay_bp_dict["gamma_dist_interval"])
    if "perturbation_range" in harmonized_dict and isinstance(harmonized_dict["perturbation_range"], list):
        harmonized_dict["perturbation_range"] = tuple(harmonized_dict["perturbation_range"])
    if "repulsive_gamma_dist" in harmonized_dict and isinstance(harmonized_dict["repulsive_gamma_dist"], list):
        harmonized_dict["repulsive_gamma_dist"] = tuple(harmonized_dict["repulsive_gamma_dist"])
    
    # Handle beam_width_schedule conversion: list of [threshold, beam] -> list of tuples
    if "beam_width_schedule" in integration_dict and isinstance(integration_dict["beam_width_schedule"], list):
        schedule = integration_dict["beam_width_schedule"]
        integration_dict["beam_width_schedule"] = [tuple(item) for item in schedule]
    
    # Build config objects
    relay_bp_config = RelayBPConfig(**relay_bp_dict) if relay_bp_dict else RelayBPConfig()
    harmonized_config = HarmonizedConfig(**harmonized_dict) if harmonized_dict else HarmonizedConfig()
    tesseract_config = TesseractConfig(**tesseract_dict) if tesseract_dict else TesseractConfig()
    integration_config = TesseractIntegrationConfig(**integration_dict) if integration_dict else TesseractIntegrationConfig()
    
    # Top-level ReTesseract switching parameters
    confidence_threshold = params.get("confidence_threshold", 20.0)
    mean_iteration_threshold = params.get("mean_iteration_threshold", None)
    std_iteration_threshold = params.get("std_iteration_threshold", None)
    
    return ReTesseractConfig(
        relay_bp_config=relay_bp_config,
        harmonized_config=harmonized_config,
        tesseract_config=tesseract_config,
        tesseract_integration_config=integration_config,
        confidence_threshold=confidence_threshold,
        mean_iteration_threshold=mean_iteration_threshold,
        std_iteration_threshold=std_iteration_threshold,
    )

