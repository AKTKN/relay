from __future__ import annotations
import pathlib

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict, FrozenSet
import scipy.sparse as sparse

import stim
from sinter import Decoder, CompiledDecoder
from autdec.igraph_auts import random_vertex_graph_auts_from_bliss
import relay_bp
from .check_matrices import CheckMatrices
from ldpc.sinter_decoders import SinterBpOsdDecoder
from ldpc.sinter_decoders.sinter_lsd_decoder import SinterLsdDecoder
from tesseract_decoder import make_tesseract_sinter_decoders_dict, TesseractSinterDecoder, tesseract
import tesseract_decoder
from .decoders import SinterDecoder_BaseBP
from .decode_result import DecodeResult


# ============================================================================
# DEM Modification Utilities
# ============================================================================

def _build_hyperedge_to_instructions_map(
    dem: stim.DetectorErrorModel,
) -> Tuple[Dict[FrozenSet[int], List[int]], List[stim.DemInstruction]]:
    """Build a mapping from hyperedge (detector set) to DEM instruction indices.
    
    This function maps detector patterns (hyperedges) to the indices of DEM
    instructions that produce those patterns. Multiple instructions can map
    to the same hyperedge.
    
    Args:
        dem: The detector error model
        
    Returns:
        Tuple of:
        - hyperedge_to_insts: Dict mapping FrozenSet[detector_ids] to list of instruction indices
        - flat_instructions: List of all flattened DEM instructions
    """
    hyperedge_to_insts: Dict[FrozenSet[int], List[int]] = {}
    flat_instructions = list(dem.flattened())
    
    for inst_idx, inst in enumerate(flat_instructions):
        if inst.type != "error":
            continue
            
        # Extract detector IDs from this instruction
        detectors: List[int] = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                detectors.append(t.val)
        
        hyperedge = frozenset(detectors)
        
        if hyperedge not in hyperedge_to_insts:
            hyperedge_to_insts[hyperedge] = []
        hyperedge_to_insts[hyperedge].append(inst_idx)
    
    return hyperedge_to_insts, flat_instructions


def _llr_to_probability(llr: float) -> float:
    """Convert log-likelihood ratio to probability.
    
    LLR = log(p / (1-p))
    => p = 1 / (1 + exp(-LLR))
    
    Clamp output to [1e-10, 1-1e-10] to avoid numerical issues.
    """
    # Avoid overflow in exp
    if llr > 30:
        return 1.0 - 1e-10
    elif llr < -30:
        return 1e-10
    
    p = 1.0 / (1.0 + np.exp(-llr))
    return np.clip(p, 1e-10, 1.0 - 1e-10)


def _probability_to_llr(p: float) -> float:
    """Convert probability to log-likelihood ratio.
    
    LLR = log(p / (1-p))
    
    Clamp input to [1e-10, 1-1e-10] to avoid numerical issues.
    """
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    return np.log(p / (1.0 - p))


def modify_dem_priors_from_posteriors(
    dem: stim.DetectorErrorModel,
    posterior_llrs: np.ndarray,
    check_matrix: sparse.spmatrix,
    modification_strength: float = 1.0,
) -> stim.DetectorErrorModel:
    """Modify DEM error probabilities based on BP posterior LLRs.
    
    This function takes the posterior LLRs from BP decoding and uses them to
    modify the error probabilities in the DEM. The idea is that BP gives us
    updated beliefs about which errors are likely, and we can use this to
    inform other decoders like Tesseract.
    
    The mapping from variable nodes (columns in check_matrix) to DEM hyperedges
    follows the same convention as beliefmatching: each hyperedge corresponds
    to a unique set of detectors that it triggers.
    
    Args:
        dem: Original detector error model
        posterior_llrs: Posterior LLR values for each variable node from BP
            Shape: (num_variables,) where num_variables = check_matrix.shape[1]
            Positive LLR means error is unlikely, negative means likely
        check_matrix: The check matrix (detectors x variables) used for decoding
        modification_strength: How much to trust the posterior (0-1).
            0 = use original priors, 1 = fully replace with posteriors
            
    Returns:
        Modified DetectorErrorModel with updated error probabilities
        
    Notes:
        - The mapping assumes variables are ordered the same as hyperedges in
          the order they appear in the DEM (after deduplication)
        - For hyperedges with multiple contributing DEM instructions, we
          distribute the posterior probability proportionally to original priors
    """
    if len(posterior_llrs) == 0:
        return dem
    
    # Build hyperedge to instruction mapping
    hyperedge_to_insts, flat_instructions = _build_hyperedge_to_instructions_map(dem)
    
    # Build mapping from variable index to hyperedge
    # This follows the beliefmatching convention: hyperedges are ordered by first appearance
    hyperedge_order: List[FrozenSet[int]] = []
    seen_hyperedges: set = set()
    
    for inst in flat_instructions:
        if inst.type != "error":
            continue
        detectors = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                detectors.append(t.val)
        hyperedge = frozenset(detectors)
        if hyperedge not in seen_hyperedges:
            hyperedge_order.append(hyperedge)
            seen_hyperedges.add(hyperedge)
    
    num_hyperedges = len(hyperedge_order)
    num_posteriors = len(posterior_llrs)
    
    if num_posteriors != num_hyperedges:
        # Shape mismatch - try to handle gracefully
        # Use min to avoid index errors
        num_to_use = min(num_posteriors, num_hyperedges)
    else:
        num_to_use = num_hyperedges
    
    # Calculate new probabilities for each instruction
    new_probs: Dict[int, float] = {}  # inst_idx -> new_probability
    
    for var_idx in range(num_to_use):
        hyperedge = hyperedge_order[var_idx]
        inst_indices = hyperedge_to_insts.get(hyperedge, [])
        
        if not inst_indices:
            continue
        
        # Convert posterior LLR to probability
        posterior_prob = _llr_to_probability(posterior_llrs[var_idx])
        
        # Get original probabilities for all instructions contributing to this hyperedge
        original_probs = []
        for inst_idx in inst_indices:
            inst = flat_instructions[inst_idx]
            original_probs.append(inst.args_copy()[0])
        
        # Calculate combined original probability (same formula as beliefmatching)
        # p_combined = sum over instructions of: p_i * product_{j!=i}(1-p_j)
        # For simplicity, if single instruction: just use it directly
        # For multiple: distribute proportionally
        
        if len(inst_indices) == 1:
            # Single instruction - directly assign posterior
            inst_idx = inst_indices[0]
            original_p = original_probs[0]
            # Blend original and posterior based on strength
            new_p = (1 - modification_strength) * original_p + modification_strength * posterior_prob
            new_probs[inst_idx] = np.clip(new_p, 1e-10, 1.0 - 1e-10)
        else:
            # Multiple instructions contribute to same hyperedge
            # Distribute the posterior probability proportionally to original priors
            total_original = sum(original_probs)
            if total_original > 0:
                for inst_idx, orig_p in zip(inst_indices, original_probs):
                    weight = orig_p / total_original
                    target_p = posterior_prob * weight
                    new_p = (1 - modification_strength) * orig_p + modification_strength * target_p
                    new_probs[inst_idx] = np.clip(new_p, 1e-10, 1.0 - 1e-10)
    
    # Build new DEM with modified probabilities
    new_dem = stim.DetectorErrorModel()
    
    for inst_idx, inst in enumerate(flat_instructions):
        if inst.type == "error":
            targets = inst.targets_copy()
            if inst_idx in new_probs:
                new_prob = new_probs[inst_idx]
            else:
                new_prob = inst.args_copy()[0]  # Keep original
            new_inst = stim.DemInstruction("error", [new_prob], targets)
            new_dem.append(new_inst)
        else:
            # Keep non-error instructions as-is
            new_dem.append(inst)
    
    return new_dem

@dataclass
class RelayBPConfig:
    alpha: Optional[float] = None
    gamma0: float = 0.1
    pre_iter: int = 60
    num_sets: int = 60
    set_max_iter: int = 60
    gamma_dist_interval: Tuple[float, float] = (-0.24, 0.66)
    explicit_gammas: Optional[np.ndarray] = None
    stop_nconv: int = 5
    stopping_criterion: str = "nconv"

@dataclass
class HarmonizedConfig:
    ensemble_size: int = 1
    selection_strategy: str = "MostLikely"
    perturbation_range: Tuple[float, float] = (0.0, 0.0)
    use_automorphism: bool = False

    # Repulsive Mode
    ensemble_mode: str = "normal"
    repulsive_size: int = 0
    repulsive_gamma_dist: Tuple[float, float] = (0.0, 0.0)
    abs_llr_threshold: Optional[float] = None
    pulse_per_leg: Optional[int] = None
    start_leg: Optional[int] = None

@dataclass
class TesseractConfig:
    # --- Search parameters ---
    det_beam: int = 20
    pqlimit: int = 1000000
    beam_climbing: bool = True
    no_revisit_dets: bool = True

    # --- Cost / Penalty Parameter ---
    det_penalty: float = 0.0

    # --- Processing Options ---
    merge_errors: bool = True
    det_orders: Optional[List[List[int]]] = None
    
    # --- Detector Ordering Generation ---
    # Number of detector orderings to generate (used with build_det_orders)
    num_det_orders: Optional[int] = None
    # Method for generating detector orderings (e.g., 'DetIndex', 'Random', 'Greedy')
    # This will be passed to tesseract_decoder.utils.DetOrder enum
    det_order_method: str = "DetIndex"

    # --- Debug / Visualization ---
    verbose: bool = False
    create_visualization: bool = False


@dataclass
class TesseractIntegrationConfig:
    """Configuration for how Tesseract integrates with ensemble BP results."""
    
    # If True, run Tesseract independently (ignore BP results)
    # If False, use BP results to guide Tesseract
    independent_mode: bool = True
    
    # --- Detector Ordering Options ---
    # If True, generate detector ordering based on LLR values from BP
    # Detectors associated with high |LLR| error channels are processed first
    use_llr_based_det_order: bool = False
    
    # --- Dynamic Beam Width Options ---
    # If True, dynamically adjust det_beam based on logical_gap
    use_dynamic_beam: bool = False
    
    # Beam width mapping: list of (gap_threshold, beam_width) tuples
    # When logical_gap < threshold, use corresponding beam_width
    # Should be sorted by threshold in ascending order
    # Example: [(5.0, 10), (10.0, 7), (20.0, 5), (inf, 3)]
    # means: gap < 5 -> beam=10, 5 <= gap < 10 -> beam=7, etc.
    beam_width_schedule: Optional[List[Tuple[float, int]]] = None
    
    # --- Prior Modification Options ---
    # If True, modify DEM error priors based on BP posteriors before running Tesseract
    # This allows Tesseract to use updated error probabilities informed by BP
    use_llr_modified_priors: bool = False
    
    # How much to trust the BP posterior when modifying priors (0.0 - 1.0)
    # 0.0 = use original DEM priors, 1.0 = fully replace with BP posteriors
    # Intermediate values blend original and posterior probabilities
    prior_modification_strength: float = 0.0


@dataclass
class ReTesseractConfig:
    relay_bp_config: RelayBPConfig = field(default_factory=RelayBPConfig)
    harmonized_config: HarmonizedConfig = field(default_factory=HarmonizedConfig)
    tesseract_config: TesseractConfig = field(default_factory=TesseractConfig)
    tesseract_integration_config: TesseractIntegrationConfig = field(default_factory=TesseractIntegrationConfig)

    # ReTesseract-specific switching parameters
    # Logical gap threshold: switch if logical_gap <= threshold
    # If set to float('inf'), switch when logical_gap is finite (opinions differ) or 'nan' (all failed)
    confidence_threshold: float = 20.0
    
    # Mean iteration threshold: switch if ensemble_mean_iteration > threshold
    # If None, this condition is disabled
    mean_iteration_threshold: Optional[float] = None
    
    # Iteration std threshold: switch if ensemble_std_iteration > threshold  
    # If None, this condition is disabled
    std_iteration_threshold: Optional[float] = None



class SinterReTesseractCompiledDecoder(CompiledDecoder):
    def __init__(
        self,
        config: ReTesseractConfig,
        observable_decoder: relay_bp.ObservableDecoderRunner,
        check_matrices: CheckMatrices,
        dem: stim.DetectorErrorModel,
        parallel: bool = False,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
    ):
        self.config = config
        self.observable_decoder = observable_decoder
        self.parallel = parallel
        self.check_matrices = check_matrices
        self.dem = dem
        self.show_progress = show_progress
        self.leave_progress_bar_on_finish = leave_progress_bar_on_finish

    def _do_switch(
        self, 
        logical_gap, 
        mean_iteration: Optional[float] = None,
        std_iteration: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Determine whether to switch to Tesseract based on multiple conditions.
        
        Switching conditions (OR logic - any condition triggers switch):
        1. Logical gap condition:
           - If confidence_threshold is inf: switch when logical_gap is finite (opinions differ) or NaN (all failed)
           - Otherwise: switch when logical_gap <= threshold or logical_gap is NaN
        2. Mean iteration condition (if mean_iteration_threshold is set):
           - Switch when ensemble_mean_iteration > threshold
        3. Std iteration condition (if std_iteration_threshold is set):
           - Switch when ensemble_std_iteration > threshold
           
        Args:
            logical_gap: The logical gap value (float, NaN, or None)
            mean_iteration: Mean iteration count from ensemble (optional)
            std_iteration: Std deviation of iteration counts from ensemble (optional)
            
        Returns:
            Tuple of (bool, str): (should_switch, reason_string)
        """

        # switch reason tracking can be added here if needed
        reason = ''
        flag = False

        # --- Condition 1: Logical gap ---
        confidence_threshold = self.config.confidence_threshold
        
        if logical_gap is None or (isinstance(logical_gap, float) and np.isnan(logical_gap)):
            # All decoders failed - always switch
            reason += '1'
            flag = True
        
        # Handle inf threshold: switch when opinions differ (finite logical_gap)
        if np.isinf(confidence_threshold):
            # inf threshold means: switch only if logical_gap is finite (not inf)
            # i.e., opinions differ among ensemble members
            if isinstance(logical_gap, (int, float)) and np.isfinite(logical_gap):
                reason += '2'
                flag = True
        else:
            # Normal threshold: switch if logical_gap <= threshold
            if isinstance(logical_gap, (int, float)) and logical_gap <= confidence_threshold:
                reason += '2'
                flag = True
        
        # --- Condition 2: Mean iteration threshold ---
        if (self.config.mean_iteration_threshold is not None 
            and mean_iteration is not None
            and mean_iteration > self.config.mean_iteration_threshold):
            reason += '3'
            flag = True
        
        # --- Condition 3: Std iteration threshold ---
        if (self.config.std_iteration_threshold is not None
            and std_iteration is not None  
            and std_iteration > self.config.std_iteration_threshold):
            reason += '4'
            flag = True
        
        return flag, reason
    
    def _compute_dynamic_beam_width(self, logical_gap: float) -> int:
        """Compute dynamic beam width based on logical gap using the schedule.
        
        Lower logical gap (less confidence) -> larger beam width for more thorough search.
        """
        integration_cfg = self.config.tesseract_integration_config
        schedule = integration_cfg.beam_width_schedule
        
        if schedule is None or not integration_cfg.use_dynamic_beam:
            # Use default beam width from tesseract config
            return self.config.tesseract_config.det_beam
        
        # Handle nan/inf logical gap
        is_nan = isinstance(logical_gap, float) and np.isnan(logical_gap)
        if is_nan:
            # All decoders failed - use maximum beam width (first entry typically)
            return schedule[0][1] if schedule else self.config.tesseract_config.det_beam
        
        # Find appropriate beam width from schedule
        for threshold, beam_width in schedule:
            if logical_gap < threshold:
                return beam_width
        
        # If no threshold matched, use the last one (should be inf threshold)
        return schedule[-1][1] if schedule else self.config.tesseract_config.det_beam
    
    def _compute_llr_based_det_order(
        self, 
        mean_posterior_ratios: Optional[np.ndarray]
    ) -> Optional[List[List[int]]]:
        """Generate detector ordering based on LLR values from BP.
        
        Detectors associated with high |LLR| (confident) error channels should be
        processed first, as they are more likely to have clear error patterns.
        
        Args:
            mean_posterior_ratios: Mean LLR values for each variable node from ensemble BP
            
        Returns:
            A list containing one detector ordering based on LLR values,
            or None if ordering cannot be computed.
        """
        integration_cfg = self.config.tesseract_integration_config
        
        if not integration_cfg.use_llr_based_det_order or mean_posterior_ratios is None:
            return None
        
        # Get the check matrix to map variable nodes to detectors
        check_matrix = self.check_matrices.check_matrix
        num_detectors = check_matrix.shape[0]
        num_variables = check_matrix.shape[1]
        
        if len(mean_posterior_ratios) != num_variables:
            # Shape mismatch, cannot compute ordering
            return None
        
        # For each detector, compute a score based on the LLRs of connected variable nodes
        # Higher |LLR| means more confident -> process these detectors first
        detector_scores = np.zeros(num_detectors)
        
        # Convert to dense for easier computation (may be slow for very large matrices)
        if sparse.issparse(check_matrix):
            check_dense = check_matrix.toarray()
        else:
            check_dense = check_matrix
            
        for det_idx in range(num_detectors):
            # Get variable nodes connected to this detector
            connected_vars = np.where(check_dense[det_idx, :] != 0)[0]
            if len(connected_vars) > 0:
                # Use maximum |LLR| among connected variables as detector score
                # Higher score = more confident = process first
                detector_scores[det_idx] = np.max(np.abs(mean_posterior_ratios[connected_vars]))
        
        # Sort detectors by score in descending order (highest confidence first)
        det_order = np.argsort(-detector_scores).tolist()
        
        return [det_order]

    def _run_tesseract(
        self, 
        syndrome: np.ndarray,
        logical_gap: Optional[float] = None,
        mean_posterior_ratios: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run Tesseract decoder with optional BP-guided configuration.
        
        Args:
            syndrome: The syndrome to decode
            logical_gap: Logical gap from BP (used for dynamic beam width)
            mean_posterior_ratios: Mean LLR values from ensemble BP (used for det ordering and prior modification)
            
        Returns:
            Predicted observables
        """
        integration_cfg = self.config.tesseract_integration_config
        tess_cfg = self.config.tesseract_config
        
        # --- LLR-based prior modification ---
        # Modify DEM error probabilities based on BP posteriors
        if (not integration_cfg.independent_mode 
            and integration_cfg.use_llr_modified_priors 
            and mean_posterior_ratios is not None):
            modified_dem = modify_dem_priors_from_posteriors(
                dem=self.dem,
                posterior_llrs=mean_posterior_ratios,
                check_matrix=self.check_matrices.check_matrix,
                modification_strength=integration_cfg.prior_modification_strength,
            )
        else:
            modified_dem = self.dem
        
        # Prepare tesseract config dict
        config_dict = {
            'dem': modified_dem,
            'pqlimit': tess_cfg.pqlimit,
            'beam_climbing': tess_cfg.beam_climbing,
            'no_revisit_dets': tess_cfg.no_revisit_dets,
            'det_penalty': tess_cfg.det_penalty,
            'merge_errors': tess_cfg.merge_errors,
            'verbose': tess_cfg.verbose,
            'create_visualization': tess_cfg.create_visualization,
        }
        
        # --- Dynamic beam width ---
        if integration_cfg.use_dynamic_beam and logical_gap is not None:
            config_dict['det_beam'] = self._compute_dynamic_beam_width(logical_gap)
        else:
            config_dict['det_beam'] = tess_cfg.det_beam
        
        # --- Detector ordering ---
        # Priority: 1) LLR-based (non-independent mode), 2) build_det_orders (independent mode with num_det_orders), 3) manual det_orders
        if not integration_cfg.independent_mode and integration_cfg.use_llr_based_det_order:
            # LLR-based detector ordering for non-independent mode
            llr_det_order = self._compute_llr_based_det_order(mean_posterior_ratios)
            if llr_det_order is not None:
                config_dict['det_orders'] = llr_det_order
            elif tess_cfg.det_orders is not None:
                config_dict['det_orders'] = tess_cfg.det_orders
            elif tess_cfg.num_det_orders is not None:
                # Fallback to build_det_orders if LLR fails
                config_dict['det_orders'] = self._build_detector_orderings(modified_dem, tess_cfg)
        elif tess_cfg.num_det_orders is not None:
            # Independent mode: use build_det_orders if num_det_orders is specified
            config_dict['det_orders'] = self._build_detector_orderings(modified_dem, tess_cfg)
        elif tess_cfg.det_orders is not None:
            # Manual det_orders provided
            config_dict['det_orders'] = tess_cfg.det_orders
        
        # Create tesseract config and decoder
        tesseract_config = tesseract.TesseractConfig(**config_dict)
        tesseract_decoder = tesseract.TesseractDecoder(tesseract_config)
        
        # Run decode
        tess_prediction = tesseract_decoder.decode(syndrome.astype(bool))
        return np.array(tess_prediction, dtype=np.uint8)
    
    def _build_detector_orderings(
        self,
        dem: stim.DetectorErrorModel,
        tess_cfg: TesseractConfig,
    ) -> List[List[int]]:
        """Build detector orderings using tesseract_decoder.utils.build_det_orders.
        
        Args:
            dem: The detector error model
            tess_cfg: Tesseract configuration containing num_det_orders and det_order_method
            
        Returns:
            List of detector orderings
        """
        # Get the DetOrder enum value from the method string
        det_order_enum = getattr(tesseract_decoder.utils.DetOrder, tess_cfg.det_order_method)
        
        det_orders = tesseract_decoder.utils.build_det_orders(
            dem=dem,
            num_det_orders=tess_cfg.num_det_orders,
            method=det_order_enum,
        )
        
        return det_orders

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: "np.ndarray",
    ) -> "np.ndarray":
        syndromes_raw = np.unpackbits(
            bit_packed_detection_event_data, bitorder="little", axis=1
        ).astype(np.uint8)

        # Tesseract expects the original DEM-sized syndrome; keep a raw copy for it.
        num_detectors = self.dem.num_detectors
        if syndromes_raw.shape[1] > num_detectors:
            syndromes_raw = syndromes_raw[:, :num_detectors]

        # Relay-BP operates on a pruned check matrix, so apply biasing/slicing only
        # to the BP view of the syndrome.
        syndromes_bp = syndromes_raw
        if self.check_matrices.syndrome_bias is not None:
            syndromes_bp = (syndromes_bp + self.check_matrices.syndrome_bias) % 2

        # In harmonized decoder, it calculates permutation of syndromes. 
        # At that time, the dimension of checks and syndromes must be the same. 
        # In decode_shots_bit_packed, the syndrome data is bit-packed, so its shape is a multiple of 8 (byte).
        # Therefore, when the dimension of check matrix (row) is not a multiple of 8, 
        # we need to slice the syndrome data to match the dimension.
        num_checks = self.check_matrices.check_matrix.shape[0]
        if syndromes_bp.shape[1] > num_checks:
            syndromes_bp = syndromes_bp[:, :num_checks]

        # === DEBUG: DEM and Syndrome Consistency Check ===
        import sys
        print(f"[DEBUG] DEM/Syndrome Check:", file=sys.stderr, flush=True)
        print(f"  Relay-BP DEM: num_detectors={self.check_matrices.check_matrix.shape[0]}, num_observables={self.check_matrices.observables_matrix.shape[0]}", file=sys.stderr, flush=True)
        print(f"  Tesseract DEM: num_detectors={self.dem.num_detectors}, num_observables={self.dem.num_observables}", file=sys.stderr, flush=True)
        
        print(f"[DEBUG] Syndrome dimensions:", file=sys.stderr, flush=True)
        print(f"  syndromes_raw shape: {syndromes_raw.shape} (num_detectors={num_detectors})", file=sys.stderr, flush=True)
        print(f"  syndromes_bp shape: {syndromes_bp.shape} (num_checks={num_checks})", file=sys.stderr, flush=True)
        
        # === DEBUG: Bias Information ===
        print(f"[DEBUG] Bias Information:", file=sys.stderr, flush=True)
        if self.check_matrices.syndrome_bias is not None:
            syndrome_bias_nonzero = np.count_nonzero(self.check_matrices.syndrome_bias)
            print(f"  syndrome_bias: shape={self.check_matrices.syndrome_bias.shape}, nonzero={syndrome_bias_nonzero}, "
                  f"values={self.check_matrices.syndrome_bias}", file=sys.stderr, flush=True)
        else:
            print(f"  syndrome_bias: None", file=sys.stderr, flush=True)
        
        if self.check_matrices.observables_bias is not None:
            observables_bias_nonzero = np.count_nonzero(self.check_matrices.observables_bias)
            print(f"  observables_bias: shape={self.check_matrices.observables_bias.shape}, nonzero={observables_bias_nonzero}, "
                  f"values={self.check_matrices.observables_bias}", file=sys.stderr, flush=True)
        else:
            print(f"  observables_bias: None", file=sys.stderr, flush=True)
        
        # === DEBUG: First shot syndrome comparison (only for first few shots) ===
        if syndromes_raw.shape[0] > 0:
            print(f"[DEBUG] First shot syndrome comparison (shot 0):", file=sys.stderr, flush=True)
            print(f"  syndromes_raw[0, :20]: {syndromes_raw[0, :20]}", file=sys.stderr, flush=True)
            print(f"  syndromes_bp[0, :20]: {syndromes_bp[0, :20]}", file=sys.stderr, flush=True)
            if self.check_matrices.syndrome_bias is not None:
                bias_applied = (syndromes_raw[0, :num_checks] + self.check_matrices.syndrome_bias) % 2
                print(f"  Expected bias-applied: {bias_applied[:20]}", file=sys.stderr, flush=True)

        results = self.observable_decoder.decode_observables_detailed_batch(
            syndromes_bp,
            parallel=self.parallel,
            progress_bar=self.show_progress,
            leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
        )
        predictions = np.array([res.observables for res in results])
        converged = np.array([res.converged for res in results])
        logical_gaps = np.array([res.logical_gap for res in results])

        # --- Iteration count handling ---
        # Try to use effective_iterations from ensemble extra if available
        extra = results[0].extra
        if extra is not None and extra.get("effective_iterations") is not None:
            # Use effective_iterations from all results
            iterations = np.array([
                res.extra["effective_iterations"] if res.extra is not None else float(res.iterations)
                for res in results
            ], dtype=float)
        else:
            # Fallback to regular iterations with inf for non-converged
            iterations = np.array([res.iterations for res in results], dtype=float) # return type from rust is int, so we need to convert to float.
            iterations[~converged] = np.inf


        # --- Detail handling ---
        mean_iterations = np.zeros(len(results), dtype=float)
        std_iterations = np.zeros(len(results), dtype=float)
        switch_count = 0
        switch_reason = ''
        correction_change = 0

        # Track rows where Tesseract overrides BP so we can handle biases correctly later.
        used_tesseract = np.zeros(len(results), dtype=bool)

        # === DEBUG: Prediction before and after observable bias ===
        predictions_before_bias = predictions.copy()

        # --- Switching logic ---
        for i, res in enumerate(results):
            # Extract mean_iteration and std_iteration from extra if available
            mean_iter = None
            std_iter = None
            mean_posterior_ratios = None
            if res.extra is not None:
                mean_iter = res.extra.get("ensemble_mean_iteration")
                std_iter = res.extra.get("ensemble_std_iteration")
                mean_posterior_ratios = res.extra.get("ensemble_mean_posterior_ratios")
            
            # Track mean and std iterations for output
            if mean_iter is not None:
                mean_iterations[i] = mean_iter
            if std_iter is not None:
                std_iterations[i] = std_iter
            
            do_switch, reason = self._do_switch(logical_gaps[i], mean_iteration=mean_iter, std_iteration=std_iter)
            if do_switch:
                switch_count += 1
                switch_reason += reason
                used_tesseract[i] = True
                
                # Run Tesseract with BP-guided configuration if available
                pred_tesseract = self._run_tesseract(
                    syndromes_raw[i, :],
                    logical_gap=logical_gaps[i],
                    mean_posterior_ratios=mean_posterior_ratios,
                )
                print(f"Debug: relay pred: {predictions[i, :]}, tesseract pred: {pred_tesseract}, logical_gap: {logical_gaps[i]}", file=sys.stderr, flush=True)
                if not np.array_equal(predictions[i,:], pred_tesseract):
                    print(f"Debug: Correction changed by Tesseract for shot {i}.")
                    correction_change += 1
                    predictions[i,:] = pred_tesseract
                # To indicate that Tesseract was used, we can set converged to True
                converged[i] = True

        # Apply observable bias only to the Relay-BP predictions. Tesseract already
        # works with the original DEM, so adding the bias again would be incorrect.
        if self.check_matrices.observables_bias is not None:
            bias = self.check_matrices.observables_bias
            predictions[~used_tesseract] = (predictions[~used_tesseract] + bias) % 2

        # === DEBUG: Observable Bias Application ===
        print(f"[DEBUG] Observable Bias Application:", file=sys.stderr, flush=True)
        print(f"  used_tesseract count: {np.count_nonzero(used_tesseract)}", file=sys.stderr, flush=True)
        if self.check_matrices.observables_bias is not None:
            bias_applied_count = np.count_nonzero(~used_tesseract)
            print(f"  Bias applied to {bias_applied_count} shots (Relay-BP only)", file=sys.stderr, flush=True)
            if bias_applied_count > 0:
                # Show example of bias application for first Relay-BP shot
                bp_indices = np.where(~used_tesseract)[0]
                if len(bp_indices) > 0:
                    example_idx = bp_indices[0]
                    print(f"  Example shot {example_idx}:", file=sys.stderr, flush=True)
                    print(f"    Before bias: {predictions_before_bias[example_idx, :]}", file=sys.stderr, flush=True)
                    print(f"    After bias:  {predictions[example_idx, :]}", file=sys.stderr, flush=True)
                    print(f"    Bias values: {self.check_matrices.observables_bias}", file=sys.stderr, flush=True)
        else:
            print(f"  No observables_bias applied", file=sys.stderr, flush=True)

        outputs = np.packbits(predictions, axis=1, bitorder="little")

        return DecodeResult(
            predictions=outputs,
            iterations=iterations,
            mean_iterations=mean_iterations,
            std_iterations=std_iterations,
            converged=converged,
            logical_gaps=logical_gaps,
            switch_count=switch_count,
            switch_resason=switch_reason,
            correction_change_count=correction_change,
        )
 



class SinterDecoderReTesseract(SinterDecoder_BaseBP):
    def __init__(
        self,
        config: ReTesseractConfig | None = None,
        *,
        seed: Optional[int] = None,
        parallel: bool = False,
        decomposed_hyperedges: bool | None = None,
        prune_decided_errors: bool = True,
        threshold: float = 0.0,
        show_progress: bool = False,
        leave_progress_bar_on_finish: bool = False,
        get_detail: bool = True, 
    ):
  
        self.config = config if config is not None else ReTesseractConfig()
        self.seed = np.random.randint(0, 2**32 - 1) if seed is None else seed

        super().__init__(
            parallel=parallel,
            decomposed_hyperedges=decomposed_hyperedges,
            prune_decided_errors=prune_decided_errors,
            threshold=threshold,
            show_progress=show_progress,
            leave_progress_bar_on_finish=leave_progress_bar_on_finish,
            get_detail_result=get_detail,
        )

    def build_observable_decoder(
        self, check_matrices: CheckMatrices
    ) -> relay_bp.ObservableDecoderRunner:

        # shorthand
        rcfg = self.config.relay_bp_config
        hcfg = self.config.harmonized_config

        # automorphism による列・行 permutation
        col_perms = None
        row_perms = None
        current_ensemble_size = hcfg.ensemble_size

        if hcfg.use_automorphism:
            assert hcfg.ensemble_size >= 1, (
                "ensemble_size should be at least 1 when using automorphism."
            )
            col_perms = []
            row_perms = []
            if hcfg.ensemble_size >= 2:
                bliss_cols, bliss_rows = random_vertex_graph_auts_from_bliss(
                    check_matrices.check_matrix, k=hcfg.ensemble_size - 1  # except identity
                )
                col_perms.extend(bliss_cols)
                row_perms.extend(bliss_rows)

            identity_col = sparse.identity(
                check_matrices.check_matrix.shape[1], dtype=int, format="csr"
            )
            identity_row = sparse.identity(
                check_matrices.check_matrix.shape[0], dtype=int, format="csr"
            )
            col_perms.insert(0, identity_col)
            row_perms.insert(0, identity_row)

            print(f"Debug: Found {len(col_perms)} automorphisms using bliss.")
            current_ensemble_size = len(col_perms)
            if current_ensemble_size < hcfg.ensemble_size:
                print(
                    f"Warning: Only found {current_ensemble_size} automorphisms, "
                    f"which is less than requested {hcfg.ensemble_size}. Using {current_ensemble_size}."
                )
                hcfg.ensemble_size = current_ensemble_size

        observable_decoder = relay_bp.ObservableDecoderRunner.with_ensemble_decoder(
            ensemble_size=hcfg.ensemble_size,
            check_matrix=check_matrices.check_matrix,
            observable_matrix=check_matrices.observables_matrix,
            error_priors=check_matrices.error_priors,
            alpha=rcfg.alpha,
            gamma0=rcfg.gamma0,
            pre_iter=rcfg.pre_iter,
            num_sets=rcfg.num_sets,
            set_max_iter=rcfg.set_max_iter,
            gamma_dist_interval=rcfg.gamma_dist_interval,
            explicit_gammas=rcfg.explicit_gammas,
            stop_nconv=rcfg.stop_nconv,
            stopping_criterion=rcfg.stopping_criterion,
            logging=False,
            selection_strategy=hcfg.selection_strategy,
            perturbation_min=hcfg.perturbation_range[0],
            perturbation_max=hcfg.perturbation_range[1],
            col_permutations=col_perms,
            row_permutations=row_perms,
            seed=self.seed,
            ensemble_mode=hcfg.ensemble_mode,
            repulsive_size=hcfg.repulsive_size,
            repulsive_gamma_dist=hcfg.repulsive_gamma_dist,
            abs_llr_threshold=hcfg.abs_llr_threshold,
            pulse_per_leg=hcfg.pulse_per_leg,
            start_leg=hcfg.start_leg,
        )
        return observable_decoder

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
        return SinterReTesseractCompiledDecoder(
            config=self.config,
            observable_decoder=observable_decoder_runner,
            check_matrices=check_matrices,
            dem=dem,
            parallel=self.parallel,
            show_progress=self.show_progress,
            leave_progress_bar_on_finish=self.leave_progress_bar_on_finish,
        )