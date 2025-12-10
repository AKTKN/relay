"""
Tests for DEM modification based on BP posteriors.
"""
import numpy as np
import stim
import pytest

# Test the DEM modification functions directly
def test_llr_to_probability():
    """Test LLR to probability conversion."""
    from relay_bp.stim.sinter.retesseract import _llr_to_probability
    
    # LLR = 0 => p = 0.5
    assert abs(_llr_to_probability(0.0) - 0.5) < 1e-6
    
    # Large positive LLR => p close to 1
    assert _llr_to_probability(30.0) > 0.999
    
    # Large negative LLR => p close to 0
    assert _llr_to_probability(-30.0) < 0.001
    
    # LLR = log(0.1/0.9) => p = 0.1
    llr = np.log(0.1 / 0.9)
    assert abs(_llr_to_probability(llr) - 0.1) < 1e-6


def test_probability_to_llr():
    """Test probability to LLR conversion."""
    from relay_bp.stim.sinter.retesseract import _probability_to_llr
    
    # p = 0.5 => LLR = 0
    assert abs(_probability_to_llr(0.5)) < 1e-6
    
    # p = 0.1 => LLR = log(0.1/0.9)
    expected_llr = np.log(0.1 / 0.9)
    assert abs(_probability_to_llr(0.1) - expected_llr) < 1e-6


def test_build_hyperedge_to_instructions_map():
    """Test hyperedge to instruction mapping."""
    from relay_bp.stim.sinter.retesseract import _build_hyperedge_to_instructions_map
    
    dem = stim.DetectorErrorModel('''
        error(0.01) D0 D1 L0
        error(0.02) D1 D2
        error(0.03) D0
        error(0.04) D0 D1 L0
    ''')
    
    hyperedge_to_insts, flat_instructions = _build_hyperedge_to_instructions_map(dem)
    
    # Check that hyperedges are correctly identified
    assert frozenset([0, 1]) in hyperedge_to_insts
    assert frozenset([1, 2]) in hyperedge_to_insts
    assert frozenset([0]) in hyperedge_to_insts
    
    # D0 D1 appears twice (indices 0 and 3)
    assert len(hyperedge_to_insts[frozenset([0, 1])]) == 2
    assert 0 in hyperedge_to_insts[frozenset([0, 1])]
    assert 3 in hyperedge_to_insts[frozenset([0, 1])]
    
    # D1 D2 appears once (index 1)
    assert len(hyperedge_to_insts[frozenset([1, 2])]) == 1
    
    # D0 appears once (index 2)
    assert len(hyperedge_to_insts[frozenset([0])]) == 1


def test_modify_dem_priors_basic():
    """Test basic DEM modification."""
    from relay_bp.stim.sinter.retesseract import modify_dem_priors_from_posteriors
    import scipy.sparse as sparse
    
    # Create a simple DEM
    dem = stim.DetectorErrorModel('''
        error(0.01) D0 D1 L0
        error(0.02) D1 D2
        error(0.03) D0
    ''')
    
    # Create a check matrix matching the hyperedge structure
    # 3 hyperedges: {D0, D1}, {D1, D2}, {D0}
    # This should match the DEM structure
    check_matrix = sparse.csr_matrix([
        [1, 0, 1],  # D0 is in hyperedge 0 and 2
        [1, 1, 0],  # D1 is in hyperedge 0 and 1
        [0, 1, 0],  # D2 is in hyperedge 1
    ])
    
    # Posterior LLRs: positive = unlikely, negative = likely
    # Let's say we think the first error is very likely (negative LLR)
    posterior_llrs = np.array([-5.0, 0.0, 2.0])  # 3 hyperedges
    
    # Modify DEM with full strength
    modified_dem = modify_dem_priors_from_posteriors(
        dem=dem,
        posterior_llrs=posterior_llrs,
        check_matrix=check_matrix,
        modification_strength=1.0,
    )
    
    # Check that the DEM was modified
    assert modified_dem != dem
    
    # Parse modified DEM to check probabilities
    modified_probs = []
    for inst in modified_dem.flattened():
        if inst.type == "error":
            modified_probs.append(inst.args_copy()[0])
    
    # First error (D0 D1) should have higher probability (negative LLR = likely)
    # Original was 0.01, posterior suggests ~0.993 (LLR=-5)
    assert modified_probs[0] > 0.5  # Should be close to 0.993
    
    # Second error (D1 D2) should be around 0.5 (LLR=0)
    assert 0.4 < modified_probs[1] < 0.6
    
    # Third error (D0) should have lower probability (positive LLR = unlikely)
    # LLR=2 => p ~= 0.12
    assert modified_probs[2] < 0.2


def test_modify_dem_priors_partial_strength():
    """Test DEM modification with partial strength."""
    from relay_bp.stim.sinter.retesseract import modify_dem_priors_from_posteriors
    import scipy.sparse as sparse
    
    dem = stim.DetectorErrorModel('''
        error(0.1) D0 D1
    ''')
    
    check_matrix = sparse.csr_matrix([
        [1],  # D0
        [1],  # D1
    ])
    
    # Strong posterior suggesting error is likely
    posterior_llrs = np.array([-5.0])
    
    # 50% strength - should blend original (0.1) and posterior (~0.993)
    modified_dem = modify_dem_priors_from_posteriors(
        dem=dem,
        posterior_llrs=posterior_llrs,
        check_matrix=check_matrix,
        modification_strength=0.5,
    )
    
    for inst in modified_dem.flattened():
        if inst.type == "error":
            p = inst.args_copy()[0]
            # Should be roughly (0.1 + 0.993) / 2 ≈ 0.5
            assert 0.4 < p < 0.6
            break


def test_modify_dem_priors_zero_strength():
    """Test DEM modification with zero strength (should keep original)."""
    from relay_bp.stim.sinter.retesseract import modify_dem_priors_from_posteriors
    import scipy.sparse as sparse
    
    dem = stim.DetectorErrorModel('''
        error(0.123) D0 D1
    ''')
    
    check_matrix = sparse.csr_matrix([
        [1],
        [1],
    ])
    
    posterior_llrs = np.array([-5.0])
    
    # Zero strength - should keep original
    modified_dem = modify_dem_priors_from_posteriors(
        dem=dem,
        posterior_llrs=posterior_llrs,
        check_matrix=check_matrix,
        modification_strength=0.0,
    )
    
    for inst in modified_dem.flattened():
        if inst.type == "error":
            p = inst.args_copy()[0]
            assert abs(p - 0.123) < 1e-6
            break


def test_modify_dem_preserves_structure():
    """Test that DEM modification preserves detector and observable structure."""
    from relay_bp.stim.sinter.retesseract import modify_dem_priors_from_posteriors
    import scipy.sparse as sparse
    
    dem = stim.DetectorErrorModel('''
        error(0.01) D0 D1 L0
        error(0.02) D1 D2 L1
        detector(0, 0, 0) D0
        logical_observable L0
    ''')
    
    check_matrix = sparse.csr_matrix([
        [1, 0],
        [1, 1],
        [0, 1],
    ])
    
    posterior_llrs = np.array([0.0, 0.0])
    
    modified_dem = modify_dem_priors_from_posteriors(
        dem=dem,
        posterior_llrs=posterior_llrs,
        check_matrix=check_matrix,
        modification_strength=1.0,
    )
    
    # Count instruction types
    error_count = 0
    detector_count = 0
    logical_count = 0
    
    for inst in modified_dem.flattened():
        if inst.type == "error":
            error_count += 1
            # Check that observables are preserved
            targets = inst.targets_copy()
            target_str = str(targets)
            if "D0" in target_str and "D1" in target_str:
                assert "L0" in target_str
        elif inst.type == "detector":
            detector_count += 1
        elif inst.type == "logical_observable":
            logical_count += 1
    
    assert error_count == 2
    assert detector_count == 1
    assert logical_count == 1


if __name__ == "__main__":
    test_llr_to_probability()
    test_probability_to_llr()
    test_build_hyperedge_to_instructions_map()
    test_modify_dem_priors_basic()
    test_modify_dem_priors_partial_strength()
    test_modify_dem_priors_zero_strength()
    test_modify_dem_preserves_structure()
    print("All tests passed!")
