# Ensemble Decoder - Decode Result Documentation

## Overview

This document describes the information available after decoding with the Ensemble relay-bp decoder.

## Basic DecodeResult (Common to All Decoders)

### Standard fields:

1.  **`decoding`** - Array1\<u8\>

      * Estimated error correction result (bit array).

2.  **`decoded_detectors`** - Array1\<u8\>

      * Detector syndrome calculated from the decoding result ($H \times decoding$).

3.  **`posterior_ratios`** - Array1\<f64\>

      * **Final marginals (LLR values) for each variable node.**
      * Log-Likelihood Ratio of all variable nodes upon decoding completion.
      * This is the most critical statistical information.

4.  **`success`** - bool

      * Whether decoding successfully converged (`decoded_detectors == actual_detectors`).

5.  **`decoding_quality`** - f64

      * Sum of log-likelihoods for the actual error pattern.
      * Higher values indicate better decoding quality.

6.  **`iterations`** - usize

      * Number of BP algorithm iterations used.

7.  **`max_iter`** - usize

      * Configured maximum number of iterations.

8.  **`logical_gap`** - Option\<f64\>

      * Logical error gap calculated by the Observable decoder (may be `None`).

9.  **`reliability`** - Option\<f64\>

      * Ensemble reliability score computed from per-decoder iterations and coset vote gap.
      * Returned by the Harmonized-BP wrapper when detailed stats are enabled.

-----

## Ensemble Decoder Specific Information (`extra` field)

In Ensemble mode, the `extra` field contains `EnsembleExtraResult`, providing the following information:

### Basic Ensemble Information:

1.  **`all_corrections`** - Vec\<Array1\<u8\>\>

      * **Correction results from all child decoders** in the ensemble.
      * Each element represents the output of one child decoder.

2.  **`llr_sums`** - Vec\<f64\>

      * **Sum of LLRs corresponding to each correction result.**
      * Sum of LLRs for all variable nodes associated with each correction.
      * Used in the selection strategy (MostLikely).

3.  **`cosets`** - Vec\<Array1\<u8\>\>

      * **Coset (logical error)** corresponding to each correction.
      * Logical error patterns detected by the Observable decoder.

4.  **`selected_index`** - usize

      * **Index of the finally selected correction result.**
      * Indicates which child decoder's result was chosen.

5.  **`child_iterations`** - Vec\<usize\>

      * **Number of iterations used by each child decoder.**
      * Useful for comparing performance between child decoders.

6.  **`child_success`** - Vec\<bool\>

      * **Whether each child decoder successfully converged.**
      * Independent success/failure flags for each decoder.

7.  **`effective_iterations`** - Option\<f64\>

      * **Effective number of iterations** for the ensemble.
      * Usually `max(child_iterations)`; $+\infty$ if any child failed.

8.  **`selected_coset_avg_iter`** - Option\<f64\>

      * **Average number of iterations for the selected coset.**
      * Average of multiple child decoders that output the same coset.

9.  **`runner_up_coset_avg_iter`** - Option\<f64\>

      * **Average number of iterations for the second most voted coset.**

10. **`selected_coset_votes`** - Option\<usize\>

      * **Number of votes for the selected coset.**
      * Result of the majority decision in the MajorityVote strategy.

11. **`runner_up_coset_votes`** - Option\<usize\>

      * **Number of votes for the second coset.**

-----

### New Statistical Information (v2.0):

#### 12\. **`converged_count`** - usize

  * **Number of decoders that converged within the ensemble.**

  * Allows calculation of the convergence rate relative to the total ensemble size.

    ```python
    convergence_rate = extra['converged_count'] / ensemble_size
    ```

#### 13\. **`ensemble_posterior_ratios`** - Vec\<Array1\<f64\>\>

  * **Final marginals (LLR values) for each variable node in each child decoder.**

  * List length = ensemble size.

  * Each element is an `Array1<f64>` with a length equal to the number of variable nodes.

  * The element at index $k$ contains the LLR values of all variable nodes for the $k$-th child decoder.

    ```python
    # LLR of the i-th variable node for the k-th decoder
    llr_k_i = extra['ensemble_posterior_ratios'][k][i]
    ```

#### 14\. **`ensemble_mean_posterior_ratios`** - Option\<Array1\<f64\>\>

  * **Mean value of final marginals (LLR values) across all child decoders for each variable node.**

  * Array length = number of variable nodes.

  * Formula: `mean[i] = (1/N) * Σ_k L_k(i)`

      * $i$: Variable node index
      * $k$: Child decoder index
      * $L_k(i)$: Final marginal of the $i$-th variable node for the $k$-th decoder
      * $N$: Ensemble size

  * **Includes child decoders that did not converge.**

    ```python
    # Ensemble mean LLR for the i-th variable node
    mean_llr_i = extra['ensemble_mean_posterior_ratios'][i]
    ```

#### 15\. **`ensemble_std_posterior_ratios`** - Option\<Array1\<f64\>\>

  * **Standard deviation of final marginals for each variable node.**

  * Array length = number of variable nodes.

  * Formula: `std[i] = sqrt((1/N) * Σ_k (L_k(i) - mean[i])^2)`

  * **Includes child decoders that did not converge.**

  * Variable nodes with large values indicate divided opinions (high uncertainty) among decoders.

    ```python
    # Standard deviation of LLR for the i-th variable node
    std_llr_i = extra['ensemble_std_posterior_ratios'][i]

    # Identify highly uncertain nodes
    uncertain_nodes = np.where(std_llr_i > threshold)[0]
    ```

#### 16\. **`ensemble_iteration_dist`** - Vec\<usize\>

  * **Distribution of iteration counts for each child decoder.**

  * Same content as `child_iterations` (provided as an alias for consistency).

  * List length = ensemble size.

    ```python
    # Visualize as a histogram
    import matplotlib.pyplot as plt
    plt.hist(extra['ensemble_iteration_dist'], bins=20)
    ```

#### 17\. **`ensemble_mean_iteration`** - Option\<f64\>

  * **Average iteration count of child decoders.**

  * Formula: `mean = (1/N) * Σ_k iter_k`

  * **If a child decoder did not converge, its `max_iter` is used for calculation.**

    ```python
    avg_iterations = extra['ensemble_mean_iteration']
    ```

#### 18\. **`ensemble_std_iteration`** - Option\<f64\>

  * **Standard deviation of iteration counts of child decoders.**

  * Formula: `std = sqrt((1/N) * Σ_k (iter_k - mean)^2)`

  * Metric indicating variability in convergence speed.

    ```python
    # Evaluate variability in convergence speed
    iteration_variability = extra['ensemble_std_iteration']
    ```

#### 19\. **`residual_result`** - Option\<Vec\<Array1\<Bit\>\>\>

  * **Returns corrections from each child decoder if the entire ensemble fails to converge.**

  * Each correction is provisional based on the final marginals (1 if LLR \< 0, else 0).

  * **Returns `None` if at least one decoder converged.**

  * List length = ensemble size (only populated upon non-convergence).

    ```python
    if extra['residual_result'] is not None:
        # Case where all decoders failed
        # Get provisional corrections from each child decoder
        provisional_corrections = extra['residual_result']
        # Example: Determine final correction via majority vote
        final_correction = majority_vote(provisional_corrections)
    else:
        # At least one decoder converged
        # Use standard decoding results
        pass
    ```

-----

## Observable Decoder Additional Information

When using the Observable decoder, the following additional information is available:

1.  **`observables`** - Array1\<u8\>

      * Estimated logical error information.

2.  **`converged`** - bool

      * Decoding convergence in logical space.

3.  **`logical_gap`** - Option\<f64\>

      * Score difference between physical and logical spaces.

4.  **`error_detected`** - Option\<bool\>

      * Whether an actual error was detected.

5.  **`error_mismatch_detected`** - Option\<bool\>

      * Detection of mismatch with error estimation.

-----

## Usage Examples (Python)

### Basic Usage

```python
from relay_bp.observable_decoder import ObservableDecoderRunner
import numpy as np

# Decoder setup (e.g., Ensemble decoder)
decoder = ObservableDecoderRunner.with_ensemble_decoder(
    ensemble_size=20,
    check_matrix=H,
    observable_matrix=obs_matrix,
    error_priors=error_priors,
    # Other parameters...
)

# Execute decoding
result = decoder.decode(detectors)

# Get basic info
correction = result.observables
converged = result.converged
iterations = result.iterations

# Get Ensemble specific info
if result.extra is not None:
    extra = result.extra
    
    # Convergence info
    converged_count = extra['converged_count']
    convergence_rate = converged_count / ensemble_size
    print(f"Convergence rate: {convergence_rate:.2%}")
    
    # Iterations for each decoder
    child_iterations = extra['child_iterations']
    child_success = extra['child_success']
    
    # Statistical info
    mean_iter = extra['ensemble_mean_iteration']
    std_iter = extra['ensemble_std_iteration']
    print(f"Average iterations: {mean_iter:.2f} ± {std_iter:.2f}")
    
    # Stats per variable node
    mean_llr = extra['ensemble_mean_posterior_ratios']  # numpy array
    std_llr = extra['ensemble_std_posterior_ratios']    # numpy array
    
    # Identify highly uncertain nodes
    threshold = 1.0
    uncertain_nodes = np.where(std_llr > threshold)[0]
    print(f"Uncertain nodes (std > {threshold}): {len(uncertain_nodes)}")
    
    # Handling case where all decoders failed
    if extra['residual_result'] is not None:
        print("All decoders failed to converge")
        provisional_corrections = extra['residual_result']
        # Final decision via majority vote, etc.
        stacked = np.vstack(provisional_corrections)
        final_correction = (stacked.sum(axis=0) > len(provisional_corrections) / 2).astype(np.uint8)
```

### Detailed Statistical Analysis

```python
import matplotlib.pyplot as plt

# Execute decoding
result = decoder.decode(detectors)
extra = result.extra

# 1. Distribution of Iteration Counts
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(extra['ensemble_iteration_dist'], bins=20, alpha=0.7)
plt.axvline(extra['ensemble_mean_iteration'], color='r', linestyle='--', 
            label=f'Mean: {extra["ensemble_mean_iteration"]:.1f}')
plt.xlabel('Iterations')
plt.ylabel('Frequency')
plt.title('Iteration Distribution')
plt.legend()

# 2. LLR Standard Deviation per Variable Node
plt.subplot(132)
std_llr = extra['ensemble_std_posterior_ratios']
plt.plot(std_llr, alpha=0.7)
plt.xlabel('Variable Node Index')
plt.ylabel('LLR Std Dev')
plt.title('LLR Uncertainty per Node')

# 3. Mean LLR vs Standard Deviation
plt.subplot(133)
mean_llr = extra['ensemble_mean_posterior_ratios']
plt.scatter(np.abs(mean_llr), std_llr, alpha=0.5)
plt.xlabel('|Mean LLR|')
plt.ylabel('LLR Std Dev')
plt.title('Confidence vs Uncertainty')

plt.tight_layout()
plt.show()

# 4. Detailed Analysis of Each Decoder
ensemble_posterior_ratios = extra['ensemble_posterior_ratios']
for k, (llrs, success, iters) in enumerate(zip(
    ensemble_posterior_ratios, 
    extra['child_success'], 
    extra['child_iterations']
)):
    print(f"Decoder {k}: Success={success}, Iterations={iters}")
    print(f"  LLR range: [{llrs.min():.2f}, {llrs.max():.2f}]")
```

### Error Analysis

```python
# Analysis of nodes with high uncertainty
extra = result.extra
mean_llr = extra['ensemble_mean_posterior_ratios']
std_llr = extra['ensemble_std_posterior_ratios']

# High uncertainty (decoders disagree)
high_uncertainty = std_llr > np.percentile(std_llr, 90)

# Weak confidence (small |mean_llr|)
low_confidence = np.abs(mean_llr) < 1.0

# Problematic nodes satisfying both conditions
problematic_nodes = high_uncertainty & low_confidence

print(f"Problematic nodes: {np.sum(problematic_nodes)}")
print(f"Indices: {np.where(problematic_nodes)[0]}")

# Detailed analysis of these nodes
for idx in np.where(problematic_nodes)[0]:
    print(f"\nNode {idx}:")
    print(f"  Mean LLR: {mean_llr[idx]:.3f}")
    print(f"  Std LLR: {std_llr[idx]:.3f}")
    # LLR for this node across each child decoder
    node_llrs = [posterior[idx] for posterior in extra['ensemble_posterior_ratios']]
    print(f"  LLR distribution: {node_llrs}")
```

-----

## Notes & Precautions

1.  **Sign of `posterior_ratios`**: If LLR \< 0, the bit is likely an error (high probability of being 1).
2.  **Handling Non-convergence**: In `ensemble_mean_iteration` and `ensemble_std_iteration`, child decoders that did not converge are treated as having used `max_iter`.
3.  **Using `residual_result`**: This can be used as a fallback measure to combine provisional corrections from each decoder when all fail.
4.  **Memory Usage**: Since `ensemble_posterior_ratios` holds all LLR values for all child decoders, memory usage may become significant with large codes or large ensemble sizes.

-----

## Version History

  * **v2.0** (2025-12-10): Added statistical information fields.

      * `converged_count`
      * `ensemble_posterior_ratios`
      * `ensemble_mean_posterior_ratios`
      * `ensemble_std_posterior_ratios`
      * `ensemble_iteration_dist`
      * `ensemble_mean_iteration`
      * `ensemble_std_iteration`
      * `residual_result`

  * **v1.0**: Initial implementation.

      * Basic ensemble information.
      * Coset voting information.