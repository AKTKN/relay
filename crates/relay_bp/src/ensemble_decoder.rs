// Import necessary components and modules
use crate::decoder::{Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix, Mod2Mul}; // Besic type and trait imports
use crate::decoder::{BPExtraResult, EnsembleExtraResult};
use ndarray::{Array1, ArrayView1}; use core::f64;
// 1-dimentional ndarray crate for error and syndrome vecrtors
use std::sync::Arc; // Smart pointer for shared owenership, this is useful for sharing data like chack matrices across multiple decoders.

/// Ensemble decoder mode
#[derive(Clone, Debug, PartialEq)]
pub enum EnsembleMode {
    Normal,
    Repulsive,
}

impl Default for EnsembleMode {
    fn default() -> Self {
        EnsembleMode::Normal
    }
}

/// Configuration for repulsive mode
#[derive(Clone, Debug)]
pub struct RepulsiveConfig {
    pub repulsive_size: usize,           // Number of decoders to use repulsive mode
    pub repulsive_gamma_dist: (f64, f64), // Gamma distribution range for repulsive mode
    pub abs_llr_threshold: f64,           // Absolute LLR threshold for applying repulsive gamma
    pub pulse_per_leg: usize,             // Apply repulsive every N legs
    pub start_leg: usize,                 // Start applying repulsive from this leg (0 = after pre_iter)
}

impl Default for RepulsiveConfig {
    fn default() -> Self {
        Self {
            repulsive_size: 0,
            repulsive_gamma_dist: (-0.5, -0.1),
            abs_llr_threshold: 2.0,
            pulse_per_leg: 1,
            start_leg: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SelectionStrategy{
    MostLikely,
    MajorityVote,
}

impl Default for SelectionStrategy{
    fn default() -> Self {
        SelectionStrategy::MostLikely
    }
}

// EnsembleDecoder struct definition
// store child decoders in a `decoders` field, which is a vector of boxed `Decoder` trait objects.
// - Vec<T>: A growable array type provided by the Rust standard library.
// - Box<T>: A smart pointer for heap allcation, allowing for dynamic dispatch of trait objects.
// - dyn Decoder: A trait object representig any type that implements the `Decoder` trait.
// - + Send: A marker trait indicating that the type can be safely transfered across thread boundaries.
#[derive(Clone)]
#[allow(dead_code)]
pub struct EnsembleDecoder{
    decoders: Vec<Box<dyn Decoder + Send>>,
    strategy: SelectionStrategy,
    original_log_priors: Option<Arc<Array1<f64>>>,
    observable_matrix: Option<Arc<SparseBitMatrix>>,
    selection_max_index: Option<usize>,
    mode: EnsembleMode,
    repulsive_config: Option<Arc<RepulsiveConfig>>,
}

// Implement the constructer for EnsembleDecoder
impl EnsembleDecoder{
    /// Create a new EnsembleDecoder 
    pub fn new(
        decoders: Vec<Box<dyn Decoder + Send>>,
        strategy: SelectionStrategy,
        original_log_priors: Option<Arc<Array1<f64>>>,
        observable_matrix: Option<Arc<SparseBitMatrix>>,
        selection_max_index: Option<usize>,
    ) -> Self{
        if decoders.is_empty(){
            panic!("EnsembleDecoder requires at least one decoder.");
        }
        // 戦略と必要な情報が一致しているか簡単なチェック
        if strategy == SelectionStrategy::MostLikely && original_log_priors.is_none() {
            panic!("'MostLikely' strategy requires original_log_priors.");
        }
        if strategy == SelectionStrategy::MajorityVote && observable_matrix.is_none() {
            panic!("'MajorityVote' strategy requires observable_matrix.");
        }

        Self { 
            decoders,
            strategy,
            original_log_priors,
            observable_matrix,
            selection_max_index,
            mode: EnsembleMode::Normal,
            repulsive_config: None,
        } 
    }

    /// Create a new EnsembleDecoder with mode configuration
    pub fn new_with_mode(
        decoders: Vec<Box<dyn Decoder + Send>>,
        strategy: SelectionStrategy,
        original_log_priors: Option<Arc<Array1<f64>>>,
        observable_matrix: Option<Arc<SparseBitMatrix>>,
        selection_max_index: Option<usize>,
        mode: EnsembleMode,
        repulsive_config: Option<Arc<RepulsiveConfig>>,
    ) -> Self{
        if decoders.is_empty(){
            panic!("EnsembleDecoder requires at least one decoder.");
        }
        // 戦略と必要な情報が一致しているか簡単なチェック
        if strategy == SelectionStrategy::MostLikely && original_log_priors.is_none() {
            panic!("'MostLikely' strategy requires original_log_priors.");
        }
        if strategy == SelectionStrategy::MajorityVote && observable_matrix.is_none() {
            panic!("'MajorityVote' strategy requires observable_matrix.");
        }
        // Repulsiveモードの検証
        if mode == EnsembleMode::Repulsive && repulsive_config.is_none() {
            panic!("Repulsive mode requires repulsive_config.");
        }
        if let Some(ref config) = repulsive_config {
            if config.repulsive_size > decoders.len() {
                panic!("repulsive_size cannot exceed the number of decoders.");
            }
        }

        Self { 
            decoders,
            strategy,
            original_log_priors,
            observable_matrix,
            selection_max_index,
            mode,
            repulsive_config,
        } 
    }
}

// Implement the Decoder trait for EnsembleDecoder
impl Decoder for EnsembleDecoder{
    /// return check matrix
    // In the future, we may want to get automorphism group og the check matrix
    fn check_matrix(&self) -> Arc<SparseBitMatrix>{
        self.decoders[0].check_matrix()
    }

    /// return log prior ratios of prior error probabilities
    /// In the future, we may want to add some pertubation to the prior ratios
    fn log_prior_ratios(&mut self) -> Array1<f64>{
        self.decoders[0].log_prior_ratios()
    }

    // Return final decoding result after running multiple decoders
    // execute decoding for each decoder in the ensemble and collect their results.
    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult{
        // 1. Collect results from all child decoders
        let results: Vec<DecodeResult> = self
            .decoders
            .iter_mut()
            .map(|decoder| decoder.decode_detailed(detectors.view()))
            .collect();

        // Calculate max runtime among children
        let max_runtime_micros = results.iter()
            .filter_map(|r| r.run_time_micros)
            .max();

        let first_result = results.first()
            .cloned()
            .expect("EnsembleDecoder requires at least one decoder.");

        // Aggregate per-child iterations and success 
        let child_iterations: Vec<usize> = results.iter().map(|r| r.iterations).collect();
        let child_success: Vec<bool> = results.iter().map(|r| r.success).collect();
        let any_non_converged = child_success.iter().any(|&s| !s);
        let max_child_iters = child_iterations.iter().copied().max().unwrap_or(0);
        let effective_iterations = if any_non_converged { f64::INFINITY } else { max_child_iters as f64 };

        // Calculate ensemble-wide statistics
        let converged_count = child_success.iter().filter(|&&s| s).count();
        
        // Collect posterior ratios from all child decoders
        let ensemble_posterior_ratios: Vec<Array1<f64>> = results.iter()
            .map(|r| r.posterior_ratios.clone())
            .collect();
        
        // Calculate mean and std of posterior ratios across all children
        let num_variables = ensemble_posterior_ratios.first()
            .map(|arr| arr.len())
            .unwrap_or(0);
        
        let ensemble_mean_posterior_ratios = if !ensemble_posterior_ratios.is_empty() && num_variables > 0 {
            let mut means = Array1::<f64>::zeros(num_variables);
            for i in 0..num_variables {
                let sum: f64 = ensemble_posterior_ratios.iter()
                    .map(|arr| arr[i])
                    .sum();
                means[i] = sum / ensemble_posterior_ratios.len() as f64;
            }
            Some(means)
        } else {
            None
        };
        
        let ensemble_std_posterior_ratios = if let Some(ref means) = ensemble_mean_posterior_ratios {
            let mut stds = Array1::<f64>::zeros(num_variables);
            for i in 0..num_variables {
                let variance: f64 = ensemble_posterior_ratios.iter()
                    .map(|arr| {
                        let diff = arr[i] - means[i];
                        diff * diff
                    })
                    .sum::<f64>() / ensemble_posterior_ratios.len() as f64;
                stds[i] = variance.sqrt();
            }
            Some(stds)
        } else {
            None
        };
        
        // Calculate iteration statistics (using max_iter for non-converged decoders)
        let ensemble_iteration_dist = child_iterations.clone();
        let ensemble_mean_iteration = if !child_iterations.is_empty() {
            let sum: f64 = child_iterations.iter()
                .map(|&iter| iter as f64)
                .sum();
            Some(sum / child_iterations.len() as f64)
        } else {
            None
        };
        
        let ensemble_std_iteration = if let Some(mean_iter) = ensemble_mean_iteration {
            let variance: f64 = child_iterations.iter()
                .map(|&iter| {
                    let diff = iter as f64 - mean_iter;
                    diff * diff
                })
                .sum::<f64>() / child_iterations.len() as f64;
            Some(variance.sqrt())
        } else {
            None
        };

        // Collect all corrections, LLR sums, and cosets
        let mut all_corrections = Vec::with_capacity(results.len());
        let mut llr_sums = Vec::with_capacity(results.len());
        let mut cosets = Vec::with_capacity(results.len());
        
        let obs_matrix_opt = self.observable_matrix.as_ref();
        let log_priors_opt = self.original_log_priors.as_ref();

        for result in &results {
            all_corrections.push(result.decoding.clone());
            
            let llr_sum = if let Some(log_priors) = log_priors_opt {
                 result.decoding.iter().zip(log_priors.iter())
                    .filter(|(&c, _)| c == 1)
                    .map(|(_, &llr_val)| llr_val)
                    .sum::<f64>()
            } else {
                0.0
            };
            llr_sums.push(llr_sum);
            
            let coset = if let Some(obs_matrix) = obs_matrix_opt {
                obs_matrix.mul_mod2(&result.decoding)
            } else {
                Array1::zeros(0)
            };
            cosets.push(coset);
        }

        let converged_results: Vec<DecodeResult> = results.iter().cloned().filter(|r| r.success).collect();
        if converged_results.is_empty()  {
            let mut res = first_result;
            res.logical_gap = None;
            res.run_time_micros = max_runtime_micros;

            // Overwrite final iterations to the max of children; effective as +inf if any failed.
            res.iterations = max_child_iters;

            // Collect residual results (provisional corrections based on final marginals)
            let residual_result: Vec<Array1<Bit>> = ensemble_posterior_ratios.iter()
                .map(|posterior| {
                    posterior.mapv(|llr| if llr < 0.0 { 1 } else { 0 })
                })
                .collect();

            // Populate ensemble extra with statistics
            res.extra = BPExtraResult::Ensemble(EnsembleExtraResult{
                all_corrections,
                llr_sums,
                cosets,
                selected_index: 0,
                child_iterations: child_iterations.clone(),
                child_success: child_success.clone(),
                effective_iterations: Some(effective_iterations),
                selected_coset_avg_iter: None,
                runner_up_coset_avg_iter: None,
                selected_coset_votes: None,
                runner_up_coset_votes: None,
                converged_count,
                ensemble_posterior_ratios,
                ensemble_mean_posterior_ratios,
                ensemble_std_posterior_ratios,
                ensemble_iteration_dist,
                ensemble_mean_iteration,
                ensemble_std_iteration,
                residual_result: Some(residual_result),
            });
            return res;
        }

        // Selection range for final decision (metrics still use all results)
        let total_results = results.len();
        let selection_max = self.selection_max_index.map(|m| m.min(total_results.saturating_sub(1)));
        let selection_indices: Vec<usize> = match selection_max {
            Some(m) => (0..=m).collect(),
            None => (0..total_results).collect(),
        };
        let selection_converged_indices: Vec<usize> = selection_indices
            .iter()
            .copied()
            .filter(|&i| results[i].success)
            .collect();
        let all_converged_indices: Vec<usize> = results
            .iter()
            .enumerate()
            .filter(|(_, r)| r.success)
            .map(|(i, _)| i)
            .collect();
        let selection_indices_for_choice = if selection_converged_indices.is_empty() {
            all_converged_indices.clone()
        } else {
            selection_converged_indices
        };

        // Pre-computation for soft-information
        use std::collections::HashMap;
        // We already have obs_matrix_opt and log_priors_opt, but let's unwrap them as they are required for converged results
        let _obs_matrix = obs_matrix_opt.expect("observable_matrix is required for decoding.");
        let _log_priors = log_priors_opt.expect("original_log_priors is required for decoding.");

        // This HashMap will store (index, DecodeResult, llrCost, iter) tuples, keyed by logical_error
        let mut coset_groups_all: HashMap<Array1<Bit>, Vec<(usize, DecodeResult, f64, usize)>> = HashMap::new();
        let mut coset_groups_sel: HashMap<Array1<Bit>, Vec<(usize, DecodeResult, f64, usize)>> = HashMap::new();

        // Calculate LLR costs and group results by coset (all converged)
        for &i in &all_converged_indices {
            let result = &results[i];
            let logical_error = cosets[i].clone();
            let llr_cost = llr_sums[i];
            let iter = result.iterations;
            coset_groups_all.entry(logical_error).or_default().push((i, result.clone(), llr_cost, iter));
        }

        // Calculate LLR costs and group results by coset (selection range only)
        for &i in &selection_indices_for_choice {
            let result = &results[i];
            if result.success {
                let logical_error = cosets[i].clone();
                let llr_cost = llr_sums[i];
                let iter = result.iterations;
                coset_groups_sel.entry(logical_error).or_default().push((i, result.clone(), llr_cost, iter));
            }
        }

        // Find the minimum LLR within each coset and calculate statistics
        let mut coset_stats_all: HashMap<Array1<Bit>, (usize, DecodeResult, f64, f64, usize)> = HashMap::new();
        let mut coset_stats_sel: HashMap<Array1<Bit>, (usize, DecodeResult, f64, f64, usize)> = HashMap::new();
        
        for (coset, results_in_coset) in coset_groups_all.iter() {
            let (best_idx, best_result, min_llr) = results_in_coset.iter()
                .min_by(|(_, _, llr_a, _), (_, _, llr_b, _)| llr_a.partial_cmp(llr_b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, r, l, _)| (*i, r.clone(), *l))
                .unwrap();
            
            // Calculate average iteration for this coset
            let avg_iter = results_in_coset.iter()
                .map(|(_, _, _, iter)| *iter as f64)
                .sum::<f64>() / results_in_coset.len() as f64;
            
            let vote_count = results_in_coset.len();
            
            coset_stats_all.insert(coset.clone(), (best_idx, best_result, min_llr, avg_iter, vote_count));
        }

        for (coset, results_in_coset) in coset_groups_sel.iter() {
            let (best_idx, best_result, min_llr) = results_in_coset.iter()
                .min_by(|(_, _, llr_a, _), (_, _, llr_b, _)| llr_a.partial_cmp(llr_b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, r, l, _)| (*i, r.clone(), *l))
                .unwrap();
            
            // Calculate average iteration for this coset
            let avg_iter = results_in_coset.iter()
                .map(|(_, _, _, iter)| *iter as f64)
                .sum::<f64>() / results_in_coset.len() as f64;
            
            let vote_count = results_in_coset.len();
            
            coset_stats_sel.insert(coset.clone(), (best_idx, best_result, min_llr, avg_iter, vote_count));
        }

        // 2. Select the final correction based on the chosen strategy
        match self.strategy {
            SelectionStrategy::MostLikely => {
                // Find the result with the overall minimum LLR cost among CONVERGED results
                // We iterate directly over results to ensure index alignment and avoid grouping issues.
                let (best_idx, mut best_result) = selection_indices_for_choice.iter()
                    .map(|&i| (i, results[i].clone()))
                    .min_by(|(i, _), (j, _)| {
                        llr_sums[*i].partial_cmp(&llr_sums[*j]).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("No converged results to compare for MostLikely strategy.");

                let llr_final = llr_sums[best_idx];
                let best_coset = &cosets[best_idx];

                // Calculate stats for the selected coset
                let mut selected_iter_sum = 0.0;
                let mut selected_votes = 0;
                for (i, r) in results.iter().enumerate() {
                    if r.success && &cosets[i] == best_coset {
                        selected_iter_sum += r.iterations as f64;
                        selected_votes += 1;
                    }
                }
                let selected_avg_iter = selected_iter_sum / selected_votes as f64;

                // Find the next smallest LLR from a *different* coset
                let runner_up = results.iter().enumerate()
                    .filter(|(i, r)| r.success && &cosets[*i] != best_coset)
                    .min_by(|(i, _), (j, _)| {
                        llr_sums[*i].partial_cmp(&llr_sums[*j]).unwrap_or(std::cmp::Ordering::Equal)
                    });

                let (soft_information, runner_up_avg_iter, runner_up_votes) = if let Some((runner_idx, _)) = runner_up {
                    let runner_llr = llr_sums[runner_idx];
                    let gap = runner_llr - llr_final;
                    
                    let runner_coset = &cosets[runner_idx];
                    let mut runner_iter_sum = 0.0;
                    let mut runner_votes = 0;
                    for (i, r) in results.iter().enumerate() {
                        if r.success && &cosets[i] == runner_coset {
                            runner_iter_sum += r.iterations as f64;
                            runner_votes += 1;
                        }
                    }
                    let runner_avg = runner_iter_sum / runner_votes as f64;
                    
                    (gap, Some(runner_avg), Some(runner_votes))
                } else {
                    (f64::INFINITY, None, None)
                };

                best_result.logical_gap = Some(soft_information);

                best_result.extra = BPExtraResult::Ensemble(EnsembleExtraResult{
                    all_corrections,
                    llr_sums,
                    cosets,
                    selected_index: best_idx,
                    child_iterations: child_iterations.clone(),
                    child_success: child_success.clone(),
                    effective_iterations: Some(effective_iterations),
                    selected_coset_avg_iter: Some(selected_avg_iter),
                    runner_up_coset_avg_iter: runner_up_avg_iter,
                    selected_coset_votes: Some(selected_votes),
                    runner_up_coset_votes: runner_up_votes,
                    converged_count,
                    ensemble_posterior_ratios,
                    ensemble_mean_posterior_ratios,
                    ensemble_std_posterior_ratios,
                    ensemble_iteration_dist,
                    ensemble_mean_iteration,
                    ensemble_std_iteration,
                    residual_result: None,  // At least one decoder converged
                });

                best_result.run_time_micros = max_runtime_micros;
                best_result
            }
            
            // --- MajorityVote Strategy ---
            SelectionStrategy::MajorityVote => {
                // Find the winning coset (the one with the most votes)
                let (winning_coset, (selected_idx, mut final_result, llr_final, _sel_avg_iter, _sel_votes)) = coset_stats_sel.iter()
                    .max_by_key(|(_, (_, _, _, _, votes))| *votes)
                    .map(|(c, (idx, r, l, avg_i, votes))| (c.clone(), (*idx, r.clone(), *l, *avg_i, *votes)))
                    .expect("No converged results to vote on for MajorityVote strategy.");

                let (selected_avg_iter, selected_votes) = coset_stats_all
                    .get(&winning_coset)
                    .map(|(_, _, _, avg_i, votes)| (*avg_i, *votes))
                    .unwrap_or((0.0, 0));

                // Find the second most voted coset
                let runner_up = coset_stats_all.iter()
                    .filter(|(coset, _)| *coset != &winning_coset)
                    .max_by_key(|(_, (_, _, _, _, votes))| *votes);

                let (soft_information, runner_up_avg_iter, runner_up_votes) = if let Some((_, (_, _, runner_llr, runner_avg_iter, runner_votes))) = runner_up {
                    let gap = *runner_llr - llr_final;
                    (gap, Some(*runner_avg_iter), Some(*runner_votes))
                } else {
                    (f64::INFINITY, None, None)
                };

                final_result.logical_gap = Some(soft_information);

                final_result.extra = BPExtraResult::Ensemble(EnsembleExtraResult{
                    all_corrections,
                    llr_sums,
                    cosets,
                    selected_index: selected_idx,
                    child_iterations: child_iterations.clone(),
                    child_success: child_success.clone(),
                    effective_iterations: Some(effective_iterations),
                    selected_coset_avg_iter: Some(selected_avg_iter),
                    runner_up_coset_avg_iter: runner_up_avg_iter,
                    selected_coset_votes: Some(selected_votes),
                    runner_up_coset_votes: runner_up_votes,
                    converged_count,
                    ensemble_posterior_ratios,
                    ensemble_mean_posterior_ratios,
                    ensemble_std_posterior_ratios,
                    ensemble_iteration_dist,
                    ensemble_mean_iteration,
                    ensemble_std_iteration,
                    residual_result: None,  // At least one decoder converged
                });

                final_result.run_time_micros = max_runtime_micros;
                final_result
            }
        }
        // match self.strategy{
        //     // --- MostLikely Strategy ---
        //     // Selects the single result with the minimum LLR cost.
        //     SelectionStrategy::MostLikely =>{
        //         // Get the log-priors, which must be log((1-p)/p) (positive costs).
        //         let log_priors = self.original_log_priors.as_ref().unwrap();

        //         converged_results.into_iter()
        //             // Find the result with the minimum cost by comparing pairs.
        //             .min_by(|a, b| {
        //                 // Calculate LLR cost for 'a'
        //                 // Cost = sum_{i where c_i=1} log((1-p_i)/p_i)
        //                 let llr_cost_a = a.decoding.iter().zip(log_priors.iter())
        //                     .filter(|(&c, _)| c == 1) // Find indices where error bit c_i is 1
        //                     .map(|(_, &llr_cost)| llr_cost) // Get the corresponding cost
        //                     .sum::<f64>();
                        
        //                 // Calculate LLR cost for 'b'
        //                 let llr_cost_b = b.decoding.iter().zip(log_priors.iter())
        //                     .filter(|(&c, _)| c == 1)
        //                     .map(|(_, &llr_cost)| llr_cost)
        //                     .sum::<f64>();
                        
        //                 // Compare f64 costs using partial_cmp
        //                 llr_cost_a.partial_cmp(&llr_cost_b).unwrap_or(std::cmp::Ordering::Equal)
        //             })
        //             .expect("No results to compare for MostLikely strategy.")
        //     }
            
        //     // --- MajorityVote Strategy ---
        //     // 1. All results vote for a logical coset.
        //     // 2. The coset with the most votes wins.
        //     // 3. From the winning coset, select the result with the minimum LLR cost.
        //     SelectionStrategy::MajorityVote => {
        //         use std::collections::HashMap;
        //         let obs_matrix = self.observable_matrix.as_ref().unwrap();
        //         // Get log-priors, as they are needed for the final LLR cost comparison.
        //         let log_priors = self.original_log_priors.as_ref()
        //             .expect("original_log_priors is required for MajorityVote strategy.");

        //         // This HashMap will store (DecodeResult, LLR_Cost) tuples, keyed by logical_error
        //         let mut coset_votes: HashMap<Array1<Bit>, Vec<(DecodeResult, f64)>> = HashMap::new();

        //         // --- Step 1 & 2: Calculate LLR costs and vote by coset in one pass ---
        //         for result in converged_results {
        //             // Determine the logical coset for this result
        //             let logical_error = obs_matrix.mul_mod2(&result.decoding);
                    
        //             // Calculate the LLR cost for this result
        //             let llr_cost = result.decoding.iter().zip(log_priors.iter())
        //                 .filter(|(&c, _)| c == 1)
        //                 .map(|(_, &llr_val)| llr_val)
        //                 .sum::<f64>();
                        
        //             // Add the (result, cost) tuple to the corresponding coset's vector
        //             coset_votes.entry(logical_error).or_default().push((result, llr_cost));
        //         }

        //         // --- Step 3: Find the winning coset (the one with the most votes) ---
        //         let winning_coset_results = coset_votes.into_values()
        //             .max_by_key(|v| v.len())
        //             .expect("No converged results to vote on for MajorityVote strategy.");

        //         // --- Step 4: Select the result with the minimum LLR cost from the winning coset ---
        //         let (final_result, _final_llr_cost) = winning_coset_results.into_iter()
        //             .min_by(|(_, llr_a), (_, llr_b)| { 
        //                 // Compare the f64 LLR costs
        //                 llr_a.partial_cmp(llr_b).unwrap_or(std::cmp::Ordering::Equal)
        //             })
        //             .expect("Winning coset was empty."); // This gives the (DecodeResult, f64) tuple
                
        //         final_result // Return only the DecodeResult part
        //     }
        // }
    }
    
    // fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult{
    //     let results: Vec<DecodeResult> = self
    //         .decoders
    //         .iter_mut()
    //         .map(|decoder| decoder.decode_detailed(detectors.view()))
    //         .collect();

    //     let first_result = results.first()
    //         .cloned()
    //         .expect("EnsembleDecoder requires at least one decoder.");
    //     let converged_results: Vec<DecodeResult> = results.into_iter().filter(|r| r.success).collect();
    //     if converged_results.is_empty()  {
    //         return first_result;
    //     }

    //     // Here we decide final correction 
    //     match self.strategy{
    //         SelectionStrategy::MostLikely =>{
    //             let log_priors = self.original_log_priors.as_ref().unwrap();

    //             // debug
    //             println!("Error priors (log ratios): {:?}", log_priors);

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ1: 各デコード結果とLLRコストを計算
    //             // -----------------------------------------------------------------
    //             let results_with_llr: Vec<(DecodeResult, f64)> = converged_results.into_iter().map(|result| {
    //                 let llr_cost = result.decoding.iter().zip(log_priors.iter())
    //                     .filter(|(&c, _)| c == 1) // エラーがある(c=1)のインデックスのみ
    //                     .map(|(_, &llr_val)| llr_val)     // 対応するlog_prior (コスト) を取得
    //                     .sum::<f64>();
    //                 (result, llr_cost)
    //             }).collect();

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ2: 全デコード結果の情報を出力
    //             // -----------------------------------------------------------------
    //             println!("\n--- [EnsembleDecoder Debug: MostLikely] ---");
    //             println!("Total results received: {}", results_with_llr.len());
                
    //             for (i, (result, llr_cost)) in results_with_llr.iter().enumerate() {
    //                 let error_indices: Vec<usize> = result.decoding.iter().enumerate()
    //                     .filter(|(_, &bit)| bit == 1)
    //                     .map(|(index, _)| index)
    //                     .collect();
    //                 let weight = error_indices.len();

    //                 println!("  Result {}: Weight = {}, LLR Cost = {:.6}", i, weight, llr_cost);
    //                 // エラーベクトル全体は巨大すぎる可能性があるため、エラー(1)のインデックスのみ表示
    //                 println!("    Error Indices: {:?}", error_indices);
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ3: 最小LLRのものを選択
    //             // -----------------------------------------------------------------
    //             // 計算済みのLLRを使って最小のものを探す
    //             let (final_result, final_llr) = results_with_llr.into_iter()
    //                 .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    //                 .expect("No results to compare for MostLikely strategy.");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ4: 最終選択の情報を出力
    //             // -----------------------------------------------------------------
    //             let final_weight = final_result.decoding.sum();
    //             println!("Final Selection: Weight = {}, LLR Cost = {:.6}", final_weight, final_llr);
    //             println!("----------------------------------------------\n");

    //             final_result
    //         }
            
    //         SelectionStrategy::MajorityVote => {
    //             use std::collections::HashMap;
    //             let obs_matrix = self.observable_matrix.as_ref().unwrap();
    //             let log_priors = self.original_log_priors.as_ref().expect("original_log_priors is required for MajorityVote strategy.");
                
    //             println!("Error priors (log ratios): {:?}", log_priors);
    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ1: 各デコード結果と論理エラーを収集
    //             // -----------------------------------------------------------------
    //             // debug
    //             let mut decoded_data = Vec::new();
    //             for (i, result) in converged_results.into_iter().enumerate() {
    //                 let logical_error = obs_matrix.mul_mod2(&result.decoding);
    //                 let weight = result.decoding.iter().zip(log_priors.iter())
    //                     .filter(|(&c, _)| c == 1)
    //                     .map(|(_, &llr_val)| llr_val)
    //                     .sum::<f64>();
    //                 decoded_data.push((i, result, logical_error, weight));
    //             }

    //             println!("\n--- [EnsembleDecoder Debug: MajorityVote] ---");
    //             println!("Total results received: {}", decoded_data.len());
    //             for (i, result, logical_error, weight) in &decoded_data {
    //                 let logical_error_vec: Vec<Bit> = logical_error.to_vec();
    //                 println!("  Result {}: Weight = {}, Logical Error = {:?}", i, *weight, logical_error_vec);
                    
    //                 let error_indices: Vec<usize> = result.decoding.iter().enumerate()
    //                     .filter(|(_, &bit)| bit == 1)
    //                     .map(|(index, _)| index)
    //                     .collect();
    //                 println!("    Error Indices: {:?}", error_indices);
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ2: コセットごとに投票
    //             // -----------------------------------------------------------------
    //             // (Result, Weight) のタプルを保存して、後の最小重み比較の計算を省略
    //             let mut coset_votes: HashMap<Array1<Bit>, Vec<(DecodeResult, f64)>> = HashMap::new();
    //             for (_, result, logical_error, weight) in decoded_data {
    //                 coset_votes.entry(logical_error).or_default().push((result, weight));
    //             }

    //             println!("Coset Voting Results:");
    //             for (logical_error, results_in_coset) in coset_votes.iter() {
    //                 let logical_error_vec: Vec<Bit> = logical_error.to_vec();
    //                 println!("  Coset {:?}: {} votes", logical_error_vec, results_in_coset.len());
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ3: 勝利コセットの決定
    //             // -----------------------------------------------------------------
    //             let (winning_coset_logical_error, winning_coset_results) = coset_votes.into_iter()
    //                 .max_by_key(|(_, v)| v.len())
    //                 .map(|(key, value)| (key, value)) // HashMapのエントリからタプルに変換
    //                 .expect("No results to vote on for MajorityVote strategy.");
                
    //             let winning_logical_error_vec: Vec<Bit> = winning_coset_logical_error.to_vec();
    //             println!("Winning Coset: {:?} (with {} votes)", winning_logical_error_vec, winning_coset_results.len());


    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ4: 勝利コセット内で最小重みを選択
    //             // -----------------------------------------------------------------
    //             // 保存しておいた重み `weight` で比較
    //             let (final_result, final_weight) = winning_coset_results.into_iter()
    //                 // min_by は Option<(DecodeResult, f64)> を返す
    //                 .min_by(|(_, llr_a), (_, llr_b)| { 
    //                     // f64 (LLRコスト) 同士を直接比較
    //                     llr_a.partial_cmp(llr_b).unwrap_or(std::cmp::Ordering::Equal)
    //                 })
    //                 // Option をアンラップして (DecodeResult, f64) を取り出す
    //                 .expect("Winning coset was empty.");

    //             // final_weight が LLR コスト
    //             println!("Final Selection: LLR Cost = {:.6}, From Coset = {:?}", final_weight, winning_logical_error_vec);
    //             println!("----------------------------------------------------------------\n");

    //             final_result
    //         }
    //     }
    // }
}
    // fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult{
    //     let results: Vec<DecodeResult> = self
    //         .decoders
    //         .iter_mut()
    //         .map(|decoder| decoder.decode_detailed(detectors.view()))
    //         .collect();

    //     // Here we decide final correction 
    //     match self.strategy{
    //         SelectionStrategy::MostLikely =>{
    //             let log_priors = self.original_log_priors.as_ref().unwrap();

    //             // debug
    //             println!("Error priors (log ratios): {:?}", log_priors);

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ1: 各デコード結果とLLRコストを計算
    //             // -----------------------------------------------------------------
    //             let results_with_llr: Vec<(DecodeResult, f64)> = results.into_iter().map(|result| {
    //                 let llr_cost = result.decoding.iter().zip(log_priors.iter())
    //                     .filter(|(&c, _)| c == 1) // エラーがある(c=1)のインデックスのみ
    //                     .map(|(_, &llr_val)| llr_val)     // 対応するlog_prior (コスト) を取得
    //                     .sum::<f64>();
    //                 (result, llr_cost)
    //             }).collect();

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ2: 全デコード結果の情報を出力
    //             // -----------------------------------------------------------------
    //             println!("\n--- [EnsembleDecoder Debug: MostLikely] ---");
    //             println!("Total results received: {}", results_with_llr.len());
                
    //             for (i, (result, llr_cost)) in results_with_llr.iter().enumerate() {
    //                 let error_indices: Vec<usize> = result.decoding.iter().enumerate()
    //                     .filter(|(_, &bit)| bit == 1)
    //                     .map(|(index, _)| index)
    //                     .collect();
    //                 let weight = error_indices.len();

    //                 println!("  Result {}: Weight = {}, LLR Cost = {:.6}", i, weight, llr_cost);
    //                 // エラーベクトル全体は巨大すぎる可能性があるため、エラー(1)のインデックスのみ表示
    //                 println!("    Error Indices: {:?}", error_indices);
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ3: 最小LLRのものを選択
    //             // -----------------------------------------------------------------
    //             // 計算済みのLLRを使って最小のものを探す
    //             let (final_result, final_llr) = results_with_llr.into_iter()
    //                 .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    //                 .expect("No results to compare for MostLikely strategy.");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ4: 最終選択の情報を出力
    //             // -----------------------------------------------------------------
    //             let final_weight = final_result.decoding.sum();
    //             println!("Final Selection: Weight = {}, LLR Cost = {:.6}", final_weight, final_llr);
    //             println!("----------------------------------------------\n");

    //             final_result
    //         }
            
    //         SelectionStrategy::MajorityVote => {
    //             use std::collections::HashMap;
    //             let obs_matrix = self.observable_matrix.as_ref().unwrap();
    //             let log_priors = self.original_log_priors.as_ref().expect("original_log_priors is required for MajorityVote strategy.");
                
    //             println!("Error priors (log ratios): {:?}", log_priors);
    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ1: 各デコード結果と論理エラーを収集
    //             // -----------------------------------------------------------------
    //             // debug
    //             let mut decoded_data = Vec::new();
    //             for (i, result) in results.into_iter().enumerate() {
    //                 let logical_error = obs_matrix.mul_mod2(&result.decoding);
    //                 let weight = result.decoding.iter().zip(log_priors.iter())
    //                     .filter(|(&c, _)| c == 1)
    //                     .map(|(_, &llr_val)| llr_val)
    //                     .sum::<f64>();
    //                 decoded_data.push((i, result, logical_error, weight));
    //             }

    //             println!("\n--- [EnsembleDecoder Debug: MajorityVote] ---");
    //             println!("Total results received: {}", decoded_data.len());
    //             for (i, result, logical_error, weight) in &decoded_data {
    //                 let logical_error_vec: Vec<Bit> = logical_error.to_vec();
    //                 println!("  Result {}: Weight = {}, Logical Error = {:?}", i, *weight, logical_error_vec);
                    
    //                 let error_indices: Vec<usize> = result.decoding.iter().enumerate()
    //                     .filter(|(_, &bit)| bit == 1)
    //                     .map(|(index, _)| index)
    //                     .collect();
    //                 println!("    Error Indices: {:?}", error_indices);
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ2: コセットごとに投票
    //             // -----------------------------------------------------------------
    //             // (Result, Weight) のタプルを保存して、後の最小重み比較の計算を省略
    //             let mut coset_votes: HashMap<Array1<Bit>, Vec<(DecodeResult, f64)>> = HashMap::new();
    //             for (_, result, logical_error, weight) in decoded_data {
    //                 coset_votes.entry(logical_error).or_default().push((result, weight));
    //             }

    //             println!("Coset Voting Results:");
    //             for (logical_error, results_in_coset) in coset_votes.iter() {
    //                 let logical_error_vec: Vec<Bit> = logical_error.to_vec();
    //                 println!("  Coset {:?}: {} votes", logical_error_vec, results_in_coset.len());
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ3: 勝利コセットの決定
    //             // -----------------------------------------------------------------
    //             let (winning_coset_logical_error, winning_coset_results) = coset_votes.into_iter()
    //                 .max_by_key(|(_, v)| v.len())
    //                 .map(|(key, value)| (key, value)) // HashMapのエントリからタプルに変換
    //                 .expect("No results to vote on for MajorityVote strategy.");
                
    //             let winning_logical_error_vec: Vec<Bit> = winning_coset_logical_error.to_vec();
    //             println!("Winning Coset: {:?} (with {} votes)", winning_logical_error_vec, winning_coset_results.len());


    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ4: 勝利コセット内で最小重みを選択
    //             // -----------------------------------------------------------------
    //             // 保存しておいた重み `weight` で比較
    //             let (final_result, final_weight) = winning_coset_results.into_iter()
    //                 // min_by は Option<(DecodeResult, f64)> を返す
    //                 .min_by(|(_, llr_a), (_, llr_b)| { 
    //                     // f64 (LLRコスト) 同士を直接比較
    //                     llr_a.partial_cmp(llr_b).unwrap_or(std::cmp::Ordering::Equal)
    //                 })
    //                 // Option をアンラップして (DecodeResult, f64) を取り出す
    //                 .expect("Winning coset was empty.");

    //             // final_weight が LLR コスト
    //             println!("Final Selection: LLR Cost = {:.6}, From Coset = {:?}", final_weight, winning_logical_error_vec);
    //             println!("----------------------------------------------------------------\n");

    //             final_result
    //         }
    //     }
    // }


// Implement the DecoderRunner trait for EnsembleDecoder
impl DecoderRunner for EnsembleDecoder {}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::bp::min_sum::MinSumDecoderConfig;
    use crate::bp::relay::{RelayDecoder, RelayDecoderConfig};
    use crate::dem::DetectorErrorModel;
    use crate::observable_decoder::ObservableDecoderRunner;
    use crate::utilities::test::get_test_data_path;
    use ndarray::{Array2, s};
    use ndarray_npy::read_npy;
    use std::sync::Arc;

    #[test]
    fn test_ensemble_relay_decode_basic() {
        // 1. テストデータの準備
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open");

        let ensemble_size = 3;
        let mut child_decoders : Vec<Box<dyn Decoder + Send>> = Vec::new();

        for _i in 0..ensemble_size{
            let bp_config_144_12_12 = Arc::new(MinSumDecoderConfig{
                error_priors: code_144_12_12.error_priors.clone(),
                max_iter: 20,
                alpha: None,
                alpha_iteration_scaling_factor: 0.,
                gamma0: Some(0.9),
                ..Default::default()
            });

            let relay_config = Arc::new(RelayDecoderConfig{
                pre_iter: 30,
                num_sets: 1,
                set_max_iter: 20,
                gamma_dist_interval: (-0.25, 0.60),
                ..Default::default()
            });
            
            let decoder = RelayDecoder::<f64>::new(
                Arc::new(code_144_12_12.detector_error_matrix.clone()),
                bp_config_144_12_12.clone(),
                relay_config.clone(),
            );

            child_decoders.push(Box::new(decoder));
        }

        let original_log_priors = Arc::new(code_144_12_12.error_priors.mapv(|p| (p/(1.0-p)).ln()));

        let mut ensemble_decoder = EnsembleDecoder::new(
            child_decoders,
            SelectionStrategy::MostLikely,
            Some(original_log_priors),
            None,
            None,
        );

        let _decode_result = ensemble_decoder.decode_detailed(
            detectors_144_12_12.row(0)
        );

        println!("Ensemble Relay Decoder Test Passed!");
    }

    #[test]
    fn test_ensemble_observable_decode() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open");

        let ensemble_size = 8;
        let mut child_decoders : Vec<Box<dyn Decoder + Send>> = Vec::new();

        for i in 0..ensemble_size{
            let bp_config_144_12_12 = Arc::new(MinSumDecoderConfig{
                error_priors: code_144_12_12.error_priors.clone(),
                max_iter: 20,
                alpha: None,
                alpha_iteration_scaling_factor: 0.,
                gamma0: Some(0.9),
                ..Default::default()
            });

            let relay_config = Arc::new(RelayDecoderConfig{
                pre_iter: 10,
                num_sets: 3,
                set_max_iter: 20,
                gamma_dist_interval: (0.0, 0.0),
                // gamma_dist_interval: (-2.25, -1.1),
                seed: i,
                ..Default::default()
            });

            
            let decoder = RelayDecoder::<f64>::new(
                Arc::new(code_144_12_12.detector_error_matrix.clone()),
                bp_config_144_12_12.clone(),
                relay_config.clone(),
            );

            child_decoders.push(Box::new(decoder));
        }

    
        let original_log_priors = Arc::new(code_144_12_12.error_priors.mapv(|p| ((1.0 - p) / p).ln()));
        let observable_matrix = Arc::new(code_144_12_12.observable_error_matrix.clone());

        let ensemble_decoder = EnsembleDecoder::new(
            child_decoders,
            SelectionStrategy::MostLikely,
            Some(original_log_priors),
            Some(observable_matrix.clone()), // MajorityVoteにはobservable_matrixが必要
            None,
        );
        // --- 2. ObservableDecoderRunnerでラップする ---
        // これが今回の核心部分です！
        // 作成した ensemble_decoder を、ObservableDecoderRunnerに渡します。
        // これで、アンサンブルデコードの結果から論理エラーを計算する準備が整いました。
        let mut observable_ensemble_decoder = ObservableDecoderRunner::new(
            Box::new(ensemble_decoder),
            observable_matrix,
            false,
        );

        // --- 3. 論理エラーを含めてデコードを実行 ---
        // ObservableDecoderRunnerが提供するメソッドを使ってデコードします。
        // これにより、内部でensemble_decoder.decode()が呼ばれ、
        // その結果を使って論理エラーが計算されます。
        let _logical_errors = observable_ensemble_decoder.decode_observables_batch(detectors_144_12_12.slice(s![0..30, ..]));

        println!("Ensemble Observable Decoding Test Passed!");
        println!("error prior: {:?}", code_144_12_12.error_priors);
        // 実際の論理エラーが正しいかは、既知のエラーとシンドロームのペアで検証する必要があります。
        // ここでは、実行が完了することを確認します。
        }
}