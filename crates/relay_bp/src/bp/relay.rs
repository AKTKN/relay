// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use crate::decoder::{Bit, SparseBitMatrix};
use crate::decoder::{DecodeResult, Decoder, DecoderRunner};
use log::debug;

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};
use std::fmt::Debug;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
//use std::string;
use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;
use std::process::exit;
use std::sync::Arc;

#[derive(Clone, PartialEq, Debug)]
pub enum StoppingCriterion {
    PreIter,
    NConv { stop_after: usize },
    All,
}

impl Default for StoppingCriterion {
    fn default() -> StoppingCriterion {
        StoppingCriterion::NConv { stop_after: 1 }
    }
}

#[derive(Clone, Debug)]
pub struct RelayDecoderConfig {
    pub pre_iter: usize,
    pub num_sets: usize,
    pub set_max_iter: usize,
    pub gamma_dist_interval: (f64, f64),
    pub explicit_gammas: Option<Array2<f64>>,
    pub stopping_criterion: StoppingCriterion,
    pub logging: bool,
    pub seed: u64,
    // Repulsive mode parameters
    pub repulsive_gamma_dist: Option<(f64, f64)>,
    pub abs_llr_threshold: Option<f64>,
    pub pulse_per_leg: Option<usize>,
    pub start_leg: Option<usize>,
}

impl Default for RelayDecoderConfig {
    fn default() -> Self {
        Self {
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            gamma_dist_interval: (-0.24, 0.66),
            explicit_gammas: None,
            stopping_criterion: StoppingCriterion::default(),
            logging: false,
            seed: 0,
            repulsive_gamma_dist: None,
            abs_llr_threshold: None,
            pulse_per_leg: None,
            start_leg: None,
        }
    }
}

#[derive(Clone)]
struct PosteriorUpdateState {
    rng_std: rand::rngs::StdRng,
    uniform: rand::distributions::Uniform<f64>,
    repulsive_uniform: Option<rand::distributions::Uniform<f64>>,
}

/// An ensemble decoder which controls an inner BP min-sum decoder.
#[derive(Clone)]
pub struct RelayDecoder<N: PartialEq + Default + Clone + Copy> {
    bp_decoder: MinSumBPDecoder<N>,
    relay_config: Arc<RelayDecoderConfig>,
    posterior_update_state: PosteriorUpdateState,
    sets_quality: Array1<f64>,
    sets_iter: Array1<usize>,
    sets_conv: Array1<bool>,
    sets_best: Array1<bool>,
    num_executed_sets: usize,
}

impl<N> RelayDecoder<N>
where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + Signed
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        relay_config: Arc<RelayDecoderConfig>,
    ) -> RelayDecoder<N> {
        if relay_config.logging {
            let log_line = format!(
                "# pre_iter: {}: sets: {} set_max_iter: {}\n\
                # gamma_distribution: {:?} # set_idx, num_iter, converged, unique_best_solution\n",
                relay_config.pre_iter,
                relay_config.num_sets,
                relay_config.set_max_iter,
                relay_config.gamma_dist_interval,
            );
            let mut file =
                File::create("relay_logging.out").expect("Unable to create file for logging.");
            file.write_all(log_line.as_bytes())
                .expect("Unable to write Relay logging data.");
        }

        // Create logging variables if applicable
        let (sets_quality, sets_iter, sets_conv, sets_best);
        if relay_config.logging {
            sets_quality = Array1::<f64>::from_elem(relay_config.num_sets + 1, f64::MAX);
            sets_iter =
                Array1::<usize>::from_elem(relay_config.num_sets + 1, relay_config.set_max_iter);
            sets_conv = Array1::<bool>::from_elem(relay_config.num_sets + 1, false);
            sets_best = Array1::<bool>::from_elem(relay_config.num_sets + 1, false);
        } else {
            sets_quality = Array1::<f64>::zeros(1);
            sets_iter = Array1::<usize>::zeros(1);
            sets_conv = Array1::<bool>::from_elem(1, false);
            sets_best = Array1::<bool>::from_elem(1, false);
        }

        if relay_config.explicit_gammas.is_some() {
            let gammas_shape = relay_config.explicit_gammas.as_ref().unwrap().shape();
            let num_variable_nodes = check_matrix.cols();
            if num_variable_nodes != gammas_shape[1] {
                println!("ERROR: Number of specified gammas {} does not match the number of variable nodes {}.", gammas_shape[1], num_variable_nodes);
                exit(-1);
            };
            if relay_config.num_sets > gammas_shape[0] {
                println!("WARNING: Number of different gamma sets {} is smaller than the number of Relay legs {}. Legs will be reused.", gammas_shape[0], relay_config.num_sets)
            }
        }

        // The actual number of sets Relay ran, depends on the stopping criterion
        let num_executed_sets = 0;

        let bp_decoder = MinSumBPDecoder::new(check_matrix, min_sum_config);

        let posterior_update_state = Self::init_dismem_state(&relay_config);

        RelayDecoder {
            bp_decoder,
            relay_config,
            posterior_update_state,
            sets_quality,
            sets_iter,
            sets_conv,
            sets_best,
            num_executed_sets,
        }
    }

    fn init_dismem_state(relay_config: &RelayDecoderConfig) -> PosteriorUpdateState {
        let rng_std: rand::prelude::StdRng = rand::rngs::StdRng::seed_from_u64(relay_config.seed);
        let low = relay_config.gamma_dist_interval.0;
        let high = relay_config.gamma_dist_interval.1;
        
        // Only create Uniform distribution if low != high
        let uniform: rand::distributions::Uniform<f64> = if (low - high).abs() < 1e-12 {
            // For fixed gamma, create a dummy uniform distribution (won't be used)
            Uniform::new(0.0, 1.0)
        } else {
            Uniform::new(low, high)
        };
        
        let repulsive_uniform = if let Some((rep_low, rep_high)) = relay_config.repulsive_gamma_dist {
            if (rep_low - rep_high).abs() < 1e-12{
                None
            } else{
                Some(Uniform::new(rep_low, rep_high))
            }
        } else {
            None
        };
        
        PosteriorUpdateState { rng_std, uniform, repulsive_uniform }
    }

    fn init_next_set(&mut self, set_idx: usize) {
        let mut gammas = Array1::zeros(self.check_matrix().cols());
        if self.relay_config.explicit_gammas.is_some() {
            let gammas_num_sets = self.relay_config.explicit_gammas.as_ref().unwrap().shape()[0];
            for i in 0..gammas.len() {
                gammas[i] = *self
                    .relay_config
                    .explicit_gammas
                    .as_ref()
                    .unwrap()
                    .get((set_idx % gammas_num_sets, i))
                    .unwrap();
            }
            self.bp_decoder.set_memory_strengths_f64(gammas);
            return;
        }
        
        // Check if gamma_dist_interval is a fixed value (low == high)
        let (low, high) = self.relay_config.gamma_dist_interval;
        let is_fixed = (low - high).abs() < 1e-12;
        
        for i in 0..gammas.len() {
            gammas[i] = if is_fixed {
                low  // Use fixed value
            } else {
                self.posterior_update_state
                    .uniform
                    .sample(&mut self.posterior_update_state.rng_std)
            };
            // println!("gamma for var {}: {}", i, gammas[i]);
        }
        self.bp_decoder.set_memory_strengths_f64(gammas);
    }

    /// Initialize next set with repulsive mode: assigns repulsive gammas to high-LLR positions
    fn init_repulsive_set(&mut self, set_idx: usize) {
        let mut gammas = Array1::zeros(self.check_matrix().cols());
        
        if self.relay_config.explicit_gammas.is_some() {
            // If explicit gammas are specified, use them as in normal mode
            let gammas_num_sets = self.relay_config.explicit_gammas.as_ref().unwrap().shape()[0];
            for i in 0..gammas.len() {
                gammas[i] = *self
                    .relay_config
                    .explicit_gammas
                    .as_ref()
                    .unwrap()
                    .get((set_idx % gammas_num_sets, i))
                    .unwrap();
            }
            self.bp_decoder.set_memory_strengths_f64(gammas);
            return;
        }

        // Get current posterior ratios (LLR values) from BP decoder
        let posterior_ratios = self.bp_decoder.get_posterior_ratios_f64();
        let abs_llr_threshold = self.relay_config.abs_llr_threshold.unwrap_or(2.0);

        let repulsive_fixed_value = self.relay_config.repulsive_gamma_dist.map(|(low, _)| low);
        
        // Check if normal gamma_dist_interval is a fixed value
        let (normal_low, normal_high) = self.relay_config.gamma_dist_interval;
        let normal_is_fixed = (normal_low - normal_high).abs() < 1e-12;
        
        // Determine which distribution to use for each variable
        // let has_repulsive_dist = self.posterior_update_state.repulsive_uniform.is_some();
        
        for i in 0..gammas.len() {
            let abs_llr = posterior_ratios[i].abs();
            
            // Apply repulsive gamma to variables with abs(LLR) <= threshold
            if abs_llr <= abs_llr_threshold {
                // repulsive_uniform がある場合はサンプリング、ない場合は固定値
                gammas[i] = if let Some(ref dist) = self.posterior_update_state.repulsive_uniform {
                    dist.sample(&mut self.posterior_update_state.rng_std)
                } else if let Some(fixed_val) = repulsive_fixed_value {
                    fixed_val  // ← 固定値を使う
                } else {
                    // repulsive_gamma_dist 自体が None の場合は通常の gamma を使う
                    if normal_is_fixed {
                        normal_low
                    } else {
                        self.posterior_update_state
                            .uniform
                            .sample(&mut self.posterior_update_state.rng_std)
                    }
                };
            } else {
                gammas[i] = if normal_is_fixed {
                    normal_low
                } else {
                    self.posterior_update_state
                        .uniform
                        .sample(&mut self.posterior_update_state.rng_std)
                };
            }
        }
        
        self.bp_decoder.set_memory_strengths_f64(gammas);
    }

    /// Check if repulsive mode should be applied for the given set index
    fn should_apply_repulsive(&self, set_idx: usize) -> bool {
        // Check if repulsive parameters are configured
        if self.relay_config.repulsive_gamma_dist.is_none() {
            return false;
        }
        
        let start_leg = self.relay_config.start_leg.unwrap_or(0);
        let pulse_per_leg = self.relay_config.pulse_per_leg.unwrap_or(1);
        
        // set_idx starts from 1 (after pre_iter), so we need to adjust
        // set_idx = 1 means first leg after pre_iter
        if set_idx < start_leg + 1 {
            return false;
        }
        
        // Apply repulsive every pulse_per_leg legs
        let leg_offset = set_idx - 1; // Convert to 0-based leg index
        (leg_offset - start_leg) % pulse_per_leg == 0
    }

    /// Decode with the inner decoder
    /// If is_repulsive_leg is true, apply repulsive gamma for the entire leg
    fn decode_inner(&mut self, detectors: ArrayView1<Bit>, max_iter: usize, is_repulsive_leg: bool) -> DecodeResult {

        // if detectors.iter().all(|&d| d == 0){
        //     debug!("No detection events. Returning success immediately.");
        //     return self.bp_decoder.build_result(true, Array1::zeros(detectors.dim()), max_iter)

        // }
        let mut success: bool = false;
        let mut decoded_detectors = Array1::default(detectors.dim());

        for iter in 0..max_iter {
            self.bp_decoder.run_iteration(detectors);
            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors, decoded_detectors.view());

            // If we have converged may now exit
            if success {
                debug!(
                    "Succeeded on iteration {:?}",
                    self.bp_decoder.current_iteration
                );
                break;
            }
            self.bp_decoder.current_iteration += 1;
        }

        self.bp_decoder
            .build_result(success, decoded_detectors, max_iter, Some(detectors))
    }

    fn write_log(&mut self, file: File) {
        let mut buf_writer = BufWriter::new(file);
        for set in 0..=self.num_executed_sets {
            let log_line = format!(
                "{}, {}, {}, {}\n",
                (set - 1) as i32,
                self.sets_iter[set],
                self.sets_conv[set] as u8,
                self.sets_best[set] as u8
            );
            buf_writer
                .write_all(log_line.as_bytes())
                .expect("Unable to write Relay logging data.");
        }
        buf_writer
            .flush()
            .expect("Unable to write Relay logging data.");
    }
}

impl<N> Decoder for RelayDecoder<N>
where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + Signed
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static,
{
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.bp_decoder.check_matrix()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.bp_decoder.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        // Initialization
        let mut num_conv = 0;
        let mut min_pm = f64::MAX;
        let mut num_sets_best = 0;
        let mut best_set_idx = 0;
        let mut total_iterations: usize = 0;
        self.num_executed_sets = 0;
        let stopping_criterion = self.relay_config.stopping_criterion.clone();

        // First Mem-BP
        self.bp_decoder.initialize_decoder();
        let mut result = self.decode_inner(detectors, self.relay_config.pre_iter, false);
        self.num_executed_sets = 1;

        // Create logging variables and log first set if applicable
        if self.relay_config.logging {
            self.sets_iter[0] = result.iterations;
            self.sets_conv[0] = result.success;
        }

        // Check early stopping criteria
        if result.success {
            num_conv += 1;
            min_pm = result.decoding_quality;
            num_sets_best += 1;
            if self.relay_config.logging {
                self.sets_quality[0] = result.decoding_quality
            };

            let mut done = false;
            if stopping_criterion == StoppingCriterion::PreIter {
                done = true;
            } else if let StoppingCriterion::NConv { stop_after } = stopping_criterion {
                if num_conv >= stop_after {
                    done = true;
                }
            }
            // If stopping criterion has been met: Log (if applicable) and return
            if done {
                if self.relay_config.logging {
                    self.sets_best[0] = true;
                    let file = OpenOptions::new()
                        .append(true)
                        .open("relay_logging.out")
                        .unwrap();
                    self.write_log(file);
                }
                return result;
            }
        }

        // Init and loop over all Relay sets
        total_iterations += result.iterations;
        for set in 1..=self.relay_config.num_sets {
            // Do not completely initialize decoder as we wish to relay
            // posterior marginals with new memory strengths.
            
            // Determine whether to apply repulsive mode for this set
            let is_repulsive = self.should_apply_repulsive(set);
            if is_repulsive {
                self.init_repulsive_set(set);
            } else {
                self.init_next_set(set);
            }
            
            self.bp_decoder.current_iteration = 0;
            self.bp_decoder.initialize_check_to_variable();
            self.bp_decoder.initialize_variable_to_check();
            let temp_result = self.decode_inner(detectors, self.relay_config.set_max_iter, is_repulsive);

            self.num_executed_sets += 1;
            total_iterations += temp_result.iterations;
            if temp_result.success {
                num_conv += 1;
                let pm = temp_result.decoding_quality;
                if self.relay_config.logging {
                    self.sets_conv[set] = true;
                    self.sets_iter[set] = temp_result.iterations;
                    self.sets_quality[set] = pm;
                }
                if pm == min_pm {
                    // Count how often we found the best solution
                    num_sets_best += 1;
                }
                if pm < min_pm {
                    // Found a new best solution
                    num_sets_best = 1;
                    best_set_idx = set;
                    min_pm = pm;
                    result = temp_result;
                }
                if let StoppingCriterion::NConv { stop_after } = stopping_criterion {
                    if num_conv >= stop_after {
                        break;
                    }
                }
            }
        }
        result.iterations = total_iterations;

        // Rest of the function is just logging
        if self.relay_config.logging {
            if num_sets_best == 1 {
                self.sets_best[best_set_idx] = true;
            }
            let file = OpenOptions::new()
                .append(true)
                .open("relay_logging.out")
                .unwrap();
            self.write_log(file);
        }

        result
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.bp_decoder.get_decoding_quality(errors)
    }
}

impl<N> DecoderRunner for RelayDecoder<N> where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + Signed
        + Bounded
        + FromPrimitive
        + ToPrimitive
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static
{
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::bipartite_graph::{BipartiteGraph, SparseBipartiteGraph};
    use env_logger;
    use ndarray::prelude::*;

    use crate::dem::DetectorErrorModel;
    use crate::utilities::test::get_test_data_path;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    // Basic test where Relay is called but only runs 1 BP iteration
    #[test]
    fn min_sum_decode_repetition_code() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];

        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let check_matrix_arc = Arc::new(check_matrix);

        let iterations = 10;
        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            alpha: Some(1.),
            alpha_iteration_scaling_factor: 1.,
            gamma0: None,
            ..Default::default()
        };
        let bp_config_arc = Arc::new(bp_config);

        let relay_config = RelayDecoderConfig {
            pre_iter: iterations,
            num_sets: 0,
            set_max_iter: 150,
            stopping_criterion: StoppingCriterion::PreIter,
            explicit_gammas: None,
            ..Default::default()
        };
        let relay_config_arc = Arc::new(relay_config);

        let mut decoder: RelayDecoder<f32> =
            RelayDecoder::new(check_matrix_arc, bp_config_arc, relay_config_arc);

        let error = array![0, 0, 0];
        let detectors: Array1<Bit> = array![0, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![1, 0, 0];
        let detectors: Array1<Bit> = array![1, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 1, 0];
        let detectors: Array1<Bit> = array![1, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 0, 1];
        let detectors: Array1<Bit> = array![0, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);
    }

    // Basic test where Relay runs 40 sets
    #[test]
    fn decode_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 0.,
            gamma0: Some(0.9),
            ..Default::default()
        };
        let relay_config = RelayDecoderConfig::default();
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config = Arc::new(bp_config_144_12_12);
        let config = Arc::new(relay_config);
        let mut decoder_144_12_12: RelayDecoder<f64> =
            RelayDecoder::new(check_matrix, bp_config, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        // All should pass for Relay.
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>()
                == (detectors_slice.shape()[0])
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    // Basic test where Relay runs 40 sets
    #[test]
    fn decode_144_12_12_int() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");

        let bits = 16;
        let scale = 8.0;

        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 0.,
            gamma0: Some(0.9),
            max_data_value: Some(((1 << bits) - 1) as f64),
            data_scale_value: Some(scale),
            ..Default::default()
        };
        let relay_config = RelayDecoderConfig {
            ..Default::default()
        };
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config = Arc::new(bp_config_144_12_12);
        let config = Arc::new(relay_config);
        let mut decoder_144_12_12: RelayDecoder<isize> =
            RelayDecoder::new(check_matrix, bp_config, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        // All should pass for Relay.
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>()
                == (detectors_slice.shape()[0])
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }
}
