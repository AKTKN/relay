use super::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use super::relay::StoppingCriterion;
use crate::decoder::{BPExtraResult, Bit, SparseBitMatrix};
use crate::decoder::{DecodeResult, Decoder, DecoderRunner};
use log::debug;

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct LRBPDecoderConfig {
    pub pre_iter: usize,
    pub num_sets: usize,
    pub set_max_iter: usize,
    pub gamma_dist_interval: (f64, f64),
    pub explicit_gammas: Option<Array2<f64>>,
    pub stopping_criterion: StoppingCriterion,
    pub logging: bool,
    pub seed: u64,
    pub osc_window: usize,
    pub friction_slope: f64,
    pub friction_shift: f64,
    pub tau: f64,
}

impl Default for LRBPDecoderConfig {
    fn default() -> Self {
        Self {
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            gamma_dist_interval: (-0.24, 0.66),
            explicit_gammas: None,
            stopping_criterion: StoppingCriterion::NConv { stop_after: 1 },
            logging: false,
            seed: 0,
            osc_window: 5,
            friction_slope: 2.0,
            friction_shift: 0.0,
            tau: 0.2,
        }
    }
}

#[derive(Clone)]
struct LRBPState {
    rng_std: rand::rngs::StdRng,
    uniform: rand::distributions::Uniform<f64>,
}

#[derive(Clone)]
pub struct LRBPDecoder<N: PartialEq + Default + Clone + Copy> {
    bp_decoder: MinSumBPDecoder<N>,
    lrbp_config: Arc<LRBPDecoderConfig>,
    state: LRBPState,
}

impl<N> LRBPDecoder<N>
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
        lrbp_config: Arc<LRBPDecoderConfig>,
    ) -> LRBPDecoder<N> {
        let bp_decoder = MinSumBPDecoder::new(check_matrix, min_sum_config);
        let state = Self::init_state(&lrbp_config);

        LRBPDecoder {
            bp_decoder,
            lrbp_config,
            state,
        }
    }

    fn init_state(lrbp_config: &LRBPDecoderConfig) -> LRBPState {
        let rng_std: rand::prelude::StdRng = rand::rngs::StdRng::seed_from_u64(lrbp_config.seed);
        let low = lrbp_config.gamma_dist_interval.0;
        let high = lrbp_config.gamma_dist_interval.1;
        let uniform: rand::distributions::Uniform<f64> = Uniform::new(low, high);
        LRBPState { rng_std, uniform }
    }

    fn init_next_set(&mut self, set_idx: usize) {
        let mut gammas = Array1::zeros(self.check_matrix().cols());
        if self.lrbp_config.explicit_gammas.is_some() {
            let gammas_num_sets = self.lrbp_config.explicit_gammas.as_ref().unwrap().shape()[0];
            for i in 0..gammas.len() {
                gammas[i] = *self
                    .lrbp_config
                    .explicit_gammas
                    .as_ref()
                    .unwrap()
                    .get((set_idx % gammas_num_sets, i))
                    .unwrap();
            }
            self.bp_decoder.set_memory_strengths_f64(gammas);
            return;
        }
        for i in 0..gammas.len() {
            gammas[i] = self.state.uniform.sample(&mut self.state.rng_std);
        }
        self.bp_decoder.set_memory_strengths_f64(gammas);
    }

    fn clear_history(&self, history: &mut Vec<VecDeque<f64>>) {
        for edge_hist in history.iter_mut() {
            edge_hist.clear();
        }
    }

    fn parity_violations(&self, detectors: ArrayView1<Bit>) -> Vec<u8> {
        let check_matrix = self.bp_decoder.check_matrix().to_csr();
        let mut violations = vec![0_u8; check_matrix.rows()];
        let decoding = self.bp_decoder.current_decoding();

        for (row_idx, row_vec) in check_matrix.outer_iterator().enumerate() {
            let mut parity = 0_u8;
            for (col_idx, val) in row_vec.iter() {
                if *val == 1 && decoding[col_idx] == 1 {
                    parity ^= 1;
                }
            }
            violations[row_idx] = parity ^ (detectors[row_idx] as u8);
        }
        

        violations
    }

    fn apply_langevin_modifier(
        &mut self,
        history: &mut [VecDeque<f64>],
        parity_violations: &[u8],
        current_iter: usize,
    ) {
        let indices = self.bp_decoder.check_to_variable_indices().to_vec();
        let raw_messages: Vec<f64> = self
            .bp_decoder
            .check_to_variable_data()
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect();

        let mut modified_messages = raw_messages.clone();
        let window_size = self.lrbp_config.osc_window.max(1);
        let beta = self.lrbp_config.friction_slope;
        let delta = self.lrbp_config.friction_shift;
        let tau = self.lrbp_config.tau.max(1e-12);

        for edge_idx in 0..raw_messages.len() {
            let raw = raw_messages[edge_idx];
            let check_idx = indices[edge_idx];

            let edge_history = &mut history[edge_idx];
            edge_history.push_back(raw);
            while edge_history.len() > window_size {
                edge_history.pop_front();
            }

            let oscillation = edge_history.iter().sum::<f64>().abs();
            let friction = 1.0 / (1.0 + (-beta * (oscillation - delta)).exp());

            let t_eff = tau/((current_iter+1) as f64).max(1.0); 
            
            let mut p_kick = (-oscillation / t_eff).exp();
        
            if parity_violations[check_idx] == 0 {
                p_kick = 0.0;
            }
            p_kick = p_kick.clamp(0.0, 1.0);

            let do_kick = self.state.rng_std.gen_bool(p_kick);
            let damped = friction.clamp(0.0, 1.0) * raw;
            modified_messages[edge_idx] = if do_kick { -damped } else { damped };
        }

        for (target, message) in self
            .bp_decoder
            .check_to_variable_data_mut()
            .iter_mut()
            .zip(modified_messages.iter())
        {
            let mapped = N::from_f64(*message).unwrap_or_else(N::zero);
            *target = mapped;
        }
    }

    fn apply_langevin_modifier_with_dropout(
        &mut self,
        history: &mut [VecDeque<f64>],
        parity_violations: &[u8],
        current_iter: usize,
        previous_messages: &[N],
    ) {
        let indices = self.bp_decoder.check_to_variable_indices().to_vec();
        let raw_messages: Vec<f64> = self
            .bp_decoder
            .check_to_variable_data()
            .iter()
            .map(|x| x.to_f64().unwrap_or(0.0))
            .collect();

        let num_checks = self.bp_decoder.check_matrix().rows();
        let mut check_dropout_decisions: Vec<Option<bool>> = vec![None; num_checks];
        let mut modified_messages: Vec<N> = vec![N::zero(); raw_messages.len()];

        let window_size = self.lrbp_config.osc_window.max(1);
        let beta = self.lrbp_config.friction_slope;
        let delta = self.lrbp_config.friction_shift;
        let tau = self.lrbp_config.tau.max(1e-12);

        for edge_idx in 0..raw_messages.len() {
            let raw = raw_messages[edge_idx];
            let check_idx = indices[edge_idx];

            let edge_history = &mut history[edge_idx];
            edge_history.push_back(raw);
            while edge_history.len() > window_size {
                edge_history.pop_front();
            }

            let oscillation = edge_history.iter().sum::<f64>().abs();
            let friction = 1.0 / (1.0 + (-beta * (oscillation - delta)).exp());

            let t_eff = tau / ((current_iter + 1) as f64).max(1.0);
            let mut p_kick = (-oscillation / t_eff).exp();
            if parity_violations[check_idx] == 0 {
                p_kick = 0.0;
            }
            p_kick = p_kick.clamp(0.0, 1.0);

            // One Bernoulli decision per check node: if true, reuse t-1 messages.
            let drop_this_check = match check_dropout_decisions[check_idx] {
                Some(decision) => decision,
                None => {
                    let decision = self.state.rng_std.gen_bool(p_kick);
                    check_dropout_decisions[check_idx] = Some(decision);
                    decision
                }
            };

            if drop_this_check {
                modified_messages[edge_idx] = previous_messages[edge_idx];
                continue;
            }

            let do_kick = self.state.rng_std.gen_bool(p_kick);
            let damped = friction.clamp(0.0, 1.0) * raw;
            let message = if do_kick { -damped } else { damped };
            modified_messages[edge_idx] = N::from_f64(message).unwrap_or_else(N::zero);
        }

        for (target, message) in self
            .bp_decoder
            .check_to_variable_data_mut()
            .iter_mut()
            .zip(modified_messages.iter())
        {
            *target = *message;
        }
    }

    // Baseline Langevin leg update without dropout.
    fn run_langevin_iteration(&mut self, detectors: ArrayView1<Bit>, history: &mut [VecDeque<f64>]) {
        let parity_violations = self.parity_violations(detectors);
        self.bp_decoder.run_check_to_variable_update(detectors);
        self.apply_langevin_modifier(
            history,
            &parity_violations,
            self.bp_decoder.current_iteration,
        );
        self.bp_decoder.run_variable_to_check_update();
        // self.bp_decoder.run_lrbp_variable_to_check_update();
    }

    // Experimental Langevin leg update with check-node dropout.
    fn run_langevin_iteration_with_dropout(
        &mut self,
        detectors: ArrayView1<Bit>,
        history: &mut [VecDeque<f64>],
    ) {
        let parity_violations = self.parity_violations(detectors);
        let previous_messages: Vec<N> = self.bp_decoder.check_to_variable_data().to_vec();
        self.bp_decoder.run_check_to_variable_update(detectors);
        self.apply_langevin_modifier_with_dropout(
            history,
            &parity_violations,
            self.bp_decoder.current_iteration,
            previous_messages.as_slice(),
        );
        self.bp_decoder.run_variable_to_check_update();
        // self.bp_decoder.run_lrbp_variable_to_check_update();
    }

    fn decode_inner(
        &mut self,
        detectors: ArrayView1<Bit>,
        max_iter: usize,
        use_langevin: bool,
        history: &mut [VecDeque<f64>],
    ) -> DecodeResult {
        let mut success: bool = false;
        let mut decoded_detectors = Array1::default(detectors.dim());

        for _ in 0..max_iter {
            if use_langevin {
                // let parity_violations = self.parity_violations(detectors);
                // self.bp_decoder.run_check_to_variable_update(detectors);
                // self.apply_langevin_modifier(history, &parity_violations, self.bp_decoder.current_iteration);
                // // Use LRBP-specific marginal blending after computing the plain BP marginal.
                // self.bp_decoder.run_variable_to_check_update();
                // self.bp_decoder.run_lrbp_variable_to_check_update();
                // Toggle this flag to quickly switch between baseline and dropout updates.
                let use_check_update_dropout = true;
                if use_check_update_dropout {
                    self.run_langevin_iteration_with_dropout(detectors, history);
                } else {
                    self.run_langevin_iteration(detectors, history);
                }
            } else {
                self.bp_decoder.run_iteration(detectors);
            }

            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors, decoded_detectors.view());

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
            .build_result(success, decoded_detectors, max_iter)
    }
}

impl<N> Decoder for LRBPDecoder<N>
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
        let mut num_conv = 0;
        let mut min_pm = f64::MAX;
        let mut total_iterations: usize = 0;
        let stopping_criterion = self.lrbp_config.stopping_criterion.clone();
        let mut leg_success: Vec<bool> = Vec::with_capacity(self.lrbp_config.num_sets + 1);
        let mut leg_iterations: Vec<usize> = Vec::with_capacity(self.lrbp_config.num_sets + 1);
        let mut leg_negative_llr_counts: Vec<usize> =
            Vec::with_capacity(self.lrbp_config.num_sets + 1);
        let mut leg_decodings: Vec<Array1<Bit>> = Vec::with_capacity(self.lrbp_config.num_sets + 1);
        let mut leg_posteriors: Vec<Array1<f64>> =
            Vec::with_capacity(self.lrbp_config.num_sets + 1);

        self.bp_decoder.initialize_decoder();

        let num_edges = self.bp_decoder.check_to_variable_data().len();
        let mut history: Vec<VecDeque<f64>> = (0..num_edges)
            .map(|_| VecDeque::with_capacity(self.lrbp_config.osc_window.max(1)))
            .collect();

        let mut result = self.decode_inner(
            detectors,
            self.lrbp_config.pre_iter,
            false,
            history.as_mut_slice(),
        );

        leg_success.push(result.success);
        leg_iterations.push(result.iterations);
        leg_negative_llr_counts.push(result.posterior_ratios.iter().filter(|x| **x < 0.0).count());
        leg_decodings.push(result.decoding.clone());
        leg_posteriors.push(result.posterior_ratios.clone());

        if result.success {
            num_conv += 1;
            min_pm = result.decoding_quality;

            let mut done = false;
            if stopping_criterion == StoppingCriterion::PreIter {
                done = true;
            } else if let StoppingCriterion::NConv { stop_after } = stopping_criterion {
                if num_conv >= stop_after {
                    done = true;
                }
            }
            if done {
                result.extra = BPExtraResult::RelayTrace {
                    leg_success,
                    leg_iterations,
                    leg_negative_llr_counts,
                    leg_decodings,
                    leg_posteriors,
                };
                return result;
            }
        }

        total_iterations += result.iterations;
        for set in 1..=self.lrbp_config.num_sets {
            self.init_next_set(set);
            self.bp_decoder.current_iteration = 0;
            self.clear_history(&mut history);

            let temp_result = self.decode_inner(
                detectors,
                self.lrbp_config.set_max_iter,
                true,
                history.as_mut_slice(),
            );

            leg_success.push(temp_result.success);
            leg_iterations.push(temp_result.iterations);
            leg_negative_llr_counts.push(
                temp_result
                    .posterior_ratios
                    .iter()
                    .filter(|x| **x < 0.0)
                    .count(),
            );
            leg_decodings.push(temp_result.decoding.clone());
            leg_posteriors.push(temp_result.posterior_ratios.clone());

            total_iterations += temp_result.iterations;
            if temp_result.success {
                num_conv += 1;
                let pm = temp_result.decoding_quality;
                if pm < min_pm {
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
        result.extra = BPExtraResult::RelayTrace {
            leg_success,
            leg_iterations,
            leg_negative_llr_counts,
            leg_decodings,
            leg_posteriors,
        };

        result
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.bp_decoder.get_decoding_quality(errors)
    }
}

impl<N> DecoderRunner for LRBPDecoder<N> where
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
    use ndarray::array;

    #[test]
    fn lrbp_decode_repetition_code() {
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];
        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let check_matrix_arc = Arc::new(check_matrix);

        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: 120,
            alpha: Some(1.),
            alpha_iteration_scaling_factor: 1.,
            gamma0: Some(0.1),
            ..Default::default()
        };
        let bp_config_arc = Arc::new(bp_config);

        let lrbp_config = LRBPDecoderConfig {
            pre_iter: 40,
            num_sets: 20,
            set_max_iter: 20,
            gamma_dist_interval: (-0.24, 0.66),
            stopping_criterion: StoppingCriterion::NConv { stop_after: 1 },
            osc_window: 4,
            friction_slope: 2.0,
            friction_shift: 0.0,
            tau: 0.2,
            seed: 7,
            ..Default::default()
        };
        let lrbp_config_arc = Arc::new(lrbp_config);

        let mut decoder: LRBPDecoder<f32> =
            LRBPDecoder::new(check_matrix_arc, bp_config_arc, lrbp_config_arc);

        let detectors = array![1, 1];
        let result = decoder.decode_detailed(detectors.view());

        assert!(result.success);
        assert_eq!(result.decoding, array![0, 1, 0]);
        assert_eq!(result.decoded_detectors, detectors);
        assert!(result.iterations <= 400);
    }
}
