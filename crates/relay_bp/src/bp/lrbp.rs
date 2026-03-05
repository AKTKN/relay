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
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub enum DynMode {
    EBP,
    PEBP,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DynGammaMode {
    Fixed,
    Random,
}

#[derive(Clone, Debug)]
pub struct LRBPDecoderConfig {
    pub pre_iter: usize,
    pub num_sets: usize,
    pub set_max_iter: usize,
    pub odd_leg_max_iter: usize,
    pub even_leg_uniform_gamma: f64,
    pub gamma_dist_interval: (f64, f64),
    pub explicit_gammas: Option<Array2<f64>>,
    pub stopping_criterion: StoppingCriterion,
    pub logging: bool,
    pub seed: u64,
    pub osc_window: usize,
    pub friction_slope: f64,
    pub friction_shift: f64,
    pub tau: f64,
    pub r_dyn: usize,
    pub t_dyn: usize,
    pub dyn_mode: DynMode,
    pub gamma_dyn_penalty: f64,
    pub dyn_gamma_mode: DynGammaMode,
    pub gamma_dyn_center: f64,
    pub gamma_dyn_min: f64,
    pub gamma_dyn_max: f64,
}

impl Default for LRBPDecoderConfig {
    fn default() -> Self {
        Self {
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            odd_leg_max_iter: 60,
            even_leg_uniform_gamma: 0.0,
            gamma_dist_interval: (-0.24, 0.66),
            explicit_gammas: None,
            stopping_criterion: StoppingCriterion::NConv { stop_after: 1 },
            logging: false,
            seed: 0,
            osc_window: 5,
            friction_slope: 2.0,
            friction_shift: 0.0,
            tau: 0.2,
            r_dyn: 0,
            t_dyn: 0,
            dyn_mode: DynMode::EBP,
            gamma_dyn_penalty: 0.0,
            dyn_gamma_mode: DynGammaMode::Fixed,
            gamma_dyn_center: 0.0,
            gamma_dyn_min: -0.24,
            gamma_dyn_max: 0.66,
        }
    }
}

#[derive(Clone)]
struct LRBPState {
    rng_std: rand::rngs::StdRng,
    relay_uniform: Option<rand::distributions::Uniform<f64>>,
    relay_fixed_gamma: Option<f64>,
    dyn_uniform: Option<rand::distributions::Uniform<f64>>,
    dyn_fixed_gamma: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct QueueEntry {
    score: f64,
    check_idx: usize,
    stamp: u64,
}

impl Eq for QueueEntry {}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.check_idx == other.check_idx
            && self.stamp == other.stamp
            && self.score.to_bits() == other.score.to_bits()
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => self
                .check_idx
                .cmp(&other.check_idx)
                .then(self.stamp.cmp(&other.stamp)),
            ord => ord,
        }
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

        let relay_low = lrbp_config.gamma_dist_interval.0;
        let relay_high = lrbp_config.gamma_dist_interval.1;
        let (relay_uniform, relay_fixed_gamma) = if relay_low == relay_high {
            (None, Some(relay_low))
        } else {
            (Some(Uniform::new(relay_low, relay_high)), None)
        };

        let (dyn_uniform, dyn_fixed_gamma) = match lrbp_config.dyn_gamma_mode {
            DynGammaMode::Fixed => (None, Some(lrbp_config.gamma_dyn_center)),
            DynGammaMode::Random => {
                let dyn_low = lrbp_config.gamma_dyn_min;
                let dyn_high = lrbp_config.gamma_dyn_max;
                if dyn_low == dyn_high {
                    (None, Some(dyn_low))
                } else {
                    (Some(Uniform::new(dyn_low, dyn_high)), None)
                }
            }
        };

        LRBPState {
            rng_std,
            relay_uniform,
            relay_fixed_gamma,
            dyn_uniform,
            dyn_fixed_gamma,
        }
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

        if let Some(fixed_gamma) = self.state.relay_fixed_gamma {
            gammas.fill(fixed_gamma);
            self.bp_decoder.set_memory_strengths_f64(gammas);
            return;
        }

        for i in 0..gammas.len() {
            gammas[i] = self
                .state
                .relay_uniform
                .as_ref()
                .unwrap()
                .sample(&mut self.state.rng_std);
        }
        self.bp_decoder.set_memory_strengths_f64(gammas);
    }

    fn sample_dyn_gamma(&mut self) -> f64 {
        if let Some(fixed_gamma) = self.state.dyn_fixed_gamma {
            return fixed_gamma;
        }
        self.state
            .dyn_uniform
            .as_ref()
            .unwrap()
            .sample(&mut self.state.rng_std)
    }

    fn init_dyn_leg(&mut self, carry_marginal: &Array1<f64>) {
        self.bp_decoder.current_iteration = 0;
        self.bp_decoder.set_log_prior_ratio_f64(carry_marginal.clone());
        self.bp_decoder.set_posterior_ratios_f64(carry_marginal.clone());
        self.bp_decoder.initialize_check_to_variable();
        self.bp_decoder.initialize_variable_to_check();

        let gamma = self.sample_dyn_gamma();
        let mut memory_strengths = Array1::zeros(self.check_matrix().cols());
        memory_strengths.fill(gamma);
        self.bp_decoder.set_memory_strengths_f64(memory_strengths);
    }

    fn set_uniform_memory_strength(&mut self, gamma: f64) {
        let mut memory_strengths = Array1::zeros(self.check_matrix().cols());
        memory_strengths.fill(gamma);
        self.bp_decoder.set_memory_strengths_f64(memory_strengths);
    }

    fn init_standard_leg_from_marginal(
        &mut self,
        carry_marginal: &Array1<f64>,
        set_idx: usize,
        fixed_gamma: Option<f64>,
    ) {
        self.bp_decoder.current_iteration = 0;
        self.bp_decoder.set_log_prior_ratio_f64(carry_marginal.clone());
        self.bp_decoder.set_posterior_ratios_f64(carry_marginal.clone());
        self.bp_decoder.initialize_check_to_variable();
        self.bp_decoder.initialize_variable_to_check();

        match fixed_gamma {
            Some(gamma) => self.set_uniform_memory_strength(gamma),
            None => self.init_next_set(set_idx),
        }
    }

    fn hard_decision_from_marginal(&self, marginal: &Array1<f64>) -> Array1<Bit> {
        marginal.mapv(|x| if x < 0.0 { 1 } else { 0 })
    }

    fn syndrome_from_hard_decision(
        &self,
        detectors: ArrayView1<Bit>,
        hard_decision: &Array1<Bit>,
    ) -> Vec<u8> {
        let check_matrix = self.bp_decoder.check_matrix().to_csr();
        let mut residual = vec![0_u8; check_matrix.rows()];
        for (row_idx, row_vec) in check_matrix.outer_iterator().enumerate() {
            let mut parity = 0_u8;
            for (col_idx, val) in row_vec.iter() {
                if *val == 1 && hard_decision[col_idx] == 1 {
                    parity ^= 1;
                }
            }
            residual[row_idx] = parity ^ (detectors[row_idx] as u8);
        }
        residual
    }

    fn build_residual_targets(
        &self,
        detectors: ArrayView1<Bit>,
        carry_marginal: &Array1<f64>,
        check_neighbors: &[Vec<usize>],
    ) -> (Vec<usize>, Vec<bool>) {
        let hard_decision = self.hard_decision_from_marginal(carry_marginal);
        let current_syndrome = self.syndrome_from_hard_decision(detectors, &hard_decision);

        let mut target_checks = Vec::<usize>::new();
        for check_idx in 0..current_syndrome.len() {
            if detectors[check_idx] == 1 && current_syndrome[check_idx] == 1 {
                target_checks.push(check_idx);
            }
        }

        let mut target_variables = vec![false; self.bp_decoder.check_matrix().cols()];
        for &check_idx in &target_checks {
            for &var_idx in &check_neighbors[check_idx] {
                target_variables[var_idx] = true;
            }
        }

        (target_checks, target_variables)
    }

    fn run_standard_leg(
        &mut self,
        detectors: ArrayView1<Bit>,
        max_iter: usize,
    ) -> (DecodeResult, usize, f64) {
        let mut success = false;
        let mut decoded_detectors = Array1::default(detectors.dim());
        let mut executed_iterations = 0usize;
        let mut elapsed_seconds = 0.0_f64;

        for _ in 0..max_iter {
            let iter_started = Instant::now();
            self.bp_decoder.run_iteration(detectors);
            elapsed_seconds += iter_started.elapsed().as_secs_f64();
            executed_iterations += 1;

            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors, decoded_detectors.view());

            if success {
                break;
            }
            self.bp_decoder.current_iteration += 1;
        }

        let mut result = self
            .bp_decoder
            .build_result(success, decoded_detectors, max_iter);
        result.iterations = executed_iterations;
        (result, executed_iterations, elapsed_seconds)
    }

    fn run_even_surgical_leg(
        &mut self,
        detectors: ArrayView1<Bit>,
        max_iter: usize,
        target_checks: &[usize],
        target_variables: &[bool],
        check_neighbors: &[Vec<usize>],
    ) -> (DecodeResult, usize, f64) {
        let mut success = false;
        let mut decoded_detectors = Array1::default(detectors.dim());
        let mut executed_iterations = 0usize;
        let mut elapsed_seconds = 0.0_f64;

        let mut is_target_check = vec![false; check_neighbors.len()];
        for &check_idx in target_checks {
            if check_idx < is_target_check.len() {
                is_target_check[check_idx] = true;
            }
        }

        for _ in 0..max_iter {
            let iter_started = Instant::now();

            let mut phase1_variables = vec![false; target_variables.len()];
            for check_idx in 0..check_neighbors.len() {
                if is_target_check[check_idx] {
                    continue;
                }
                self.bp_decoder
                    .run_check_to_variable_update_for_check(detectors, check_idx);
                for &var_idx in &check_neighbors[check_idx] {
                    phase1_variables[var_idx] = true;
                }
            }

            for (var_idx, should_update) in phase1_variables.iter().enumerate() {
                if *should_update {
                    self.bp_decoder.run_variable_to_check_update_for_variable(var_idx);
                }
            }

            for &check_idx in target_checks {
                self.bp_decoder
                    .run_check_to_variable_update_for_check(detectors, check_idx);
                for &var_idx in &check_neighbors[check_idx] {
                    self.bp_decoder.run_variable_to_check_update_for_variable(var_idx);
                }
            }

            self.bp_decoder.recompute_hard_decision();

            elapsed_seconds += iter_started.elapsed().as_secs_f64();
            executed_iterations += 1;

            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors, decoded_detectors.view());

            if success {
                break;
            }
            self.bp_decoder.current_iteration += 1;
        }

        let mut result = self
            .bp_decoder
            .build_result(success, decoded_detectors, max_iter);
        result.iterations = executed_iterations;
        (result, executed_iterations, elapsed_seconds)
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
        let _tau = self.lrbp_config.tau.max(1e-12);

        for edge_idx in 0..raw_messages.len() {
            let raw = raw_messages[edge_idx];
            let check_idx = indices[edge_idx];

            let edge_history = &mut history[edge_idx];
            edge_history.push_back(raw);
            while edge_history.len() > window_size {
                edge_history.pop_front();
            }

            let oscillation = edge_history.iter().sum::<f64>().abs();
            let _friction = 1.0 / (1.0 + (-beta * (oscillation - delta)).exp());

            let _t_eff = self.lrbp_config.tau.max(1e-12) / ((current_iter + 1) as f64).max(1.0);
            // let mut p_kick = (-oscillation / t_eff).exp();
            let mut p_kick: f64 = 0.3;
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

            // let do_kick = self.state.rng_std.gen_bool(p_kick);
            // let damped = friction.clamp(0.0, 1.0) * raw;
            let damped = 1.0 * raw; // In dropout version, we skip the Langevin "kick" and only apply damping to the new messages.
            // let message = if do_kick { -damped } else { damped };
            let message = damped;
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
    ) -> (DecodeResult, usize, f64) {
        let mut success: bool = false;
        let mut decoded_detectors = Array1::default(detectors.dim());
        let mut executed_iterations = 0usize;
        let mut elapsed_seconds = 0.0_f64;

        for _ in 0..max_iter {
            let iter_started = Instant::now();
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
            elapsed_seconds += iter_started.elapsed().as_secs_f64();
            executed_iterations += 1;

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

        let mut result = self
            .bp_decoder
            .build_result(success, decoded_detectors, max_iter);
        result.iterations = executed_iterations;
        (result, executed_iterations, elapsed_seconds)
    }

    fn check_error_probability(
        &self,
        check_idx: usize,
        posterior_ratios: &Array1<f64>,
        check_neighbors: &[Vec<usize>],
    ) -> f64 {
        let mut product = 1.0_f64;
        for &var_idx in &check_neighbors[check_idx] {
            let lj = posterior_ratios[var_idx].abs();
            let term = 1.0 - 2.0 / (1.0 + lj.exp());
            product *= term;
        }
        let p = 0.5 * (1.0 - product);
        p.clamp(0.0, 1.0)
    }

    fn check_priority_score(
        &self,
        check_idx: usize,
        posterior_ratios: &Array1<f64>,
        check_neighbors: &[Vec<usize>],
        update_counts: &[usize],
    ) -> f64 {
        let p = self.check_error_probability(check_idx, posterior_ratios, check_neighbors);
        match self.lrbp_config.dyn_mode {
            DynMode::EBP => p,
            DynMode::PEBP => {
                let penalty = self.lrbp_config.gamma_dyn_penalty.clamp(0.0, 1.0);
                p + penalty * update_counts[check_idx] as f64
            }
        }
    }

    fn push_check(
        &self,
        heap: &mut BinaryHeap<QueueEntry>,
        check_idx: usize,
        posterior_ratios: &Array1<f64>,
        check_neighbors: &[Vec<usize>],
        update_counts: &[usize],
        stamps: &mut [u64],
    ) {
        stamps[check_idx] = stamps[check_idx].saturating_add(1);
        heap.push(QueueEntry {
            score: self.check_priority_score(
                check_idx,
                posterior_ratios,
                check_neighbors,
                update_counts,
            ),
            check_idx,
            stamp: stamps[check_idx],
        });
    }

    fn run_dyn_sweep(
        &mut self,
        detectors: ArrayView1<Bit>,
        check_neighbors: &[Vec<usize>],
        variable_neighbors: &[Vec<usize>],
    ) -> bool {
        let num_checks = check_neighbors.len();
        if num_checks == 0 {
            return false;
        }

        let mut posterior_ratios = self.bp_decoder.posterior_ratios_f64();
        let mut heap = BinaryHeap::<QueueEntry>::new();
        let mut stamps = vec![0_u64; num_checks];
        let mut update_counts = vec![0_usize; num_checks];

        for check_idx in 0..num_checks {
            self.push_check(
                &mut heap,
                check_idx,
                &posterior_ratios,
                check_neighbors,
                &update_counts,
                &mut stamps,
            );
        }

        match self.lrbp_config.dyn_mode {
            DynMode::EBP => {
                let mut updated = vec![false; num_checks];
                let mut num_updated = 0usize;

                while num_updated < num_checks {
                    let Some(entry) = heap.pop() else {
                        break;
                    };
                    if stamps[entry.check_idx] != entry.stamp || updated[entry.check_idx] {
                        continue;
                    }

                    let check_idx = entry.check_idx;
                    updated[check_idx] = true;
                    num_updated += 1;

                    self.bp_decoder
                        .run_check_to_variable_update_for_check(detectors, check_idx);

                    for &var_idx in &check_neighbors[check_idx] {
                        self.bp_decoder
                            .run_variable_to_check_update_for_variable(var_idx);
                        posterior_ratios[var_idx] = self.bp_decoder.posterior_ratio_f64(var_idx);

                        for &neighbor_check in &variable_neighbors[var_idx] {
                            if neighbor_check == check_idx || updated[neighbor_check] {
                                continue;
                            }
                            self.push_check(
                                &mut heap,
                                neighbor_check,
                                &posterior_ratios,
                                check_neighbors,
                                &update_counts,
                                &mut stamps,
                            );
                        }
                    }
                }
            }
            DynMode::PEBP => {
                for _ in 0..num_checks {
                    let mut selected: Option<usize> = None;
                    while let Some(entry) = heap.pop() {
                        if stamps[entry.check_idx] != entry.stamp {
                            continue;
                        }
                        selected = Some(entry.check_idx);
                        break;
                    }

                    let Some(check_idx) = selected else {
                        break;
                    };

                    self.bp_decoder
                        .run_check_to_variable_update_for_check(detectors, check_idx);

                    for &var_idx in &check_neighbors[check_idx] {
                        self.bp_decoder
                            .run_variable_to_check_update_for_variable(var_idx);
                        posterior_ratios[var_idx] = self.bp_decoder.posterior_ratio_f64(var_idx);

                        for &neighbor_check in &variable_neighbors[var_idx] {
                            self.push_check(
                                &mut heap,
                                neighbor_check,
                                &posterior_ratios,
                                check_neighbors,
                                &update_counts,
                                &mut stamps,
                            );
                        }
                    }

                    update_counts[check_idx] = update_counts[check_idx].saturating_add(1);
                    self.push_check(
                        &mut heap,
                        check_idx,
                        &posterior_ratios,
                        check_neighbors,
                        &update_counts,
                        &mut stamps,
                    );
                }
            }
        }

        self.bp_decoder.recompute_hard_decision();
        let decoded_detectors = self.bp_decoder.compute_decoded_detectors();
        self.bp_decoder
            .check_convergence(detectors, decoded_detectors.view())
    }

    fn run_dyn_leg(
        &mut self,
        detectors: ArrayView1<Bit>,
        check_neighbors: &[Vec<usize>],
        variable_neighbors: &[Vec<usize>],
    ) -> (DecodeResult, usize, f64) {
        let mut success = false;
        let mut executed_iterations = 0usize;
        let mut elapsed_seconds = 0.0_f64;

        for _ in 0..self.lrbp_config.t_dyn {
            let iter_started = Instant::now();
            success = self.run_dyn_sweep(detectors, check_neighbors, variable_neighbors);
            elapsed_seconds += iter_started.elapsed().as_secs_f64();
            executed_iterations += 1;

            if success {
                break;
            }
            self.bp_decoder.current_iteration += 1;
        }

        let decoded_detectors = self.bp_decoder.compute_decoded_detectors();
        let mut result = self
            .bp_decoder
            .build_result(success, decoded_detectors, self.lrbp_config.t_dyn);
        result.iterations = executed_iterations;
        (result, executed_iterations, elapsed_seconds)
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
        // Legacy LRBP (Langevin/dropout/dynamic queue) code paths are intentionally retained in
        // this file for experimental reference, but decode_detailed now uses the alternating
        // relay/surgical-leg flow described by the current LR-BP experiment design.

        let total_legs = self.lrbp_config.num_sets.max(1);
        let mut total_iterations: usize = 0;

        let mut leg_success: Vec<bool> = Vec::with_capacity(total_legs);
        let mut leg_iterations: Vec<usize> = Vec::with_capacity(total_legs);
        let mut leg_negative_llr_counts: Vec<usize> = Vec::with_capacity(total_legs);
        let mut leg_decodings: Vec<Array1<Bit>> = Vec::with_capacity(total_legs);
        let mut leg_posteriors: Vec<Array1<f64>> = Vec::with_capacity(total_legs);

        let mut relay_parallel_elapsed_seconds = 0.0_f64;
        let mut relay_parallel_iterations = 0usize;
        let mut dyn_phase_elapsed_seconds = 0.0_f64;
        let mut dyn_phase_iterations = 0usize;

        self.bp_decoder.initialize_decoder();

        let num_checks = self.bp_decoder.check_matrix().rows();
        let check_neighbors: Vec<Vec<usize>> = (0..num_checks)
            .map(|check_idx| self.bp_decoder.variable_neighbors(check_idx))
            .collect();

        let (mut result, pre_iter_count, pre_elapsed) =
            self.run_standard_leg(detectors, self.lrbp_config.pre_iter);
        relay_parallel_elapsed_seconds += pre_elapsed;
        relay_parallel_iterations += pre_iter_count;
        total_iterations += result.iterations;

        leg_success.push(result.success);
        leg_iterations.push(result.iterations);
        leg_negative_llr_counts.push(result.posterior_ratios.iter().filter(|x| **x < 0.0).count());
        leg_decodings.push(result.decoding.clone());
        leg_posteriors.push(result.posterior_ratios.clone());

        if result.success {
            result.iterations = total_iterations;
            result.extra = BPExtraResult::RelayTrace {
                leg_success,
                leg_iterations,
                leg_negative_llr_counts,
                leg_decodings,
                leg_posteriors,
                relay_parallel_avg_iter_seconds: if relay_parallel_iterations > 0 {
                    Some(relay_parallel_elapsed_seconds / relay_parallel_iterations as f64)
                } else {
                    None
                },
                dyn_phase_avg_iter_seconds: if dyn_phase_iterations > 0 {
                    Some(dyn_phase_elapsed_seconds / dyn_phase_iterations as f64)
                } else {
                    None
                },
            };
            return result;
        }

        let mut carry_marginal = self.bp_decoder.posterior_ratios_f64();

        for leg_idx in 1..total_legs {
            let (temp_result, leg_iter_count, leg_elapsed, is_surgical_leg) = if leg_idx == 1 {
                self.init_standard_leg_from_marginal(&carry_marginal, leg_idx, None);
                let (temp_result, leg_iter_count, leg_elapsed) =
                    self.run_standard_leg(detectors, self.lrbp_config.set_max_iter);
                (temp_result, leg_iter_count, leg_elapsed, false)
            } else if leg_idx % 2 == 0 {
                let (target_checks, target_variables) =
                    self.build_residual_targets(detectors, &carry_marginal, &check_neighbors);
                let seeded_marginal = carry_marginal.clone();

                self.bp_decoder.current_iteration = 0;
                self.bp_decoder.set_log_prior_ratio_f64(seeded_marginal.clone());
                self.bp_decoder.set_posterior_ratios_f64(seeded_marginal);
                self.bp_decoder.initialize_check_to_variable();
                self.bp_decoder.initialize_variable_to_check();
                self.set_uniform_memory_strength(self.lrbp_config.even_leg_uniform_gamma);

                let (temp_result, leg_iter_count, leg_elapsed) = self.run_even_surgical_leg(
                    detectors,
                    self.lrbp_config.set_max_iter,
                    &target_checks,
                    &target_variables,
                    &check_neighbors,
                );
                (temp_result, leg_iter_count, leg_elapsed, true)
            } else {
                self.init_standard_leg_from_marginal(
                    &carry_marginal,
                    leg_idx,
                    None,
                );
                let (temp_result, leg_iter_count, leg_elapsed) =
                    self.run_standard_leg(detectors, self.lrbp_config.odd_leg_max_iter);
                (temp_result, leg_iter_count, leg_elapsed, false)
            };

            if is_surgical_leg {
                dyn_phase_elapsed_seconds += leg_elapsed;
                dyn_phase_iterations += leg_iter_count;
            } else {
                relay_parallel_elapsed_seconds += leg_elapsed;
                relay_parallel_iterations += leg_iter_count;
            }

            total_iterations += temp_result.iterations;
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

            result = temp_result;
            if result.success {
                break;
            }

            carry_marginal = self.bp_decoder.posterior_ratios_f64();
        }

        result.iterations = total_iterations;
        result.extra = BPExtraResult::RelayTrace {
            leg_success,
            leg_iterations,
            leg_negative_llr_counts,
            leg_decodings,
            leg_posteriors,
            relay_parallel_avg_iter_seconds: if relay_parallel_iterations > 0 {
                Some(relay_parallel_elapsed_seconds / relay_parallel_iterations as f64)
            } else {
                None
            },
            dyn_phase_avg_iter_seconds: if dyn_phase_iterations > 0 {
                Some(dyn_phase_elapsed_seconds / dyn_phase_iterations as f64)
            } else {
                None
            },
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
