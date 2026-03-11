use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use crate::decoder::{BPExtraResult, Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix};

pub mod config;
pub mod evaluate;
pub mod ga_ops;
pub mod init_population;
pub mod trace;

use config::{
    AdaptiveMemoryMode, AdaptivePerturbationSignMode, AdaptivePerturbationThresholdMode,
    GammaMode, InitStrategy, PerturbationMethod, SLGMBPDecoderConfig,
};
use evaluate::{fitness_from_final_marginal, fitness_from_ms_cumsum, residual_weight};
use ga_ops::{build_next_generation, PopulationMember};
use init_population::{
    apply_perturbation_state, initialize_llr_population, sample_perturbation_state,
    PerturbationState,
};
use trace::{
    SLGMBPDetailedDynamicsTrace, SLGMBPDynamicsEntry, SLGMBPScoreSpikeGenerationSnapshot,
    SLGMBPScoreSpikeTrace,
};

#[derive(Clone)]
pub struct SLGMBPDecoder {
    check_matrix: Arc<SparseBitMatrix>,
    base_min_sum_config: Arc<MinSumDecoderConfig>,
    config: Arc<SLGMBPDecoderConfig>,
}

impl SLGMBPDecoder {
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        config: Arc<SLGMBPDecoderConfig>,
    ) -> Self {
        Self {
            check_matrix,
            base_min_sum_config: min_sum_config,
            config,
        }
    }

    fn prior_llr(&self) -> Array1<f64> {
        self.base_min_sum_config.log_prior_ratios()
    }

    fn run_min_sum_phase(
        &self,
        detectors: ArrayView1<'_, Bit>,
        init_llr: &Array1<f64>,
        drop_seed: Option<u64>,
    ) -> (Array1<Bit>, Array1<f64>, usize, bool, f64) {
        let priors_prob = self.base_min_sum_config.error_priors.clone();
        let cfg = self.config.min_sum_template_from_priors(
            priors_prob,
            self.config.t_ms,
            false,
            drop_seed,
        );

        let mut decoder = MinSumBPDecoder::<f64>::new(self.check_matrix.clone(), Arc::new(cfg));

        decoder.current_iteration = 0;
        decoder.set_log_prior_ratio_f64(init_llr.clone());
        decoder.set_posterior_ratios_f64(init_llr.clone());
        decoder.initialize_check_to_variable();
        decoder.initialize_variable_to_check();

        let mut decoded = decoder.compute_decoded_detectors();
        let mut success = false;
        let mut llr_cumsum_abs_sum = 0.0;

        for _ in 0..self.config.t_ms {
            decoder.run_iteration(detectors);
            decoder.current_iteration += 1;
            let posterior = decoder.posterior_ratios_f64();
            llr_cumsum_abs_sum += posterior.iter().sum::<f64>().abs();
            decoded = decoder.compute_decoded_detectors();
            success = decoder.check_convergence(detectors, decoded.view());
            if success {
                break;
            }
        }

        (
            decoder.current_decoding().clone(),
            decoder.posterior_ratios_f64(),
            decoder.current_iteration,
            success,
            llr_cumsum_abs_sum,
        )
    }

    fn run_mem_bp_phase(
        &self,
        detectors: ArrayView1<'_, Bit>,
        prior_llr: &Array1<f64>,
        child_llr: &Array1<f64>,
        gamma_per_variable: &Array1<f64>,
        max_iter: usize,
        reset_marginal: bool,
        enable_variable_message_drop: bool,
        drop_seed: Option<u64>,
    ) -> (Array1<Bit>, Array1<f64>, usize, bool) {
        let priors_prob = self.base_min_sum_config.error_priors.clone();
        let cfg = self.config.min_sum_template_from_priors(
            priors_prob,
            max_iter,
            enable_variable_message_drop,
            drop_seed,
        );
        let mut decoder = MinSumBPDecoder::<f64>::new(self.check_matrix.clone(), Arc::new(cfg));

        let mut prev_marginal = self.initial_marginal(prior_llr, child_llr, reset_marginal);

        decoder.current_iteration = 0;
        decoder.set_log_prior_ratio_f64(prev_marginal.clone());
        decoder.set_posterior_ratios_f64(prev_marginal.clone());
        decoder.initialize_check_to_variable();
        decoder.initialize_variable_to_check();

        let mut decoded = decoder.compute_decoded_detectors();
        let mut success = false;

        for _ in 0..max_iter {
            let one_minus_gamma = gamma_per_variable.mapv(|g| 1.0 - g);
            let bias = (&one_minus_gamma * prior_llr) + (gamma_per_variable * &prev_marginal);
            decoder.set_log_prior_ratio_f64(bias);

            decoder.run_iteration(detectors);
            decoder.current_iteration += 1;

            prev_marginal = decoder.posterior_ratios_f64();
            decoded = decoder.compute_decoded_detectors();
            success = decoder.check_convergence(detectors, decoded.view());
            if success {
                break;
            }
        }

        (
            decoder.current_decoding().clone(),
            decoder.posterior_ratios_f64(),
            decoder.current_iteration,
            success,
        )
    }

    fn initial_marginal(
        &self,
        prior_llr: &Array1<f64>,
        child_llr: &Array1<f64>,
        reset_marginal: bool,
    ) -> Array1<f64> {
        if reset_marginal {
            prior_llr.clone()
        } else {
            child_llr.mapv(|v| self.config.eta * v)
        }
    }

    fn adaptive_perturbation_llr_threshold_bounds(&self) -> (f64, f64) {
        let lo = self
            .config
            .adaptive_perturbation_llr_threshold_min
            .abs()
            .min(self.config.adaptive_perturbation_llr_threshold_max.abs());
        let hi = self
            .config
            .adaptive_perturbation_llr_threshold_min
            .abs()
            .max(self.config.adaptive_perturbation_llr_threshold_max.abs());
        (lo, hi)
    }

    fn initial_adaptive_perturbation_llr_threshold(&self) -> f64 {
        match self.config.adaptive_perturbation_llr_threshold_mode {
            AdaptivePerturbationThresholdMode::Constant => {
                self.config.adaptive_perturbation_llr_threshold.abs()
            }
            AdaptivePerturbationThresholdMode::GenerationLog10Iter => {
                let (lo, _) = self.adaptive_perturbation_llr_threshold_bounds();
                lo
            }
        }
    }

    fn next_adaptive_perturbation_llr_threshold(
        &self,
        current_threshold: f64,
        generation_iters: usize,
    ) -> f64 {
        match self.config.adaptive_perturbation_llr_threshold_mode {
            AdaptivePerturbationThresholdMode::Constant => {
                self.config.adaptive_perturbation_llr_threshold.abs()
            }
            AdaptivePerturbationThresholdMode::GenerationLog10Iter => {
                let (lo, hi) = self.adaptive_perturbation_llr_threshold_bounds();
                let increment = self.config.adaptive_perturbation_llr_threshold_factor
                    * (generation_iters.max(1) as f64).log10();
                (current_threshold.clamp(lo, hi) + increment).clamp(lo, hi)
            }
        }
    }

    fn build_adaptive_prior(
        &self,
        base_prior_llr: &Array1<f64>,
        posterior: &Array1<f64>,
        adaptive_perturbation_llr_threshold: f64,
    ) -> Array1<f64> {
        if !self.config.adaptive_perturbation {
            return base_prior_llr.clone();
        }

        let threshold = adaptive_perturbation_llr_threshold.abs();
        let add_val = self.config.adaptive_perturbation_factor.ln();
        
        let mut rng = rand::thread_rng();

        Array1::from_iter(base_prior_llr.iter().zip(posterior.iter()).map(|(&prior, &llr)| {
            if llr.abs() < threshold {
                let sign = match self.config.adaptive_perturbation_sign_mode {
                    AdaptivePerturbationSignMode::Random => {
                        if rng.gen_bool(self.config.adaptive_perturbation_positive_sign_prob) {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    AdaptivePerturbationSignMode::AlwaysNegative => -1.0,
                };
                prior + (sign * add_val)
            } else {
                prior
            }
        }))
    }

    fn build_generation_phase_prior(
        &self,
        base_prior_llr: &Array1<f64>,
        child: &mut PopulationMember,
        rng: &mut StdRng,
        adaptive_perturbation_llr_threshold: f64,
    ) -> Array1<f64> {
        let adaptive_prior = self.build_adaptive_prior(
            base_prior_llr,
            &child.posterior,
            adaptive_perturbation_llr_threshold,
        );
        if !self.config.continue_perturbation {
            return adaptive_prior;
        }

        if self.config.sequential_mc
            || self.config.perturbation_method == PerturbationMethod::Resample
            || matches!(child.perturbation_state, PerturbationState::None)
        {
            child.perturbation_state = sample_perturbation_state(
                adaptive_prior.len(),
                self.config.init_perturbation_mode,
                self.config.sigma2,
                self.config.delta,
                rng,
            );
        }

        apply_perturbation_state(&adaptive_prior, &child.perturbation_state)
    }

    fn sample_default_memory_strength(&self, n_variables: usize, rng: &mut StdRng) -> Array1<f64> {
        match self.config.gamma_mode {
            GammaMode::Fixed => Array1::from_elem(n_variables, self.config.gamma_fixed),
            // interval_random: sample one gamma per variable node.
            GammaMode::IntervalRandomPerVariable => {
                let (a, b) = self.config.gamma_interval;
                let lo = a.min(b);
                let hi = a.max(b);
                if (hi - lo).abs() < f64::EPSILON {
                    Array1::from_elem(n_variables, lo)
                } else {
                    Array1::from_shape_simple_fn(n_variables, || rng.gen_range(lo..=hi))
                }
            }
        }
    }

    fn sample_adaptive_adjacent_gamma(&self, rng: &mut StdRng) -> f64 {
        let (a, b) = self.config.adaptive_memory_adjacent_gamma_interval;
        match self.config.adaptive_memory_mode {
            AdaptiveMemoryMode::DirectInterval => {
                let lo = a.min(b);
                let hi = a.max(b);
                if (hi - lo).abs() < f64::EPSILON {
                    lo
                } else {
                    rng.gen_range(lo..=hi)
                }
            }
            AdaptiveMemoryMode::ProbabilisticFlip => {
                let lo = a.abs().min(b.abs());
                let hi = a.abs().max(b.abs());
                if (hi - lo).abs() < f64::EPSILON {
                    lo
                } else {
                    rng.gen_range(lo..=hi)
                }
            }
        }
    }

    fn sample_member_memory_strength(
        &self,
        posterior: &Array1<f64>,
        residual_adjacent_variable_indices: &[usize],
        rng: &mut StdRng,
    ) -> Array1<f64> {
        let mut gamma = self.sample_default_memory_strength(posterior.len(), rng);
        if !self.config.adaptive_memory || residual_adjacent_variable_indices.is_empty() {
            return gamma;
        }

        let mut adjacent_sorted = residual_adjacent_variable_indices.to_vec();
        adjacent_sorted.sort_by(|&lhs, &rhs| {
            posterior[rhs]
                .abs()
                .partial_cmp(&posterior[lhs].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for idx in adjacent_sorted {
            if idx >= gamma.len() {
                continue;
            }
            let sampled = self.sample_adaptive_adjacent_gamma(rng);
            gamma[idx] = match self.config.adaptive_memory_mode {
                AdaptiveMemoryMode::DirectInterval => sampled,
                AdaptiveMemoryMode::ProbabilisticFlip => {
                    let llr_abs = posterior[idx].abs().max(f64::EPSILON);
                    let p_neg = (-self.config.adaptive_memory_zeta / llr_abs).exp().clamp(0.0, 1.0);
                    if rng.gen_bool(p_neg) {
                        -sampled.abs()
                    } else {
                        sampled.abs()
                    }
                }
            };
        }

        gamma
    }

    fn next_drop_seed(&self, rng: &mut StdRng) -> Option<u64> {
        if self.config.drop_p > 0.0 {
            Some(rng.gen())
        } else {
            None
        }
    }

    fn max_iter(&self) -> usize {
        if self.config.biased_relay_mode {
            return self.max_iter_biased_relay();
        }
        self.config.t_ms + self.config.g_max * self.config.t_mem
    }

    fn max_iter_biased_relay(&self) -> usize {
        self.config.biased_relay_maximum_round
            * (self.config.biased_relay_t_0
                + self.config.biased_relay_r_relay * self.config.biased_relay_r_relay_iter)
    }

    fn sample_relay_gamma(&self, n_variables: usize, rng: &mut StdRng) -> Array1<f64> {
        let (a, b) = self.config.gamma_interval;
        let lo = a.min(b);
        let hi = a.max(b);
        if (hi - lo).abs() < f64::EPSILON {
            Array1::from_elem(n_variables, lo)
        } else {
            Array1::from_shape_simple_fn(n_variables, || rng.gen_range(lo..=hi))
        }
    }

    fn estimated_error_weight(&self, decoding: &Array1<Bit>, prior_llr: &Array1<f64>) -> f64 {
        decoding
            .iter()
            .zip(prior_llr.iter())
            .map(|(&bit, &llr)| (bit as f64) * llr)
            .sum::<f64>()
    }

    fn residual_check_indices(
        &self,
        decoded_detectors: &Array1<Bit>,
        detectors: ArrayView1<'_, Bit>,
    ) -> Vec<usize> {
        decoded_detectors
            .iter()
            .zip(detectors.iter())
            .enumerate()
            .filter_map(|(idx, (a, b))| if *a != *b { Some(idx) } else { None })
            .collect()
    }

    fn residual_adjacent_llrs(
        &self,
        check_matrix_csr: &SparseBitMatrix,
        residual_checks: &[usize],
        posterior: &Array1<f64>,
    ) -> (Vec<usize>, Vec<f32>) {
        let mut vars = BTreeSet::<usize>::new();
        for &check_idx in residual_checks {
            if let Some(row) = check_matrix_csr.outer_view(check_idx) {
                for &var_idx in row.indices() {
                    vars.insert(var_idx);
                }
            }
        }

        let var_indices: Vec<usize> = vars.into_iter().collect();
        let llrs = var_indices
            .iter()
            .map(|&i| posterior[i] as f32)
            .collect::<Vec<_>>();

        (var_indices, llrs)
    }

    fn build_score_spike_trace(
        &self,
        generation_best_fitness: &[f64],
        generation_best_posteriors: &[Array1<f64>],
        generation_best_adjacent_variable_indices: &[Vec<usize>],
        generation_memory_strengths: &[Array1<f64>],
        gamma_history: &[f64],
    ) -> Option<SLGMBPScoreSpikeTrace> {
        if generation_best_fitness.len() < 2
            || generation_best_posteriors.len() != generation_best_fitness.len()
            || generation_best_adjacent_variable_indices.len() != generation_best_fitness.len()
            || generation_memory_strengths.len() != generation_best_fitness.len()
            || gamma_history.len() != generation_best_fitness.len()
        {
            return None;
        }

        let mut spike_generation_index: Option<usize> = None;
        let mut best_delta = f64::NEG_INFINITY;

        for gen in 1..generation_best_fitness.len() {
            let prev = generation_best_fitness[gen - 1];
            let curr = generation_best_fitness[gen];
            if !(prev.is_finite() && curr.is_finite()) {
                continue;
            }
            let delta = curr - prev;
            if delta > best_delta {
                best_delta = delta;
                spike_generation_index = Some(gen);
            }
        }

        let spike_gen = spike_generation_index?;
        if !(best_delta.is_finite() && best_delta > 0.0) {
            return None;
        }

        let prev_gen = spike_gen - 1;
        let prev_score = generation_best_fitness[prev_gen];
        let spike_score = generation_best_fitness[spike_gen];
        let delta_order_log10 = if prev_score > 0.0 && spike_score > 0.0 {
            Some(spike_score.log10() - prev_score.log10())
        } else {
            None
        };

        let window_start = spike_gen.saturating_sub(3);
        let window_end = (spike_gen + 3).min(generation_best_fitness.len() - 1);

        let mut snapshots = Vec::<SLGMBPScoreSpikeGenerationSnapshot>::new();
        for gen in window_start..=window_end {
            let posterior = generation_best_posteriors[gen].clone().to_vec();
            let memory_strength_all_variables = generation_memory_strengths[gen].clone().to_vec();
            if memory_strength_all_variables.len() != posterior.len() {
                return None;
            }
            snapshots.push(SLGMBPScoreSpikeGenerationSnapshot {
                generation_index: gen,
                posterior_llr_all_variables: posterior,
                memory_strength_all_variables,
            });
        }

        Some(SLGMBPScoreSpikeTrace {
            spike_generation_index: spike_gen,
            previous_generation_index: prev_gen,
            spike_prev_score: prev_score,
            spike_score,
            spike_delta_score: best_delta,
            spike_delta_order_log10: delta_order_log10,
            window_start_generation: window_start,
            window_end_generation: window_end,
            spike_residual_adjacent_variable_indices: generation_best_adjacent_variable_indices
                [spike_gen]
                .clone(),
            snapshots,
        })
    }

    /// Biased-relay decode mode: combines adaptive perturbation (bias) with relay-bp
    /// gamma resampling. Each "round" maps to one GA generation and consists of:
    ///   (a) bias + constant mem-bp leg  (round ≥ 2; round 1 has no bias)
    ///   (b) R_relay relay-bp legs with per-variable gamma resampling
    #[allow(clippy::too_many_arguments)]
    fn decode_detailed_biased_relay(
        &mut self,
        detectors: ArrayView1<'_, Bit>,
        prior_llr: Array1<f64>,
        mut members: Vec<PopulationMember>,
        mut best_decoding: Array1<Bit>,
        mut best_posterior: Array1<f64>,
        mut best_fitness: f64,
        phase1_iterations: usize,
        initial_adaptive_threshold: f64,
        mut rng: StdRng,
    ) -> DecodeResult {
        let max_iter = self.max_iter_biased_relay();
        let mut total_iterations = phase1_iterations;
        let mut generation_best_fitness = Vec::<f64>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut residual_weight_history = Vec::<usize>::new();
        let mut adaptive_perturbation_llr_threshold = initial_adaptive_threshold;

        // Round 1 relay legs (phase 1 constant mem-bp already ran as the init population).
        // Now run R_relay relay-bp legs for each member, carrying marginals forward.
        {
            let mut round1_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;
            let mut round1_iterations = 0usize;

            for member in members.iter_mut() {
                let mut current_posterior = member.posterior.clone();
                let mut relay_converged = false;
                let mut last_decoding = Array1::<Bit>::zeros(prior_llr.len());
                let mut member_iterations = 0usize;

                for _relay_leg in 0..self.config.biased_relay_r_relay {
                    let relay_gamma = self.sample_relay_gamma(prior_llr.len(), &mut rng);

                    let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                        detectors,
                        &prior_llr,
                        &current_posterior,
                        &relay_gamma,
                        self.config.biased_relay_r_relay_iter,
                        false, // relay legs always carry marginals forward
                        self.config.drop_p > 0.0,
                        self.next_drop_seed(&mut rng),
                    );

                    member_iterations += iters;
                    current_posterior = posterior.clone();
                    last_decoding = decoding.clone();

                    if success {
                        let solution_llr_cost = self.estimated_error_weight(&decoding, &prior_llr);
                        let replace = match round1_success {
                            None => true,
                            Some((best_iter, best_llr_cost, _, _)) => {
                                member_iterations < best_iter
                                    || (member_iterations == best_iter
                                        && solution_llr_cost < best_llr_cost)
                            }
                        };
                        if replace {
                            round1_success = Some((
                                member_iterations,
                                solution_llr_cost,
                                decoding.clone(),
                                posterior.clone(),
                            ));
                        }
                        relay_converged = true;
                        break;
                    }
                }

                // Score member using final posterior (after all relay legs).
                let decoded_detectors_final = self.get_detectors(last_decoding.view());
                let residual_w = residual_weight(&decoded_detectors_final, detectors);
                let fitness = fitness_from_final_marginal(
                    residual_w,
                    &current_posterior,
                    self.config.fitness_alpha,
                    self.config.fitness_beta,
                    self.config.fitness_low_llr_mu,
                    self.config.fitness_low_llr_threshold,
                );
                residual_weight_history.push(residual_w);

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_decoding = last_decoding.clone();
                    best_posterior = current_posterior.clone();
                }

                member.posterior = current_posterior;
                member.fitness = fitness;

                round1_iterations += member_iterations;

                if relay_converged {
                    break;
                }
            }

            let round1_gamma_mean = self.config.init_gamma;
            gamma_history.push(round1_gamma_mean);

            if let Some((_, _, r1_decoding, r1_posterior)) = round1_success {
                total_iterations += round1_iterations;
                return DecodeResult {
                    decoding: r1_decoding.clone(),
                    decoded_detectors: self.get_detectors(r1_decoding.view()),
                    posterior_ratios: r1_posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(r1_decoding.view()),
                    iterations: total_iterations,
                    max_iter,
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(r1_posterior),
                        residual_weight_history,
                        gamma_history,
                        detailed_dynamics: None,
                        score_spike_trace: None,
                    },
                };
            }

            total_iterations += round1_iterations;
            let gen_best = members.iter().map(|m| m.fitness).fold(f64::NEG_INFINITY, f64::max);
            generation_best_fitness.push(gen_best);
        }

        // Rounds 2..maximum_round: bias + constant mem-bp + R_relay relay legs.
        for _round in 1..self.config.biased_relay_maximum_round {
            let children = build_next_generation(
                &members,
                self.config.elite_count,
                self.config.sequential_mc,
                self.config.mutation_rate,
                self.config.mutation_llr_abs_threshold,
                self.config.selection_mode,
                self.config.weighted_selection_mode,
                self.config.perturbation_method,
                self.config.tournament_size,
                &mut rng,
            );

            let mut next_members = Vec::<PopulationMember>::with_capacity(children.len());
            let mut gen_best = f64::NEG_INFINITY;
            let mut generation_iterations = 0usize;
            let mut generation_max_iters = 0usize;
            let mut generation_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;
            let mut gen_best_gamma_mean = 0.0f64;

            for mut child in children {
                let mut child_iterations = 0usize;
                // (a) Build biased prior from the base prior + child's posterior.
                let biased_prior = self.build_adaptive_prior(
                    &prior_llr,
                    &child.posterior,
                    adaptive_perturbation_llr_threshold,
                );

                // Apply perturbation on top of biased prior if continue_perturbation is set.
                let phase_prior = if self.config.continue_perturbation {
                    if self.config.sequential_mc
                        || self.config.perturbation_method == PerturbationMethod::Resample
                        || matches!(child.perturbation_state, PerturbationState::None)
                    {
                        child.perturbation_state = sample_perturbation_state(
                            biased_prior.len(),
                            self.config.init_perturbation_mode,
                            self.config.sigma2,
                            self.config.delta,
                            &mut rng,
                        );
                    }
                    apply_perturbation_state(&biased_prior, &child.perturbation_state)
                } else {
                    biased_prior
                };

                // (b) Constant mem-bp leg with init_gamma.
                let constant_gamma =
                    Array1::from_elem(prior_llr.len(), self.config.init_gamma);

                let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                    detectors,
                    &phase_prior,
                    &child.posterior,
                    &constant_gamma,
                    self.config.biased_relay_t_0,
                    self.config.reset_marginal,
                    self.config.drop_p > 0.0,
                    self.next_drop_seed(&mut rng),
                );

                child_iterations += iters;
                generation_max_iters = generation_max_iters.max(iters);

                if success {
                    let solution_llr_cost = self.estimated_error_weight(&decoding, &prior_llr);
                    let replace = match generation_success {
                        None => true,
                        Some((best_iter, best_llr_cost, _, _)) => {
                            child_iterations < best_iter
                                || (child_iterations == best_iter
                                    && solution_llr_cost < best_llr_cost)
                        }
                    };
                    if replace {
                        generation_success = Some((
                            child_iterations,
                            solution_llr_cost,
                            decoding.clone(),
                            posterior.clone(),
                        ));
                    }

                    {
                        let const_fitness = fitness_from_final_marginal(
                            residual_weight(&self.get_detectors(decoding.view()), detectors),
                            &posterior,
                            self.config.fitness_alpha,
                            self.config.fitness_beta,
                            self.config.fitness_low_llr_mu,
                            self.config.fitness_low_llr_threshold,
                        );
                        if const_fitness > best_fitness {
                            best_fitness = const_fitness;
                            best_decoding = decoding.clone();
                            best_posterior = posterior.clone();
                        }
                    }

                    // On constant-leg convergence, skip relay legs for this member.
                    generation_iterations += child_iterations;
                    continue;
                }

                // (c) R_relay relay-bp legs carrying marginals forward.
                let mut current_posterior = posterior;
                let mut relay_converged = false;
                let mut last_relay_decoding = decoding;

                for _relay_leg in 0..self.config.biased_relay_r_relay {
                    let relay_gamma = self.sample_relay_gamma(prior_llr.len(), &mut rng);

                    let (relay_dec, relay_post, relay_iters, relay_success) =
                        self.run_mem_bp_phase(
                            detectors,
                            &phase_prior,
                            &current_posterior,
                            &relay_gamma,
                            self.config.biased_relay_r_relay_iter,
                            false, // relay legs always carry marginals
                            self.config.drop_p > 0.0,
                            self.next_drop_seed(&mut rng),
                        );

                    child_iterations += relay_iters;
                    generation_max_iters = generation_max_iters.max(relay_iters);
                    current_posterior = relay_post.clone();
                    last_relay_decoding = relay_dec.clone();

                    if relay_success {
                        let solution_llr_cost =
                            self.estimated_error_weight(&relay_dec, &prior_llr);
                        let replace = match generation_success {
                            None => true,
                            Some((best_iter, best_llr_cost, _, _)) => {
                                child_iterations < best_iter
                                    || (child_iterations == best_iter
                                        && solution_llr_cost < best_llr_cost)
                            }
                        };
                        if replace {
                            generation_success = Some((
                                child_iterations,
                                solution_llr_cost,
                                relay_dec.clone(),
                                relay_post.clone(),
                            ));
                        }
                        relay_converged = true;
                        break;
                    }
                }

                // Score member using final posterior (after all relay legs).
                let decoded_for_score = self.get_detectors(last_relay_decoding.view());
                let member_residual = residual_weight(&decoded_for_score, detectors);
                let member_fitness = fitness_from_final_marginal(
                    member_residual,
                    &current_posterior,
                    self.config.fitness_alpha,
                    self.config.fitness_beta,
                    self.config.fitness_low_llr_mu,
                    self.config.fitness_low_llr_threshold,
                );
                residual_weight_history.push(member_residual);
                gen_best = gen_best.max(member_fitness);

                if member_fitness > best_fitness {
                    best_fitness = member_fitness;
                    best_decoding = last_relay_decoding.clone();
                    best_posterior = current_posterior.clone();
                }

                gen_best_gamma_mean = self.config.init_gamma;

                generation_iterations += child_iterations;

                if relay_converged {
                    break;
                }

                next_members.push(PopulationMember {
                    posterior: current_posterior,
                    fitness: member_fitness,
                    perturbation_state: child.perturbation_state,
                    residual_adjacent_variable_indices: Vec::new(),
                });
            }

            gamma_history.push(gen_best_gamma_mean);

            if let Some((_, _, gen_dec, gen_post)) = generation_success {
                total_iterations += generation_iterations;
                generation_best_fitness.push(gen_best);
                return DecodeResult {
                    decoding: gen_dec.clone(),
                    decoded_detectors: self.get_detectors(gen_dec.view()),
                    posterior_ratios: gen_post.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(gen_dec.view()),
                    iterations: total_iterations,
                    max_iter,
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(gen_post),
                        residual_weight_history,
                        gamma_history,
                        detailed_dynamics: None,
                        score_spike_trace: None,
                    },
                };
            }

            total_iterations += generation_iterations;
            generation_best_fitness.push(gen_best);
            adaptive_perturbation_llr_threshold = self.next_adaptive_perturbation_llr_threshold(
                adaptive_perturbation_llr_threshold,
                generation_max_iters,
            );
            members = next_members;
        }

        // No convergence after all rounds.
        DecodeResult {
            decoding: best_decoding.clone(),
            decoded_detectors: self.get_detectors(best_decoding.view()),
            posterior_ratios: best_posterior.clone(),
            success: false,
            decoding_quality: self.get_decoding_quality(best_decoding.view()),
            iterations: total_iterations,
            max_iter,
            extra: BPExtraResult::SLGMBPTrace {
                phase1_converged: false,
                phase1_iterations,
                total_iterations,
                generation_count: gamma_history.len(),
                generation_best_fitness,
                selected_solution_posterior: Some(best_posterior),
                residual_weight_history,
                gamma_history,
                detailed_dynamics: None,
                score_spike_trace: None,
            },
        }
    }
}

impl Decoder for SLGMBPDecoder {
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.check_matrix.clone()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.base_min_sum_config.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let prior_llr = self.prior_llr();
        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // Phase 1: diversity initialization + configurable initial BP strategy.
        let init_population = initialize_llr_population(
            &prior_llr,
            self.config.init_perturbation_mode,
            self.config.ensemble_size,
            self.config.sigma2,
            self.config.delta,
            &mut rng,
        );

        let mut members = Vec::<PopulationMember>::with_capacity(init_population.len());
        let mut best_decoding = Array1::<Bit>::zeros(self.check_matrix.cols());
        let mut best_posterior = prior_llr.clone();
        let mut best_fitness = f64::NEG_INFINITY;
        let mut generation_best_fitness = Vec::<f64>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut phase1_iterations = 0usize;
        let mut adaptive_perturbation_llr_threshold =
            self.initial_adaptive_perturbation_llr_threshold();
        // Store (iters, llr_cost, decoding, posterior) for tie-breaking.
        // For equal iteration counts, prefer the solution with smaller sum(bit_i * prior_llr_i).
        let mut phase1_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;

        for (init_llr, perturbation_state) in &init_population {
            let (decoding, posterior, iters, success, fitness) = match self.config.init_strategy {
                InitStrategy::MinSum => {
                    let (decoding, posterior, iters, success, cumsum_abs) =
                        self.run_min_sum_phase(detectors, init_llr, None);
                    let residual = residual_weight(&self.get_detectors(decoding.view()), detectors);
                    let fitness = fitness_from_ms_cumsum(
                        residual,
                        cumsum_abs,
                        &posterior,
                        self.config.fitness_alpha,
                        self.config.fitness_beta,
                        self.config.fitness_low_llr_mu,
                        self.config.fitness_low_llr_threshold,
                    );
                    (decoding, posterior, iters, success, fitness)
                }
                InitStrategy::MemBp => {
                    let gamma_vec = Array1::from_elem(prior_llr.len(), self.config.init_gamma);
                    let phase1_t = if self.config.biased_relay_mode {
                        self.config.biased_relay_t_0
                    } else {
                        self.config.t_ms
                    };
                    let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                        detectors,
                        &prior_llr,
                        init_llr,
                        &gamma_vec,
                        phase1_t,
                        false,
                        false,
                        None,
                    );
                    let residual = residual_weight(&self.get_detectors(decoding.view()), detectors);
                    let fitness = fitness_from_final_marginal(
                        residual,
                        &posterior,
                        self.config.fitness_alpha,
                        self.config.fitness_beta,
                        self.config.fitness_low_llr_mu,
                        self.config.fitness_low_llr_threshold,
                    );
                    (decoding, posterior, iters, success, fitness)
                }
            };
            // Ensemble members are treated as parallel within this layer.
            phase1_iterations = phase1_iterations.max(iters);

            if fitness > best_fitness {
                best_fitness = fitness;
                best_decoding = decoding.clone();
                best_posterior = posterior.clone();
            }

            if success {
                let solution_llr_cost = decoding
                    .iter()
                    .zip(prior_llr.iter())
                    .map(|(&bit, &llr)| (bit as f64) * llr)
                    .sum::<f64>();
                let replace = match phase1_success {
                    None => true,
                    Some((best_iter, best_llr_cost, _, _)) => {
                        iters < best_iter
                            || (iters == best_iter && solution_llr_cost < best_llr_cost)
                    }
                };
                if replace {
                    phase1_success = Some((
                        iters,
                        solution_llr_cost,
                        decoding.clone(),
                        posterior.clone(),
                    ));
                }
                // If a member converged in one iteration we already reached the minimum.
                if iters == 1 {
                    break;
                }
                continue;
            }

            members.push(PopulationMember {
                posterior,
                fitness,
                perturbation_state: perturbation_state.clone(),
                residual_adjacent_variable_indices: Vec::new(),
            });
        }

        if let Some((phase1_success_iters, _, phase1_decoding, phase1_posterior)) = phase1_success {
            return DecodeResult {
                decoding: phase1_decoding.clone(),
                decoded_detectors: self.get_detectors(phase1_decoding.view()),
                posterior_ratios: phase1_posterior,
                success: true,
                decoding_quality: self.get_decoding_quality(phase1_decoding.view()),
                iterations: phase1_success_iters,
                max_iter: self.max_iter(),
                extra: BPExtraResult::SLGMBPTrace {
                    phase1_converged: true,
                    phase1_iterations: phase1_success_iters,
                    total_iterations: phase1_success_iters,
                    generation_count: 0,
                    generation_best_fitness,
                    selected_solution_posterior: None,
                    residual_weight_history: Vec::new(),
                    gamma_history,
                    detailed_dynamics: None,
                    score_spike_trace: None,
                },
            };
        }

        // Branch: biased_relay mode uses a completely different generational loop.
        if self.config.biased_relay_mode {
            return self.decode_detailed_biased_relay(
                detectors,
                prior_llr,
                members,
                best_decoding,
                best_posterior,
                best_fitness,
                phase1_iterations,
                adaptive_perturbation_llr_threshold,
                rng,
            );
        }

        let mut total_iterations = phase1_iterations;

        // Phases 2-5: GA + Mem-BP generational search.
        let mut residual_weight_history = Vec::<usize>::new();

        for _gen in 0..self.config.g_max {
            let children = build_next_generation(
                &members,
                self.config.elite_count,
                self.config.sequential_mc,
                self.config.mutation_rate,
                self.config.mutation_llr_abs_threshold,
                self.config.selection_mode,
                self.config.weighted_selection_mode,
                self.config.perturbation_method,
                self.config.tournament_size,
                &mut rng,
            );

            let mut next_members = Vec::<PopulationMember>::with_capacity(children.len());
            let mut gen_best = f64::NEG_INFINITY;
            let mut generation_iterations = 0usize;
            let mut generation_max_iters = 0usize;
            // Store (iters, llr_cost, decoding, posterior) for tie-breaking.
            let mut generation_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;
            let mut gen_best_member_fitness = f64::NEG_INFINITY;
            let mut gen_best_gamma: Option<Array1<f64>> = None;

            for mut child in children {
                let phase_prior = self.build_generation_phase_prior(
                    &prior_llr,
                    &mut child,
                    &mut rng,
                    adaptive_perturbation_llr_threshold,
                );

                let member_gamma = self.sample_member_memory_strength(
                    &child.posterior,
                    &child.residual_adjacent_variable_indices,
                    &mut rng,
                );

                let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                    detectors,
                    &phase_prior,
                    &child.posterior,
                    &member_gamma,
                    self.config.t_mem,
                    self.config.reset_marginal,
                    self.config.drop_p > 0.0,
                    self.next_drop_seed(&mut rng),
                );

                let decoded_detectors = self.get_detectors(decoding.view());
                let residual = residual_weight(&decoded_detectors, detectors);
                residual_weight_history.push(residual);
                let fitness = fitness_from_final_marginal(
                    residual,
                    &posterior,
                    self.config.fitness_alpha,
                    self.config.fitness_beta,
                    self.config.fitness_low_llr_mu,
                    self.config.fitness_low_llr_threshold,
                );
                gen_best = gen_best.max(fitness);

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_decoding = decoding.clone();
                    best_posterior = posterior.clone();
                }

                if fitness > gen_best_member_fitness {
                    gen_best_member_fitness = fitness;
                    gen_best_gamma = Some(member_gamma.clone());
                }

                generation_max_iters = generation_max_iters.max(iters);

                if success {
                    let solution_llr_cost = decoding
                        .iter()
                        .zip(prior_llr.iter())
                        .map(|(&bit, &llr)| (bit as f64) * llr)
                        .sum::<f64>();
                    let replace = match generation_success {
                        None => true,
                        Some((best_iter, best_llr_cost, _, _)) => {
                            iters < best_iter
                                || (iters == best_iter && solution_llr_cost < best_llr_cost)
                        }
                    };
                    if replace {
                        generation_success = Some((
                            iters,
                            solution_llr_cost,
                            decoding.clone(),
                            posterior.clone(),
                        ));
                    }
                    // Minimum possible iteration for this layer is 1.
                    if iters == 1 {
                        break;
                    }
                    continue;
                }

                // No success yet in this generation: account the layer by max iters.
                generation_iterations = generation_iterations.max(iters);
                next_members.push(PopulationMember {
                    posterior,
                    fitness,
                    perturbation_state: child.perturbation_state,
                    residual_adjacent_variable_indices: child.residual_adjacent_variable_indices,
                });
            }

            if let Some(best_gamma) = &gen_best_gamma {
                let gamma_mean = if best_gamma.is_empty() {
                    0.0
                } else {
                    best_gamma.sum() / (best_gamma.len() as f64)
                };
                gamma_history.push(gamma_mean);
            } else {
                gamma_history.push(0.0);
            }

            if let Some((gen_success_iters, _, gen_success_decoding, gen_success_posterior)) = generation_success {
                total_iterations += gen_success_iters;
                let gen_decoded_detectors = self.get_detectors(gen_success_decoding.view());
                return DecodeResult {
                    decoding: gen_success_decoding.clone(),
                    decoded_detectors: gen_decoded_detectors,
                    posterior_ratios: gen_success_posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(gen_success_decoding.view()),
                    iterations: total_iterations,
                    max_iter: self.max_iter(),
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(gen_success_posterior),
                        residual_weight_history,
                        gamma_history,
                        detailed_dynamics: None,
                        score_spike_trace: None,
                    },
                };
            }

            total_iterations += generation_iterations;
            generation_best_fitness.push(gen_best);
            adaptive_perturbation_llr_threshold = self.next_adaptive_perturbation_llr_threshold(
                adaptive_perturbation_llr_threshold,
                generation_max_iters,
            );
            members = next_members;
        }

        DecodeResult {
            decoding: best_decoding.clone(),
            decoded_detectors: self.get_detectors(best_decoding.view()),
            posterior_ratios: best_posterior.clone(),
            success: false,
            decoding_quality: self.get_decoding_quality(best_decoding.view()),
            iterations: total_iterations,
            max_iter: self.max_iter(),
            extra: BPExtraResult::SLGMBPTrace {
                phase1_converged: false,
                phase1_iterations,
                total_iterations,
                generation_count: gamma_history.len(),
                generation_best_fitness,
                selected_solution_posterior: Some(best_posterior),
                residual_weight_history,
                gamma_history,
                detailed_dynamics: None,
                score_spike_trace: None,
            },
        }
    }

    fn decode_detailed_dynamics(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let prior_llr = self.prior_llr();
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let check_matrix_csr = self.check_matrix.to_csr();
        let observed_syndrome_weight = detectors.iter().filter(|&&b| b == 1).count();

        let init_population = initialize_llr_population(
            &prior_llr,
            self.config.init_perturbation_mode,
            self.config.ensemble_size,
            self.config.sigma2,
            self.config.delta,
            &mut rng,
        );

        let mut members = Vec::<PopulationMember>::with_capacity(init_population.len());
        let mut best_decoding = Array1::<Bit>::zeros(self.check_matrix.cols());
        let mut best_posterior = prior_llr.clone();
        let mut best_fitness = f64::NEG_INFINITY;
        let mut generation_best_fitness = Vec::<f64>::new();
        let mut generation_best_posteriors = Vec::<Array1<f64>>::new();
        let mut generation_best_adjacent_variable_indices = Vec::<Vec<usize>>::new();
        let mut generation_memory_strengths = Vec::<Array1<f64>>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut phase1_iterations = 0usize;
        let mut adaptive_perturbation_llr_threshold =
            self.initial_adaptive_perturbation_llr_threshold();
        let mut phase1_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;

        let mut dynamics_entries = Vec::<SLGMBPDynamicsEntry>::new();

        for (member_index, (init_llr, perturbation_state)) in init_population.iter().enumerate() {
            let (decoding, posterior, iters, success, fitness) = match self.config.init_strategy {
                InitStrategy::MinSum => {
                    let (decoding, posterior, iters, success, cumsum_abs) =
                        self.run_min_sum_phase(detectors, init_llr, None);
                    let residual = residual_weight(&self.get_detectors(decoding.view()), detectors);
                    let fitness = fitness_from_ms_cumsum(
                        residual,
                        cumsum_abs,
                        &posterior,
                        self.config.fitness_alpha,
                        self.config.fitness_beta,
                        self.config.fitness_low_llr_mu,
                        self.config.fitness_low_llr_threshold,
                    );
                    (decoding, posterior, iters, success, fitness)
                }
                InitStrategy::MemBp => {
                    let gamma_vec = Array1::from_elem(prior_llr.len(), self.config.init_gamma);
                    let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                        detectors,
                        &prior_llr,
                        init_llr,
                        &gamma_vec,
                        self.config.t_ms,
                        false,
                        false,
                        None,
                    );
                    let residual = residual_weight(&self.get_detectors(decoding.view()), detectors);
                    let fitness = fitness_from_final_marginal(
                        residual,
                        &posterior,
                        self.config.fitness_alpha,
                        self.config.fitness_beta,
                        self.config.fitness_low_llr_mu,
                        self.config.fitness_low_llr_threshold,
                    );
                    (decoding, posterior, iters, success, fitness)
                }
            };

            let decoded_detectors = self.get_detectors(decoding.view());
            let residual = residual_weight(&decoded_detectors, detectors);
            let residual_checks = self.residual_check_indices(&decoded_detectors, detectors);
            let (adjacent_indices, adjacent_llrs) =
                self.residual_adjacent_llrs(&check_matrix_csr, &residual_checks, &posterior);
            dynamics_entries.push(SLGMBPDynamicsEntry {
                stage: "phase1".to_string(),
                generation_index: 0,
                member_index,
                residual_syndrome_count: residual,
                residual_adjacent_variable_count: adjacent_indices.len(),
                score: fitness,
                residual_adjacent_variable_indices: adjacent_indices.clone(),
                residual_adjacent_variable_llrs: adjacent_llrs,
                converged: success,
                iteration_count: iters,
                estimated_error_weight: self.estimated_error_weight(&decoding, &prior_llr),
            });

            phase1_iterations = phase1_iterations.max(iters);

            if fitness > best_fitness {
                best_fitness = fitness;
                best_decoding = decoding.clone();
                best_posterior = posterior.clone();
            }

            if success {
                let solution_llr_cost = self.estimated_error_weight(&decoding, &prior_llr);
                let replace = match phase1_success {
                    None => true,
                    Some((best_iter, best_llr_cost, _, _)) => {
                        iters < best_iter || (iters == best_iter && solution_llr_cost < best_llr_cost)
                    }
                };
                if replace {
                    phase1_success = Some((
                        iters,
                        solution_llr_cost,
                        decoding.clone(),
                        posterior.clone(),
                    ));
                }
                if iters == 1 {
                    break;
                }
                continue;
            }

            members.push(PopulationMember {
                posterior,
                fitness,
                perturbation_state: perturbation_state.clone(),
                residual_adjacent_variable_indices: adjacent_indices,
            });
        }

        if let Some((phase1_success_iters, _, phase1_decoding, phase1_posterior)) = phase1_success {
            return DecodeResult {
                decoding: phase1_decoding.clone(),
                decoded_detectors: self.get_detectors(phase1_decoding.view()),
                posterior_ratios: phase1_posterior,
                success: true,
                decoding_quality: self.get_decoding_quality(phase1_decoding.view()),
                iterations: phase1_success_iters,
                max_iter: self.max_iter(),
                extra: BPExtraResult::SLGMBPTrace {
                    phase1_converged: true,
                    phase1_iterations: phase1_success_iters,
                    total_iterations: phase1_success_iters,
                    generation_count: 0,
                    generation_best_fitness,
                    selected_solution_posterior: None,
                    residual_weight_history: Vec::new(),
                    gamma_history,
                    detailed_dynamics: Some(SLGMBPDetailedDynamicsTrace {
                        observed_syndrome_weight,
                        entries: dynamics_entries,
                    }),
                    score_spike_trace: None,
                },
            };
        }

        let mut total_iterations = phase1_iterations;
        let mut residual_weight_history = Vec::<usize>::new();

        for gen_idx in 0..self.config.g_max {
            let children = build_next_generation(
                &members,
                self.config.elite_count,
                self.config.sequential_mc,
                self.config.mutation_rate,
                self.config.mutation_llr_abs_threshold,
                self.config.selection_mode,
                self.config.weighted_selection_mode,
                self.config.perturbation_method,
                self.config.tournament_size,
                &mut rng,
            );

            let mut next_members = Vec::<PopulationMember>::with_capacity(children.len());
            let mut gen_best = f64::NEG_INFINITY;
            let mut generation_iterations = 0usize;
            let mut generation_max_iters = 0usize;
            let mut generation_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;
            let mut gen_best_member_fitness = f64::NEG_INFINITY;
            let mut gen_best_member_posterior: Option<Array1<f64>> = None;
            let mut gen_best_member_adjacent_indices = Vec::<usize>::new();
            let mut gen_best_member_gamma: Option<Array1<f64>> = None;

            for (member_index, mut child) in children.into_iter().enumerate() {
                let phase_prior = self.build_generation_phase_prior(
                    &prior_llr,
                    &mut child,
                    &mut rng,
                    adaptive_perturbation_llr_threshold,
                );

                let member_gamma = self.sample_member_memory_strength(
                    &child.posterior,
                    &child.residual_adjacent_variable_indices,
                    &mut rng,
                );

                let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                    detectors,
                    &phase_prior,
                    &child.posterior,
                    &member_gamma,
                    self.config.t_mem,
                    self.config.reset_marginal,
                    self.config.drop_p > 0.0,
                    self.next_drop_seed(&mut rng),
                );

                let decoded_detectors = self.get_detectors(decoding.view());
                let residual = residual_weight(&decoded_detectors, detectors);
                residual_weight_history.push(residual);
                let fitness = fitness_from_final_marginal(
                    residual,
                    &posterior,
                    self.config.fitness_alpha,
                    self.config.fitness_beta,
                    self.config.fitness_low_llr_mu,
                    self.config.fitness_low_llr_threshold,
                );
                gen_best = gen_best.max(fitness);

                let residual_checks = self.residual_check_indices(&decoded_detectors, detectors);
                let (adjacent_indices, adjacent_llrs) =
                    self.residual_adjacent_llrs(&check_matrix_csr, &residual_checks, &posterior);

                if fitness > gen_best_member_fitness {
                    gen_best_member_fitness = fitness;
                    gen_best_member_posterior = Some(posterior.clone());
                    gen_best_member_adjacent_indices = adjacent_indices.clone();
                    gen_best_member_gamma = Some(member_gamma.clone());
                }

                dynamics_entries.push(SLGMBPDynamicsEntry {
                    stage: "generation".to_string(),
                    generation_index: gen_idx,
                    member_index,
                    residual_syndrome_count: residual,
                    residual_adjacent_variable_count: adjacent_indices.len(),
                    score: fitness,
                    residual_adjacent_variable_indices: adjacent_indices.clone(),
                    residual_adjacent_variable_llrs: adjacent_llrs,
                    converged: success,
                    iteration_count: iters,
                    estimated_error_weight: self.estimated_error_weight(&decoding, &prior_llr),
                });

                generation_max_iters = generation_max_iters.max(iters);

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_decoding = decoding.clone();
                    best_posterior = posterior.clone();
                }

                if success {
                    let solution_llr_cost = self.estimated_error_weight(&decoding, &prior_llr);
                    let replace = match generation_success {
                        None => true,
                        Some((best_iter, best_llr_cost, _, _)) => {
                            iters < best_iter || (iters == best_iter && solution_llr_cost < best_llr_cost)
                        }
                    };
                    if replace {
                        generation_success = Some((
                            iters,
                            solution_llr_cost,
                            decoding.clone(),
                            posterior.clone(),
                        ));
                    }
                    if iters == 1 {
                        break;
                    }
                    continue;
                }

                generation_iterations = generation_iterations.max(iters);
                next_members.push(PopulationMember {
                    posterior,
                    fitness,
                    perturbation_state: child.perturbation_state,
                    residual_adjacent_variable_indices: adjacent_indices,
                });
            }

            if let Some(best_gamma) = &gen_best_member_gamma {
                let gamma_mean = if best_gamma.is_empty() {
                    0.0
                } else {
                    best_gamma.sum() / (best_gamma.len() as f64)
                };
                gamma_history.push(gamma_mean);
            } else {
                gamma_history.push(0.0);
            }

            if let Some((gen_success_iters, _, gen_success_decoding, gen_success_posterior)) = generation_success {
                generation_best_fitness.push(gen_best);
                generation_best_posteriors.push(
                    gen_best_member_posterior.unwrap_or_else(|| gen_success_posterior.clone()),
                );
                generation_best_adjacent_variable_indices.push(gen_best_member_adjacent_indices);
                generation_memory_strengths.push(
                    gen_best_member_gamma
                        .clone()
                        .unwrap_or_else(|| Array1::from_elem(prior_llr.len(), self.config.init_gamma)),
                );
                let score_spike_trace = self.build_score_spike_trace(
                    &generation_best_fitness,
                    &generation_best_posteriors,
                    &generation_best_adjacent_variable_indices,
                    &generation_memory_strengths,
                    &gamma_history,
                );
                total_iterations += gen_success_iters;
                let gen_decoded_detectors = self.get_detectors(gen_success_decoding.view());
                return DecodeResult {
                    decoding: gen_success_decoding.clone(),
                    decoded_detectors: gen_decoded_detectors,
                    posterior_ratios: gen_success_posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(gen_success_decoding.view()),
                    iterations: total_iterations,
                    max_iter: self.max_iter(),
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(gen_success_posterior),
                        residual_weight_history,
                        gamma_history,
                        detailed_dynamics: Some(SLGMBPDetailedDynamicsTrace {
                            observed_syndrome_weight,
                            entries: dynamics_entries,
                        }),
                        score_spike_trace,
                    },
                };
            }

            total_iterations += generation_iterations;
            generation_best_fitness.push(gen_best);
            generation_best_posteriors
                .push(gen_best_member_posterior.unwrap_or_else(|| best_posterior.clone()));
            generation_best_adjacent_variable_indices.push(gen_best_member_adjacent_indices);
            generation_memory_strengths.push(
                gen_best_member_gamma
                    .clone()
                    .unwrap_or_else(|| Array1::from_elem(prior_llr.len(), self.config.init_gamma)),
            );
            adaptive_perturbation_llr_threshold = self.next_adaptive_perturbation_llr_threshold(
                adaptive_perturbation_llr_threshold,
                generation_max_iters,
            );
            members = next_members;
        }

        let score_spike_trace = self.build_score_spike_trace(
            &generation_best_fitness,
            &generation_best_posteriors,
            &generation_best_adjacent_variable_indices,
            &generation_memory_strengths,
            &gamma_history,
        );

        DecodeResult {
            decoding: best_decoding.clone(),
            decoded_detectors: self.get_detectors(best_decoding.view()),
            posterior_ratios: best_posterior.clone(),
            success: false,
            decoding_quality: self.get_decoding_quality(best_decoding.view()),
            iterations: total_iterations,
            max_iter: self.max_iter(),
            extra: BPExtraResult::SLGMBPTrace {
                phase1_converged: false,
                phase1_iterations,
                total_iterations,
                generation_count: gamma_history.len(),
                generation_best_fitness,
                selected_solution_posterior: Some(best_posterior),
                residual_weight_history,
                gamma_history,
                detailed_dynamics: Some(SLGMBPDetailedDynamicsTrace {
                    observed_syndrome_weight,
                    entries: dynamics_entries,
                }),
                score_spike_trace,
            },
        }
    }
}

impl DecoderRunner for SLGMBPDecoder {}

#[cfg(test)]
mod tests {
    use super::SLGMBPDecoder;
    use crate::bp::min_sum::MinSumDecoderConfig;
    use crate::bp::slg_mbp::config::{
        AdaptivePerturbationSignMode, AdaptivePerturbationThresholdMode,
        SLGMBPDecoderConfig,
    };
    use crate::bipartite_graph::BipartiteGraph;
    use crate::decoder::SparseBitMatrix;
    use ndarray::array;
    use std::sync::Arc;

    fn make_decoder(config: SLGMBPDecoderConfig) -> SLGMBPDecoder {
        let check_matrix = SparseBitMatrix::from_dense(array![[1u8, 0u8], [0u8, 1u8]]);
        let min_sum_config = MinSumDecoderConfig {
            error_priors: array![0.2, 0.3],
            max_iter: 4,
            alpha: None,
            alpha_iteration_scaling_factor: 1.0,
            gamma0: None,
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
            enable_variable_message_drop: false,
            drop_probability: 0.0,
            drop_llr_threshold: 0.0,
            rng_seed: None,
        };
        SLGMBPDecoder::new(
            Arc::new(check_matrix),
            Arc::new(min_sum_config),
            Arc::new(config),
        )
    }

    #[test]
    fn adaptive_prior_rebiases_only_low_confidence_variables() {
        let config = SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            ..SLGMBPDecoderConfig::default()
        };
        let decoder = make_decoder(config);
        let base_prior = array![1.0, -2.0, 0.75];
        let posterior = array![0.1, 0.3, -0.249];

        let rebased = decoder.build_adaptive_prior(&base_prior, &posterior, 0.25);

        let add_val = 3.0_f64.ln();
        assert_eq!(rebased, array![1.0 - add_val, -2.0, 0.75 - add_val]);
    }

    #[test]
    fn adaptive_prior_random_sign_mode_respects_positive_probability_extremes() {
        let base_prior = array![1.0, -2.0, 0.75];
        let posterior = array![0.1, 0.3, -0.249];
        let add_val = 3.0_f64.ln();

        let positive_decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::Random,
            adaptive_perturbation_positive_sign_prob: 1.0,
            ..SLGMBPDecoderConfig::default()
        });
        let positive_rebased = positive_decoder.build_adaptive_prior(&base_prior, &posterior, 0.25);
        assert_eq!(positive_rebased, array![1.0 + add_val, -2.0, 0.75 + add_val]);

        let negative_decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::Random,
            adaptive_perturbation_positive_sign_prob: 0.0,
            ..SLGMBPDecoderConfig::default()
        });
        let negative_rebased = negative_decoder.build_adaptive_prior(&base_prior, &posterior, 0.25);
        assert_eq!(negative_rebased, array![1.0 - add_val, -2.0, 0.75 - add_val]);
    }

    #[test]
    fn dynamic_adaptive_threshold_starts_at_min_and_clips_to_max() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation_llr_threshold_mode:
                AdaptivePerturbationThresholdMode::GenerationLog10Iter,
            adaptive_perturbation_llr_threshold_min: 0.25,
            adaptive_perturbation_llr_threshold_max: 0.5,
            adaptive_perturbation_llr_threshold_factor: 0.15,
            ..SLGMBPDecoderConfig::default()
        });

        let initial = decoder.initial_adaptive_perturbation_llr_threshold();
        let after_ten = decoder.next_adaptive_perturbation_llr_threshold(initial, 10);
        let after_hundred = decoder.next_adaptive_perturbation_llr_threshold(after_ten, 100);

        assert!((initial - 0.25).abs() < 1e-12);
        assert!((after_ten - 0.4).abs() < 1e-12);
        assert!((after_hundred - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reset_marginal_uses_phase_prior_instead_of_carried_posterior() {
        let config = SLGMBPDecoderConfig {
            eta: 0.5,
            ..SLGMBPDecoderConfig::default()
        };
        let decoder = make_decoder(config);
        let phase_prior = array![1.5, -0.25];
        let child_posterior = array![10.0, -8.0];

        let reset = decoder.initial_marginal(&phase_prior, &child_posterior, true);
        let carried = decoder.initial_marginal(&phase_prior, &child_posterior, false);

        assert_eq!(reset, phase_prior);
        assert_eq!(carried, array![5.0, -4.0]);
    }
}
