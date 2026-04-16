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
    AdaptiveMemoryMode, AdaptivePerturbationBiasMode, AdaptivePerturbationFactorMode,
    AdaptivePerturbationPriorBaseMode, AdaptivePerturbationSignMode, AdaptivePerturbationTarget,
    AdaptivePerturbationVariableBaseMode,
    AdaptivePerturbationThresholdMode, FinalSolutionSelection, GammaMode, InitStrategy,
    PerturbationMethod, SLGMBPDecoderConfig, SolutionCollectionMode,
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
struct CollectedSolution {
    discovery_iteration: usize,
    llr_cost: f64,
    decoding: Array1<Bit>,
    posterior: Array1<f64>,
}

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

    fn initial_phase_prior_llr(&self, prior_llr: &Array1<f64>) -> Array1<f64> {
        if self.config.initial_prior_bias.abs() < f64::EPSILON {
            return prior_llr.clone();
        }
        prior_llr.mapv(|x| x - self.config.initial_prior_bias)
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
        initial_marginal_override: Option<&Array1<f64>>,
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

        let mut prev_marginal = initial_marginal_override
            .cloned()
            .unwrap_or_else(|| self.initial_marginal(prior_llr, child_llr, reset_marginal));

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
        let carry_damping = if reset_marginal {
            0.0
        } else {
            self.config.marginal_carry_damping_factor.clamp(0.0, 1.0)
        };

        if carry_damping <= 0.0 {
            prior_llr.clone()
        } else {
            let threshold = self.config.marginal_carry_llr_abs_threshold;
            let use_all_variables = (threshold + 1.0).abs() < f64::EPSILON;
            let threshold_abs = threshold.abs();

            Array1::from_iter(prior_llr.iter().zip(child_llr.iter()).map(|(&prior, &child)| {
                let eligible = use_all_variables || child.abs() >= threshold_abs;
                if !eligible {
                    prior
                } else {
                    let carried = self.config.eta * child;
                    (1.0 - carry_damping) * prior + carry_damping * carried
                }
            }))
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

    fn apply_adaptive_perturbation(
        &self,
        target_llr: &Array1<f64>,
        posterior: &Array1<f64>,
        fixed_target_mask: Option<&[bool]>,
        adaptive_perturbation_llr_threshold: f64,
    ) -> Array1<f64> {
        if !self.config.adaptive_perturbation {
            return target_llr.clone();
        }

        let threshold = adaptive_perturbation_llr_threshold.abs();
        let mut rng = rand::thread_rng();
        let interval_a = self.config.adaptive_perturbation_factor_interval.0;
        let interval_b = self.config.adaptive_perturbation_factor_interval.1;
        let interval_lo = interval_a.min(interval_b);
        let interval_hi = interval_a.max(interval_b);

        let sample_factor = |rng: &mut rand::rngs::ThreadRng| -> f64 {
            let sampled = match self.config.adaptive_perturbation_factor_mode {
                AdaptivePerturbationFactorMode::Fixed => self.config.adaptive_perturbation_factor,
                AdaptivePerturbationFactorMode::UniformPerVariable => {
                    if (interval_hi - interval_lo).abs() < f64::EPSILON {
                        interval_lo
                    } else {
                        rng.gen_range(interval_lo..=interval_hi)
                    }
                }
            };
            sampled.max(f64::EPSILON)
        };

        Array1::from_iter(target_llr.iter().enumerate().map(|(idx, &target)| {
            let should_bias = match fixed_target_mask {
                Some(mask) => mask.get(idx).copied().unwrap_or(false),
                None => posterior[idx].abs() < threshold,
            };
            if should_bias {
                let factor = sample_factor(&mut rng);
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
                match self.config.adaptive_perturbation_bias_mode {
                    AdaptivePerturbationBiasMode::Additive => {
                        let add_val = factor.ln();
                        target + (sign * add_val)
                    }
                    AdaptivePerturbationBiasMode::Scale => {
                        if sign >= 0.0 {
                            target / factor
                        } else {
                            target * factor
                        }
                    }
                }
            } else {
                target
            }
        }))
    }

    fn build_adaptive_memory_strength(
        &self,
        posterior: &Array1<f64>,
        fixed_target_mask: Option<&[bool]>,
        adaptive_perturbation_llr_threshold: f64,
        rng: &mut StdRng,
    ) -> Option<Array1<f64>> {
        if !self.config.adaptive_perturbation
            || self.config.adaptive_perturbation_target
                != AdaptivePerturbationTarget::MemoryStrength
        {
            return None;
        }

        let threshold = adaptive_perturbation_llr_threshold.abs();
        let gamma_abs = self.config.init_gamma.abs();
        let mut gamma = Array1::from_elem(posterior.len(), self.config.init_gamma);

        for idx in 0..posterior.len() {
            let llr_abs = posterior[idx].abs();
            let mask_selected = fixed_target_mask
                .and_then(|mask| mask.get(idx))
                .copied()
                .unwrap_or(false);
            let should_flip = if fixed_target_mask.is_some() {
                mask_selected
            } else {
                llr_abs < threshold
            };

            let should_reset_to_init =
                self.config.adaptive_perturbation_reset_on_threshold_exit && llr_abs >= threshold;

            if should_flip && !should_reset_to_init {
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
                gamma[idx] = sign * gamma_abs;
            } else {
                gamma[idx] = self.config.init_gamma;
            }
        }

        Some(gamma)
    }

    fn build_adaptive_prior(
        &self,
        base_prior_llr: &Array1<f64>,
        previous_adaptive_prior: Option<&Array1<f64>>,
        fixed_target_mask: Option<&[bool]>,
        posterior: &Array1<f64>,
        adaptive_perturbation_llr_threshold: f64,
    ) -> Array1<f64> {
        match self.config.adaptive_perturbation_target {
            AdaptivePerturbationTarget::Prior | AdaptivePerturbationTarget::Both => {
                let target_llr = match self.config.adaptive_perturbation_prior_base_mode {
                    AdaptivePerturbationPriorBaseMode::Initial => base_prior_llr,
                    AdaptivePerturbationPriorBaseMode::PreviousBiased => {
                        previous_adaptive_prior.unwrap_or(base_prior_llr)
                    }
                };

                let rebased = self.apply_adaptive_perturbation(
                    target_llr,
                    posterior,
                    fixed_target_mask,
                    adaptive_perturbation_llr_threshold,
                );

                if self.config.adaptive_perturbation_prior_base_mode
                    == AdaptivePerturbationPriorBaseMode::PreviousBiased
                    && self.config.adaptive_perturbation_reset_on_threshold_exit
                {
                    let threshold = adaptive_perturbation_llr_threshold.abs();
                    Array1::from_iter(
                        rebased
                            .iter()
                            .zip(base_prior_llr.iter())
                            .zip(posterior.iter())
                            .map(|((&biased, &base), &llr)| {
                                if llr.abs() < threshold {
                                    biased
                                } else {
                                    base
                                }
                            }),
                    )
                } else {
                    rebased
                }
            }
            AdaptivePerturbationTarget::Posterior
            | AdaptivePerturbationTarget::MemoryStrength => base_prior_llr.clone(),
        }
    }

    fn build_adaptive_initial_marginal(
        &self,
        phase_prior: &Array1<f64>,
        posterior: &Array1<f64>,
        fixed_target_mask: Option<&[bool]>,
        adaptive_perturbation_llr_threshold: f64,
        reset_marginal: bool,
    ) -> Array1<f64> {
        let initial_marginal = self.initial_marginal(phase_prior, posterior, reset_marginal);
        match self.config.adaptive_perturbation_target {
            AdaptivePerturbationTarget::Posterior | AdaptivePerturbationTarget::Both => self
                .apply_adaptive_perturbation(
                    &initial_marginal,
                    posterior,
                    fixed_target_mask,
                    adaptive_perturbation_llr_threshold,
                ),
            AdaptivePerturbationTarget::Prior | AdaptivePerturbationTarget::MemoryStrength => {
                initial_marginal
            }
        }
    }

    fn ensure_adaptive_target_mask(
        &self,
        child: &mut PopulationMember,
        adaptive_perturbation_llr_threshold: f64,
    ) {
        if self.config.adaptive_perturbation_variable_base_mode
            != AdaptivePerturbationVariableBaseMode::FirstLegPosteriorFixed
        {
            return;
        }

        if child.adaptive_target_mask.is_some() {
            return;
        }

        let threshold = adaptive_perturbation_llr_threshold.abs();
        child.adaptive_target_mask = Some(
            child
                .posterior
                .iter()
                .map(|&llr| llr.abs() < threshold)
                .collect(),
        );
    }

    fn build_generation_phase_prior(
        &self,
        base_prior_llr: &Array1<f64>,
        child: &mut PopulationMember,
        rng: &mut StdRng,
        adaptive_perturbation_llr_threshold: f64,
    ) -> Array1<f64> {
        self.ensure_adaptive_target_mask(child, adaptive_perturbation_llr_threshold);
        let adaptive_prior = self.build_adaptive_prior(
            base_prior_llr,
            child.adaptive_prior.as_ref(),
            child.adaptive_target_mask.as_deref(),
            &child.posterior,
            adaptive_perturbation_llr_threshold,
        );
        child.adaptive_prior = Some(adaptive_prior.clone());
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
                let sign_flip_prob = self.config.gamma_random_sign_flip_prob.clamp(0.0, 1.0);
                if (hi - lo).abs() < f64::EPSILON {
                    Array1::from_shape_simple_fn(n_variables, || {
                        let sampled = lo;
                        if rng.gen_bool(sign_flip_prob) {
                            -sampled
                        } else {
                            sampled
                        }
                    })
                } else {
                    Array1::from_shape_simple_fn(n_variables, || {
                        let sampled = rng.gen_range(lo..=hi);
                        if rng.gen_bool(sign_flip_prob) {
                            -sampled
                        } else {
                            sampled
                        }
                    })
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

    fn sample_gamma_from_interval(
        &self,
        interval: (f64, f64),
        n_variables: usize,
        rng: &mut StdRng,
    ) -> Array1<f64> {
        let (a, b) = interval;
        let lo = a.min(b);
        let hi = a.max(b);
        let sign_flip_prob = self.config.gamma_random_sign_flip_prob.clamp(0.0, 1.0);
        if (hi - lo).abs() < f64::EPSILON {
            Array1::from_shape_simple_fn(n_variables, || {
                let sampled = lo;
                if rng.gen_bool(sign_flip_prob) {
                    -sampled
                } else {
                    sampled
                }
            })
        } else {
            Array1::from_shape_simple_fn(n_variables, || {
                let sampled = rng.gen_range(lo..=hi);
                if rng.gen_bool(sign_flip_prob) {
                    -sampled
                } else {
                    sampled
                }
            })
        }
    }

    fn sample_relay_gamma(&self, n_variables: usize, rng: &mut StdRng) -> Array1<f64> {
        self.sample_gamma_from_interval(self.config.relay_gamma_interval, n_variables, rng)
    }

    fn switch_relay_enabled(&self) -> bool {
        !self.config.biased_relay_mode && self.config.switch_relay_leg.is_some()
    }

    fn should_use_switched_relay_leg(&self, relay_leg_idx: usize) -> bool {
        if !self.switch_relay_enabled() {
            return false;
        }
        match self.config.switch_relay_leg {
            Some(switch_leg_1based) => relay_leg_idx + 1 >= switch_leg_1based,
            None => false,
        }
    }

    fn should_use_switched_relay_generation(&self, generation_idx: usize) -> bool {
        self.should_use_switched_relay_leg(generation_idx)
    }

    fn estimated_error_weight(&self, decoding: &Array1<Bit>, prior_llr: &Array1<f64>) -> f64 {
        decoding
            .iter()
            .zip(prior_llr.iter())
            .map(|(&bit, &llr)| (bit as f64) * llr)
            .sum::<f64>()
    }

    fn target_solution_count(&self) -> usize {
        self.config.n_solutions.max(1)
    }

    fn use_post_selection(&self) -> bool {
        self.config.solution_collection_mode == SolutionCollectionMode::PostSelection
    }

    fn should_stop_after_success_count(&self, count: usize) -> bool {
        if self.use_post_selection() {
            count >= self.target_solution_count()
        } else {
            count >= 1
        }
    }

    fn should_early_break_on_single_iter_success(&self) -> bool {
        !self.use_post_selection()
    }

    fn collect_solution(
        &self,
        out: &mut Vec<CollectedSolution>,
        discovery_iteration: usize,
        decoding: &Array1<Bit>,
        posterior: &Array1<f64>,
        prior_llr: &Array1<f64>,
    ) {
        out.push(CollectedSolution {
            discovery_iteration,
            llr_cost: self.estimated_error_weight(decoding, prior_llr),
            decoding: decoding.clone(),
            posterior: posterior.clone(),
        });
    }

    fn choose_final_solution(&self, collected: &[CollectedSolution]) -> Option<CollectedSolution> {
        if collected.is_empty() {
            return None;
        }

        let mut accepted = collected.to_vec();
        accepted.sort_by(|a, b| {
            a.discovery_iteration
                .cmp(&b.discovery_iteration)
                .then_with(|| a.llr_cost.total_cmp(&b.llr_cost))
        });
        if self.use_post_selection() {
            accepted.truncate(self.target_solution_count());
        } else {
            accepted.truncate(1);
        }

        let selected = match self.config.final_solution_selection {
            FinalSolutionSelection::Fastest => accepted
                .iter()
                .min_by(|a, b| {
                    a.discovery_iteration
                        .cmp(&b.discovery_iteration)
                        .then_with(|| a.llr_cost.total_cmp(&b.llr_cost))
                })
                .cloned(),
            FinalSolutionSelection::MinWeight => accepted
                .iter()
                .min_by(|a, b| {
                    a.llr_cost
                        .total_cmp(&b.llr_cost)
                        .then_with(|| a.discovery_iteration.cmp(&b.discovery_iteration))
                })
                .cloned(),
        };
        selected
    }

    fn accepted_solutions(&self, collected: &[CollectedSolution]) -> Vec<CollectedSolution> {
        if collected.is_empty() {
            return Vec::new();
        }
        let mut out = collected.to_vec();
        out.sort_by(|a, b| {
            a.discovery_iteration
                .cmp(&b.discovery_iteration)
                .then_with(|| a.llr_cost.total_cmp(&b.llr_cost))
        });
        if self.use_post_selection() {
            out.truncate(self.target_solution_count());
        } else {
            out.truncate(1);
        }
        out
    }

    fn accepted_solution_trace_vectors(
        &self,
        collected: &[CollectedSolution],
    ) -> (Vec<Array1<Bit>>, Vec<f64>, Vec<usize>) {
        let accepted = self.accepted_solutions(collected);
        let decodings = accepted.iter().map(|s| s.decoding.clone()).collect::<Vec<_>>();
        let weights = accepted.iter().map(|s| s.llr_cost).collect::<Vec<_>>();
        let iters = accepted
            .iter()
            .map(|s| s.discovery_iteration)
            .collect::<Vec<_>>();
        (decodings, weights, iters)
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
    /// gamma resampling. Each round consists of:
    ///   (a) bias + constant mem-bp leg
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
        mut collected_solutions: Vec<CollectedSolution>,
        mut rng: StdRng,
    ) -> DecodeResult {
        let max_iter = self.max_iter_biased_relay();
        let mut total_iterations = phase1_iterations;
        let mut generation_best_fitness = Vec::<f64>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut residual_weight_history = Vec::<usize>::new();
        let mut adaptive_perturbation_llr_threshold = initial_adaptive_threshold;

        // Round 1: run bias + constant mem-bp, then R_relay relay legs.
        {
            let mut round1_successes = Vec::<CollectedSolution>::new();
            let mut round1_iterations = 0usize;
            let mut round1_best = f64::NEG_INFINITY;

            for member in members.iter_mut() {
                self.ensure_adaptive_target_mask(member, adaptive_perturbation_llr_threshold);
                let mut member_iterations = 0usize;

                // (a) Build biased prior from the base prior + member posterior.
                let biased_prior = self.build_adaptive_prior(
                    &prior_llr,
                    member.adaptive_prior.as_ref(),
                    member.adaptive_target_mask.as_deref(),
                    &member.posterior,
                    adaptive_perturbation_llr_threshold,
                );
                member.adaptive_prior = Some(biased_prior.clone());

                // Apply perturbation on top of biased prior if continue_perturbation is set.
                let phase_prior = if self.config.continue_perturbation {
                    if self.config.sequential_mc
                        || self.config.perturbation_method == PerturbationMethod::Resample
                        || matches!(member.perturbation_state, PerturbationState::None)
                    {
                        member.perturbation_state = sample_perturbation_state(
                            biased_prior.len(),
                            self.config.init_perturbation_mode,
                            self.config.sigma2,
                            self.config.delta,
                            &mut rng,
                        );
                    }
                    apply_perturbation_state(&biased_prior, &member.perturbation_state)
                } else {
                    biased_prior
                };

                // (b) Constant mem-bp leg with init_gamma.
                let constant_gamma = Array1::from_elem(prior_llr.len(), self.config.init_gamma);
                let initial_marginal = self.build_adaptive_initial_marginal(
                    &phase_prior,
                    &member.posterior,
                    member.adaptive_target_mask.as_deref(),
                    adaptive_perturbation_llr_threshold,
                    self.config.reset_marginal,
                );

                let (const_decoding, const_posterior, const_iters, const_success) =
                    self.run_mem_bp_phase(
                        detectors,
                        &phase_prior,
                        &member.posterior,
                        Some(&initial_marginal),
                        &constant_gamma,
                        self.config.biased_relay_t_0,
                        self.config.reset_marginal,
                        self.config.drop_p > 0.0,
                        self.next_drop_seed(&mut rng),
                    );

                member_iterations += const_iters;

                if const_success {
                    self.collect_solution(
                        &mut round1_successes,
                        total_iterations + member_iterations,
                        &const_decoding,
                        &const_posterior,
                        &prior_llr,
                    );

                    let const_residual = residual_weight(
                        &self.get_detectors(const_decoding.view()),
                        detectors,
                    );
                    let const_fitness = fitness_from_final_marginal(
                        const_residual,
                        &const_posterior,
                        self.config.fitness_alpha,
                        self.config.fitness_beta,
                        self.config.fitness_low_llr_mu,
                        self.config.fitness_low_llr_threshold,
                    );
                    residual_weight_history.push(const_residual);
                    round1_best = round1_best.max(const_fitness);

                    if const_fitness > best_fitness {
                        best_fitness = const_fitness;
                        best_decoding = const_decoding.clone();
                        best_posterior = const_posterior.clone();
                    }

                    member.posterior = const_posterior;
                    member.fitness = const_fitness;
                    round1_iterations += member_iterations;
                    continue;
                }

                let mut current_posterior = const_posterior;
                let mut relay_converged = false;
                let mut last_decoding = const_decoding;

                for relay_leg_idx in 0..self.config.biased_relay_r_relay {
                    let switched_to_relay = self.should_use_switched_relay_leg(relay_leg_idx);
                    let adaptive_reset_each_leg = self.switch_relay_enabled();
                    let relay_phase_prior = if switched_to_relay {
                        &prior_llr
                    } else {
                        &phase_prior
                    };
                    let relay_initial_marginal = if switched_to_relay {
                        // Keep the immediately previous leg marginal when switching to relay mode.
                        current_posterior.clone()
                    } else {
                        // Adaptive-perturbation legs reset marginal each leg.
                        self.build_adaptive_initial_marginal(
                            &phase_prior,
                            &current_posterior,
                            member.adaptive_target_mask.as_deref(),
                            adaptive_perturbation_llr_threshold,
                            adaptive_reset_each_leg,
                        )
                    };
                    let relay_gamma = if switched_to_relay {
                        self.sample_relay_gamma(prior_llr.len(), &mut rng)
                    } else {
                        self.sample_gamma_from_interval(
                            self.config.gamma_interval,
                            prior_llr.len(),
                            &mut rng,
                        )
                    };

                    let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                        detectors,
                        relay_phase_prior,
                        &current_posterior,
                        Some(&relay_initial_marginal),
                        &relay_gamma,
                        self.config.biased_relay_r_relay_iter,
                        adaptive_reset_each_leg && !switched_to_relay,
                        self.config.drop_p > 0.0,
                        self.next_drop_seed(&mut rng),
                    );

                    member_iterations += iters;
                    current_posterior = posterior.clone();
                    last_decoding = decoding.clone();

                    if success {
                        self.collect_solution(
                            &mut round1_successes,
                            total_iterations + member_iterations,
                            &decoding,
                            &posterior,
                            &prior_llr,
                        );
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
                round1_best = round1_best.max(fitness);

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_decoding = last_decoding.clone();
                    best_posterior = current_posterior.clone();
                }

                member.posterior = current_posterior;
                member.fitness = fitness;

                round1_iterations += member_iterations;

                if relay_converged && self.should_early_break_on_single_iter_success() {
                    break;
                }
            }

            let round1_gamma_mean = self.config.init_gamma;
            gamma_history.push(round1_gamma_mean);

            total_iterations += round1_iterations;
            if !round1_successes.is_empty() {
                collected_solutions.extend(round1_successes);
            }
            if self.should_stop_after_success_count(collected_solutions.len()) {
                let accepted = self.accepted_solutions(&collected_solutions);
                let (accepted_decodings, accepted_weights, accepted_iters) =
                    self.accepted_solution_trace_vectors(&collected_solutions);
                let final_iters = accepted
                    .iter()
                    .map(|s| s.discovery_iteration)
                    .max()
                    .unwrap_or(total_iterations);
                let selected = self
                    .choose_final_solution(&collected_solutions)
                    .unwrap_or_else(|| accepted[0].clone());
                return DecodeResult {
                    decoding: selected.decoding.clone(),
                    decoded_detectors: self.get_detectors(selected.decoding.view()),
                    posterior_ratios: selected.posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(selected.decoding.view()),
                    iterations: final_iters,
                    max_iter,
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations: final_iters,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(selected.posterior.clone()),
                        accepted_solution_decodings: accepted_decodings,
                        accepted_solution_weights: accepted_weights,
                        accepted_solution_discovery_iterations: accepted_iters,
                        residual_weight_history,
                        gamma_history,
                        detailed_dynamics: None,
                        score_spike_trace: None,
                    },
                };
            }

            let gen_best = if round1_best.is_finite() {
                round1_best
            } else {
                members.iter().map(|m| m.fitness).fold(f64::NEG_INFINITY, f64::max)
            };
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
            let mut generation_successes = Vec::<CollectedSolution>::new();
            let mut gen_best_gamma_mean = 0.0f64;

            for mut child in children {
                let mut child_iterations = 0usize;
                self.ensure_adaptive_target_mask(&mut child, adaptive_perturbation_llr_threshold);
                // (a) Build biased prior from the base prior + child's posterior.
                let biased_prior = self.build_adaptive_prior(
                    &prior_llr,
                    child.adaptive_prior.as_ref(),
                    child.adaptive_target_mask.as_deref(),
                    &child.posterior,
                    adaptive_perturbation_llr_threshold,
                );
                child.adaptive_prior = Some(biased_prior.clone());

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
                let initial_marginal = self.build_adaptive_initial_marginal(
                    &phase_prior,
                    &child.posterior,
                    child.adaptive_target_mask.as_deref(),
                    adaptive_perturbation_llr_threshold,
                    self.config.reset_marginal,
                );

                let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                    detectors,
                    &phase_prior,
                    &child.posterior,
                    Some(&initial_marginal),
                    &constant_gamma,
                    self.config.biased_relay_t_0,
                    self.config.reset_marginal,
                    self.config.drop_p > 0.0,
                    self.next_drop_seed(&mut rng),
                );

                child_iterations += iters;
                generation_max_iters = generation_max_iters.max(iters);

                if success {
                    self.collect_solution(
                        &mut generation_successes,
                        total_iterations + child_iterations,
                        &decoding,
                        &posterior,
                        &prior_llr,
                    );

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

                for relay_leg_idx in 0..self.config.biased_relay_r_relay {
                    let switched_to_relay = self.should_use_switched_relay_leg(relay_leg_idx);
                    let adaptive_reset_each_leg = self.switch_relay_enabled();
                    let relay_phase_prior = if switched_to_relay {
                        &prior_llr
                    } else {
                        &phase_prior
                    };
                    let relay_initial_marginal = if switched_to_relay {
                        // Keep the immediately previous leg marginal when switching to relay mode.
                        current_posterior.clone()
                    } else {
                        // Adaptive-perturbation legs reset marginal each leg.
                        self.build_adaptive_initial_marginal(
                            &phase_prior,
                            &current_posterior,
                            child.adaptive_target_mask.as_deref(),
                            adaptive_perturbation_llr_threshold,
                            adaptive_reset_each_leg,
                        )
                    };
                    let relay_gamma = if switched_to_relay {
                        self.sample_relay_gamma(prior_llr.len(), &mut rng)
                    } else {
                        self.sample_gamma_from_interval(
                            self.config.gamma_interval,
                            prior_llr.len(),
                            &mut rng,
                        )
                    };

                    let (relay_dec, relay_post, relay_iters, relay_success) =
                        self.run_mem_bp_phase(
                            detectors,
                            relay_phase_prior,
                            &current_posterior,
                            Some(&relay_initial_marginal),
                            &relay_gamma,
                            self.config.biased_relay_r_relay_iter,
                            adaptive_reset_each_leg && !switched_to_relay,
                            self.config.drop_p > 0.0,
                            self.next_drop_seed(&mut rng),
                        );

                    child_iterations += relay_iters;
                    generation_max_iters = generation_max_iters.max(relay_iters);
                    current_posterior = relay_post.clone();
                    last_relay_decoding = relay_dec.clone();

                    if relay_success {
                        self.collect_solution(
                            &mut generation_successes,
                            total_iterations + child_iterations,
                            &relay_dec,
                            &relay_post,
                            &prior_llr,
                        );
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

                if relay_converged && self.should_early_break_on_single_iter_success() {
                    break;
                }

                next_members.push(PopulationMember {
                    posterior: current_posterior,
                    fitness: member_fitness,
                    perturbation_state: child.perturbation_state,
                    residual_adjacent_variable_indices: Vec::new(),
                    adaptive_prior: child.adaptive_prior,
                    adaptive_target_mask: child.adaptive_target_mask,
                });
            }

            gamma_history.push(gen_best_gamma_mean);

            total_iterations += generation_iterations;
            if !generation_successes.is_empty() {
                collected_solutions.extend(generation_successes);
            }
            if self.should_stop_after_success_count(collected_solutions.len()) {
                let accepted = self.accepted_solutions(&collected_solutions);
                let (accepted_decodings, accepted_weights, accepted_iters) =
                    self.accepted_solution_trace_vectors(&collected_solutions);
                let final_iters = accepted
                    .iter()
                    .map(|s| s.discovery_iteration)
                    .max()
                    .unwrap_or(total_iterations);
                let selected = self
                    .choose_final_solution(&collected_solutions)
                    .unwrap_or_else(|| accepted[0].clone());
                generation_best_fitness.push(gen_best);
                return DecodeResult {
                    decoding: selected.decoding.clone(),
                    decoded_detectors: self.get_detectors(selected.decoding.view()),
                    posterior_ratios: selected.posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(selected.decoding.view()),
                    iterations: final_iters,
                    max_iter,
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations: final_iters,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(selected.posterior.clone()),
                        accepted_solution_decodings: accepted_decodings,
                        accepted_solution_weights: accepted_weights,
                        accepted_solution_discovery_iterations: accepted_iters,
                        residual_weight_history,
                        gamma_history,
                        detailed_dynamics: None,
                        score_spike_trace: None,
                    },
                };
            }

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
                accepted_solution_decodings: Vec::new(),
                accepted_solution_weights: Vec::new(),
                accepted_solution_discovery_iterations: Vec::new(),
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
        let initial_phase_prior_llr = self.initial_phase_prior_llr(&prior_llr);
        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // Phase 1: diversity initialization + configurable initial BP strategy.
        let init_population = initialize_llr_population(
            &initial_phase_prior_llr,
            self.config.init_perturbation_mode,
            self.config.ensemble_size,
            self.config.sigma2,
            self.config.delta,
            &mut rng,
        );

        let mut members = Vec::<PopulationMember>::with_capacity(init_population.len());
        let mut best_decoding = Array1::<Bit>::zeros(self.check_matrix.cols());
        let mut best_posterior = initial_phase_prior_llr.clone();
        let mut best_fitness = f64::NEG_INFINITY;
        let mut generation_best_fitness = Vec::<f64>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut phase1_iterations = 0usize;
        let mut adaptive_perturbation_llr_threshold =
            self.initial_adaptive_perturbation_llr_threshold();
        let mut collected_solutions = Vec::<CollectedSolution>::new();

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
                        &initial_phase_prior_llr,
                        init_llr,
                        None,
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
                self.collect_solution(
                    &mut collected_solutions,
                    iters,
                    &decoding,
                    &posterior,
                    &initial_phase_prior_llr,
                );
                if self.should_stop_after_success_count(collected_solutions.len()) {
                    break;
                }
                // If a member converged in one iteration we already reached the minimum.
                if iters == 1 && self.should_early_break_on_single_iter_success() {
                    break;
                }
                continue;
            }

            members.push(PopulationMember {
                posterior,
                fitness,
                perturbation_state: perturbation_state.clone(),
                residual_adjacent_variable_indices: Vec::new(),
                adaptive_prior: None,
                adaptive_target_mask: None,
            });
        }

        if self.should_stop_after_success_count(collected_solutions.len()) {
            let accepted = self.accepted_solutions(&collected_solutions);
            let (accepted_decodings, accepted_weights, accepted_iters) =
                self.accepted_solution_trace_vectors(&collected_solutions);
            let phase1_success_iters = accepted
                .iter()
                .map(|s| s.discovery_iteration)
                .max()
                .unwrap_or(phase1_iterations);
            let selected = self
                .choose_final_solution(&collected_solutions)
                .unwrap_or_else(|| accepted[0].clone());
            return DecodeResult {
                decoding: selected.decoding.clone(),
                decoded_detectors: self.get_detectors(selected.decoding.view()),
                posterior_ratios: selected.posterior.clone(),
                success: true,
                decoding_quality: self.get_decoding_quality(selected.decoding.view()),
                iterations: phase1_success_iters,
                max_iter: self.max_iter(),
                extra: BPExtraResult::SLGMBPTrace {
                    phase1_converged: true,
                    phase1_iterations: phase1_success_iters,
                    total_iterations: phase1_success_iters,
                    generation_count: 0,
                    generation_best_fitness,
                    selected_solution_posterior: Some(selected.posterior.clone()),
                    accepted_solution_decodings: accepted_decodings,
                    accepted_solution_weights: accepted_weights,
                    accepted_solution_discovery_iterations: accepted_iters,
                    residual_weight_history: Vec::new(),
                    gamma_history,
                    detailed_dynamics: None,
                    score_spike_trace: None,
                },
            };
        }

        // Branch: biased relay flow is used only when biased_relay_mode is enabled.
        if self.config.biased_relay_mode {
            return self.decode_detailed_biased_relay(
                detectors,
                initial_phase_prior_llr,
                members,
                best_decoding,
                best_posterior,
                best_fitness,
                phase1_iterations,
                adaptive_perturbation_llr_threshold,
                collected_solutions,
                rng,
            );
        }

        let mut total_iterations = phase1_iterations;

        // Phases 2-5: GA + Mem-BP generational search.
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
            let mut generation_successes = Vec::<CollectedSolution>::new();
            let mut gen_best_member_fitness = f64::NEG_INFINITY;
            let mut gen_best_gamma: Option<Array1<f64>> = None;

            for mut child in children {
                let switched_to_relay = self.should_use_switched_relay_generation(gen_idx);

                let phase_prior = if switched_to_relay {
                    prior_llr.clone()
                } else {
                    self.build_generation_phase_prior(
                        &prior_llr,
                        &mut child,
                        &mut rng,
                        adaptive_perturbation_llr_threshold,
                    )
                };

                let member_gamma = if switched_to_relay {
                    self.sample_relay_gamma(prior_llr.len(), &mut rng)
                } else {
                    self.build_adaptive_memory_strength(
                        &child.posterior,
                        child.adaptive_target_mask.as_deref(),
                        adaptive_perturbation_llr_threshold,
                        &mut rng,
                    )
                    .unwrap_or_else(|| {
                        self.sample_member_memory_strength(
                            &child.posterior,
                            &child.residual_adjacent_variable_indices,
                            &mut rng,
                        )
                    })
                };

                let initial_marginal = if switched_to_relay {
                    child.posterior.clone()
                } else {
                    self.build_adaptive_initial_marginal(
                        &phase_prior,
                        &child.posterior,
                        child.adaptive_target_mask.as_deref(),
                        adaptive_perturbation_llr_threshold,
                        self.config.reset_marginal,
                    )
                };

                let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                    detectors,
                    &phase_prior,
                    &child.posterior,
                    Some(&initial_marginal),
                    &member_gamma,
                    self.config.t_mem,
                    if switched_to_relay {
                        false
                    } else {
                        self.config.reset_marginal
                    },
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
                    self.collect_solution(
                        &mut generation_successes,
                        total_iterations + iters,
                        &decoding,
                        &posterior,
                        &prior_llr,
                    );
                    // Minimum possible iteration for this layer is 1.
                    if iters == 1 && self.should_early_break_on_single_iter_success() {
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
                    adaptive_prior: child.adaptive_prior,
                    adaptive_target_mask: child.adaptive_target_mask,
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

            if !generation_successes.is_empty() {
                collected_solutions.extend(generation_successes.clone());
            }

            if self.should_stop_after_success_count(collected_solutions.len()) {
                let accepted = self.accepted_solutions(&collected_solutions);
                let (accepted_decodings, accepted_weights, accepted_iters) =
                    self.accepted_solution_trace_vectors(&collected_solutions);
                let final_iters = accepted
                    .iter()
                    .map(|s| s.discovery_iteration)
                    .max()
                    .unwrap_or(total_iterations);
                let selected = self
                    .choose_final_solution(&collected_solutions)
                    .unwrap_or_else(|| accepted[0].clone());
                let gen_decoded_detectors = self.get_detectors(selected.decoding.view());
                return DecodeResult {
                    decoding: selected.decoding.clone(),
                    decoded_detectors: gen_decoded_detectors,
                    posterior_ratios: selected.posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(selected.decoding.view()),
                    iterations: final_iters,
                    max_iter: self.max_iter(),
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations: final_iters,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(selected.posterior.clone()),
                        accepted_solution_decodings: accepted_decodings,
                        accepted_solution_weights: accepted_weights,
                        accepted_solution_discovery_iterations: accepted_iters,
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
                accepted_solution_decodings: Vec::new(),
                accepted_solution_weights: Vec::new(),
                accepted_solution_discovery_iterations: Vec::new(),
                residual_weight_history,
                gamma_history,
                detailed_dynamics: None,
                score_spike_trace: None,
            },
        }
    }

    fn decode_detailed_dynamics(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let prior_llr = self.prior_llr();
        let initial_phase_prior_llr = self.initial_phase_prior_llr(&prior_llr);
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let check_matrix_csr = self.check_matrix.to_csr();
        let observed_syndrome_weight = detectors.iter().filter(|&&b| b == 1).count();

        let init_population = initialize_llr_population(
            &initial_phase_prior_llr,
            self.config.init_perturbation_mode,
            self.config.ensemble_size,
            self.config.sigma2,
            self.config.delta,
            &mut rng,
        );

        let mut members = Vec::<PopulationMember>::with_capacity(init_population.len());
        let mut best_decoding = Array1::<Bit>::zeros(self.check_matrix.cols());
        let mut best_posterior = initial_phase_prior_llr.clone();
        let mut best_fitness = f64::NEG_INFINITY;
        let mut generation_best_fitness = Vec::<f64>::new();
        let mut generation_best_posteriors = Vec::<Array1<f64>>::new();
        let mut generation_best_adjacent_variable_indices = Vec::<Vec<usize>>::new();
        let mut generation_memory_strengths = Vec::<Array1<f64>>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut phase1_iterations = 0usize;
        let mut adaptive_perturbation_llr_threshold =
            self.initial_adaptive_perturbation_llr_threshold();
        let mut collected_solutions = Vec::<CollectedSolution>::new();

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
                        &initial_phase_prior_llr,
                        init_llr,
                        None,
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
                self.collect_solution(
                    &mut collected_solutions,
                    iters,
                    &decoding,
                    &posterior,
                    &initial_phase_prior_llr,
                );
                if self.should_stop_after_success_count(collected_solutions.len()) {
                    break;
                }
                if iters == 1 && self.should_early_break_on_single_iter_success() {
                    break;
                }
                continue;
            }

            members.push(PopulationMember {
                posterior,
                fitness,
                perturbation_state: perturbation_state.clone(),
                residual_adjacent_variable_indices: adjacent_indices,
                adaptive_prior: None,
                adaptive_target_mask: None,
            });
        }

        if self.should_stop_after_success_count(collected_solutions.len()) {
            let accepted = self.accepted_solutions(&collected_solutions);
            let (accepted_decodings, accepted_weights, accepted_iters) =
                self.accepted_solution_trace_vectors(&collected_solutions);
            let phase1_success_iters = accepted
                .iter()
                .map(|s| s.discovery_iteration)
                .max()
                .unwrap_or(phase1_iterations);
            let selected = self
                .choose_final_solution(&collected_solutions)
                .unwrap_or_else(|| accepted[0].clone());
            return DecodeResult {
                decoding: selected.decoding.clone(),
                decoded_detectors: self.get_detectors(selected.decoding.view()),
                posterior_ratios: selected.posterior.clone(),
                success: true,
                decoding_quality: self.get_decoding_quality(selected.decoding.view()),
                iterations: phase1_success_iters,
                max_iter: self.max_iter(),
                extra: BPExtraResult::SLGMBPTrace {
                    phase1_converged: true,
                    phase1_iterations: phase1_success_iters,
                    total_iterations: phase1_success_iters,
                    generation_count: 0,
                    generation_best_fitness,
                    selected_solution_posterior: Some(selected.posterior.clone()),
                    accepted_solution_decodings: accepted_decodings,
                    accepted_solution_weights: accepted_weights,
                    accepted_solution_discovery_iterations: accepted_iters,
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
            let mut generation_successes = Vec::<CollectedSolution>::new();
            let mut gen_best_member_fitness = f64::NEG_INFINITY;
            let mut gen_best_member_posterior: Option<Array1<f64>> = None;
            let mut gen_best_member_adjacent_indices = Vec::<usize>::new();
            let mut gen_best_member_gamma: Option<Array1<f64>> = None;

            for (member_index, mut child) in children.into_iter().enumerate() {
                let switched_to_relay = self.should_use_switched_relay_generation(gen_idx);

                let phase_prior = if switched_to_relay {
                    prior_llr.clone()
                } else {
                    self.build_generation_phase_prior(
                        &prior_llr,
                        &mut child,
                        &mut rng,
                        adaptive_perturbation_llr_threshold,
                    )
                };

                let member_gamma = if switched_to_relay {
                    self.sample_relay_gamma(prior_llr.len(), &mut rng)
                } else {
                    self.build_adaptive_memory_strength(
                        &child.posterior,
                        child.adaptive_target_mask.as_deref(),
                        adaptive_perturbation_llr_threshold,
                        &mut rng,
                    )
                    .unwrap_or_else(|| {
                        self.sample_member_memory_strength(
                            &child.posterior,
                            &child.residual_adjacent_variable_indices,
                            &mut rng,
                        )
                    })
                };

                let initial_marginal = if switched_to_relay {
                    child.posterior.clone()
                } else {
                    self.build_adaptive_initial_marginal(
                        &phase_prior,
                        &child.posterior,
                        child.adaptive_target_mask.as_deref(),
                        adaptive_perturbation_llr_threshold,
                        self.config.reset_marginal,
                    )
                };

                let (decoding, posterior, iters, success) = self.run_mem_bp_phase(
                    detectors,
                    &phase_prior,
                    &child.posterior,
                    Some(&initial_marginal),
                    &member_gamma,
                    self.config.t_mem,
                    if switched_to_relay {
                        false
                    } else {
                        self.config.reset_marginal
                    },
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
                    self.collect_solution(
                        &mut generation_successes,
                        total_iterations + iters,
                        &decoding,
                        &posterior,
                        &prior_llr,
                    );
                    if iters == 1 && self.should_early_break_on_single_iter_success() {
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
                    adaptive_prior: child.adaptive_prior,
                    adaptive_target_mask: child.adaptive_target_mask,
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

            if !generation_successes.is_empty() {
                collected_solutions.extend(generation_successes.clone());
            }

            if self.should_stop_after_success_count(collected_solutions.len()) {
                generation_best_fitness.push(gen_best);
                generation_best_posteriors.push(
                    gen_best_member_posterior.unwrap_or_else(|| best_posterior.clone()),
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
                let accepted = self.accepted_solutions(&collected_solutions);
                let (accepted_decodings, accepted_weights, accepted_iters) =
                    self.accepted_solution_trace_vectors(&collected_solutions);
                let final_iters = accepted
                    .iter()
                    .map(|s| s.discovery_iteration)
                    .max()
                    .unwrap_or(total_iterations);
                let selected = self
                    .choose_final_solution(&collected_solutions)
                    .unwrap_or_else(|| accepted[0].clone());
                let gen_decoded_detectors = self.get_detectors(selected.decoding.view());
                return DecodeResult {
                    decoding: selected.decoding.clone(),
                    decoded_detectors: gen_decoded_detectors,
                    posterior_ratios: selected.posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(selected.decoding.view()),
                    iterations: final_iters,
                    max_iter: self.max_iter(),
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations: final_iters,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(selected.posterior.clone()),
                        accepted_solution_decodings: accepted_decodings,
                        accepted_solution_weights: accepted_weights,
                        accepted_solution_discovery_iterations: accepted_iters,
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
                accepted_solution_decodings: Vec::new(),
                accepted_solution_weights: Vec::new(),
                accepted_solution_discovery_iterations: Vec::new(),
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
        AdaptivePerturbationBiasMode, AdaptivePerturbationFactorMode,
        AdaptivePerturbationPriorBaseMode, AdaptivePerturbationSignMode,
        AdaptivePerturbationTarget, AdaptivePerturbationThresholdMode,
        AdaptivePerturbationVariableBaseMode,
        SLGMBPDecoderConfig,
    };
    use crate::bipartite_graph::BipartiteGraph;
    use crate::decoder::SparseBitMatrix;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
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

        let rebased = decoder.build_adaptive_prior(&base_prior, None, None, &posterior, 0.25);

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
        let positive_rebased = positive_decoder.build_adaptive_prior(
            &base_prior,
            None,
            None,
            &posterior,
            0.25,
        );
        assert_eq!(positive_rebased, array![1.0 + add_val, -2.0, 0.75 + add_val]);

        let negative_decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::Random,
            adaptive_perturbation_positive_sign_prob: 0.0,
            ..SLGMBPDecoderConfig::default()
        });
        let negative_rebased = negative_decoder.build_adaptive_prior(
            &base_prior,
            None,
            None,
            &posterior,
            0.25,
        );
        assert_eq!(negative_rebased, array![1.0 - add_val, -2.0, 0.75 - add_val]);
    }

    #[test]
    fn adaptive_prior_scale_mode_scales_low_confidence_llr() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 0.2,
            adaptive_perturbation_factor_mode: AdaptivePerturbationFactorMode::Fixed,
            adaptive_perturbation_bias_mode: AdaptivePerturbationBiasMode::Scale,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            ..SLGMBPDecoderConfig::default()
        });

        let base_prior = array![1.0, -2.0, 0.75];
        let posterior = array![0.1, 0.3, -0.249];
        let rebased = decoder.build_adaptive_prior(&base_prior, None, None, &posterior, 0.25);
        let expected = array![0.2, -2.0, 0.15];
        for (&got, &exp) in rebased.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12);
        }
    }

    #[test]
    fn adaptive_prior_uniform_factor_mode_uses_interval_for_scale() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor_mode: AdaptivePerturbationFactorMode::UniformPerVariable,
            adaptive_perturbation_factor_interval: (0.5, 0.5),
            adaptive_perturbation_bias_mode: AdaptivePerturbationBiasMode::Scale,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            ..SLGMBPDecoderConfig::default()
        });

        let base_prior = array![1.0, -2.0, 0.75];
        let posterior = array![0.1, 0.3, -0.249];
        let rebased = decoder.build_adaptive_prior(&base_prior, None, None, &posterior, 0.25);

        assert_eq!(rebased, array![0.5, -2.0, 0.375]);
    }

    #[test]
    fn adaptive_initial_marginal_supports_posterior_and_both_targets() {
        let base_prior = array![1.0, -2.0, 0.75];
        let posterior = array![0.1, 0.3, -0.249];
        let add_val = 3.0_f64.ln();

        let posterior_decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            adaptive_perturbation_target: AdaptivePerturbationTarget::Posterior,
            marginal_carry_damping_factor: 1.0,
            ..SLGMBPDecoderConfig::default()
        });
        let posterior_initial = posterior_decoder.build_adaptive_initial_marginal(
            &base_prior,
            &posterior,
            None,
            0.25,
            false,
        );
        assert_eq!(posterior_initial, array![0.1 - add_val, 0.3, -0.249 - add_val]);

        let both_decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            adaptive_perturbation_target: AdaptivePerturbationTarget::Both,
            marginal_carry_damping_factor: 1.0,
            ..SLGMBPDecoderConfig::default()
        });
        let both_prior = both_decoder.build_adaptive_prior(&base_prior, None, None, &posterior, 0.25);
        let both_initial = both_decoder.build_adaptive_initial_marginal(
            &base_prior,
            &posterior,
            None,
            0.25,
            false,
        );
        assert_eq!(both_prior, array![1.0 - add_val, -2.0, 0.75 - add_val]);
        assert_eq!(both_initial, array![0.1 - add_val, 0.3, -0.249 - add_val]);
    }

    #[test]
    fn adaptive_memory_strength_flips_low_llr_sign_from_init_gamma() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            init_gamma: 0.125,
            adaptive_perturbation: true,
            adaptive_perturbation_target: AdaptivePerturbationTarget::MemoryStrength,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::Random,
            adaptive_perturbation_positive_sign_prob: 0.0,
            ..SLGMBPDecoderConfig::default()
        });

        let posterior = array![0.1, 0.3, -0.2];
        let mut rng = StdRng::seed_from_u64(7);
        let gamma = decoder
            .build_adaptive_memory_strength(&posterior, None, 0.25, &mut rng)
            .expect("memory_strength should produce gamma vector");

        assert_eq!(gamma, array![-0.125, 0.125, -0.125]);
    }

    #[test]
    fn adaptive_memory_strength_resets_to_init_gamma_on_threshold_exit_with_fixed_mask() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            init_gamma: 0.125,
            adaptive_perturbation: true,
            adaptive_perturbation_target: AdaptivePerturbationTarget::MemoryStrength,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            adaptive_perturbation_reset_on_threshold_exit: true,
            ..SLGMBPDecoderConfig::default()
        });

        let posterior = array![0.1, 0.6, -0.2];
        let fixed_mask = vec![true, true, true];
        let mut rng = StdRng::seed_from_u64(3);
        let gamma = decoder
            .build_adaptive_memory_strength(&posterior, Some(&fixed_mask), 0.25, &mut rng)
            .expect("memory_strength should produce gamma vector");

        assert_eq!(gamma, array![-0.125, 0.125, -0.125]);
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

    #[test]
    fn adaptive_prior_can_carry_previous_biased_prior() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            adaptive_perturbation_prior_base_mode:
                AdaptivePerturbationPriorBaseMode::PreviousBiased,
            ..SLGMBPDecoderConfig::default()
        });

        let base_prior = array![1.0, -2.0, 0.75];
        let previous_biased = array![0.5, -2.5, 0.1];
        let posterior = array![0.1, 0.6, -0.2];
        let add_val = 3.0_f64.ln();

        let rebased = decoder.build_adaptive_prior(
            &base_prior,
            Some(&previous_biased),
            None,
            &posterior,
            0.25,
        );

        assert_eq!(rebased, array![0.5 - add_val, -2.5, 0.1 - add_val]);
    }

    #[test]
    fn adaptive_prior_can_reset_bias_when_threshold_is_exited() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            adaptive_perturbation_prior_base_mode:
                AdaptivePerturbationPriorBaseMode::PreviousBiased,
            adaptive_perturbation_reset_on_threshold_exit: true,
            ..SLGMBPDecoderConfig::default()
        });

        let base_prior = array![1.0, -2.0, 0.75];
        let previous_biased = array![0.5, -2.5, 0.1];
        let posterior = array![0.1, 0.6, -0.2];
        let add_val = 3.0_f64.ln();

        let rebased = decoder.build_adaptive_prior(
            &base_prior,
            Some(&previous_biased),
            None,
            &posterior,
            0.25,
        );

        assert_eq!(rebased, array![0.5 - add_val, -2.0, 0.1 - add_val]);
    }

    #[test]
    fn adaptive_prior_first_leg_fixed_mode_uses_fixed_mask_not_current_posterior() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::AlwaysNegative,
            adaptive_perturbation_variable_base_mode:
                AdaptivePerturbationVariableBaseMode::FirstLegPosteriorFixed,
            ..SLGMBPDecoderConfig::default()
        });

        let base_prior = array![1.0, -2.0, 0.75];
        // Only variables 0 and 2 are selected by the first-leg fixed mask.
        let fixed_mask = vec![true, false, true];
        // Current posterior would bias only index 1 if previous-leg mode were used.
        let current_posterior = array![0.6, 0.1, 0.7];
        let add_val = 3.0_f64.ln();

        let rebased = decoder.build_adaptive_prior(
            &base_prior,
            None,
            Some(&fixed_mask),
            &current_posterior,
            0.25,
        );

        assert_eq!(rebased, array![1.0 - add_val, -2.0, 0.75 - add_val]);
    }

    #[test]
    fn adaptive_prior_fixed_mask_resamples_sign_each_leg() {
        let decoder = make_decoder(SLGMBPDecoderConfig {
            adaptive_perturbation: true,
            adaptive_perturbation_llr_threshold: 0.25,
            adaptive_perturbation_factor: 3.0,
            adaptive_perturbation_sign_mode: AdaptivePerturbationSignMode::Random,
            adaptive_perturbation_positive_sign_prob: 0.5,
            adaptive_perturbation_variable_base_mode:
                AdaptivePerturbationVariableBaseMode::FirstLegPosteriorFixed,
            ..SLGMBPDecoderConfig::default()
        });

        let base_prior = array![1.0, -2.0, 0.75];
        let fixed_mask = vec![true, false, true];
        let current_posterior = array![0.6, 0.1, 0.7];
        let add_val = 3.0_f64.ln();

        let mut saw_positive = false;
        let mut saw_negative = false;
        for _ in 0..256 {
            let rebased = decoder.build_adaptive_prior(
                &base_prior,
                None,
                Some(&fixed_mask),
                &current_posterior,
                0.25,
            );
            let diff = rebased[0] - base_prior[0];
            if (diff - add_val).abs() < 1e-12 {
                saw_positive = true;
            }
            if (diff + add_val).abs() < 1e-12 {
                saw_negative = true;
            }
            if saw_positive && saw_negative {
                break;
            }
        }

        assert!(saw_positive && saw_negative);
    }
}
