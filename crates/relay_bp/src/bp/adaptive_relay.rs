use super::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use crate::decoder::{BPExtraResult, Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix};

use ndarray::{Array1, ArrayView1};
use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum AdaptiveRelayUpdateMode {
    PerIteration,
    PerLeg,
}

impl Default for AdaptiveRelayUpdateMode {
    fn default() -> Self {
        Self::PerIteration
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AdaptiveRelayPerturbationMode {
    Uniform,
    Gaussian,
}

impl Default for AdaptiveRelayPerturbationMode {
    fn default() -> Self {
        Self::Uniform
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PosteriorMarginalClampMode {
    NoClamp,
    AbsThresholdClamp,
}

impl Default for PosteriorMarginalClampMode {
    fn default() -> Self {
        Self::NoClamp
    }
}

#[derive(Clone, Debug)]
pub struct AdaptiveRelayDecoderConfig {
    pub initial_gamma: f64,
    pub gamma_min: f64,
    pub gamma_max: f64,
    pub tau: f64,
    pub beta: f64,
    pub pre_decoding: bool,
    pub pre_iteration: usize,
    pub maximum_iteration: usize,
    pub iter_per_leg: usize,
    pub update_mode: AdaptiveRelayUpdateMode,
    pub perturbation_mode: AdaptiveRelayPerturbationMode,
    pub perturbation_interval: (f64, f64),
    pub perturbation_sigma: f64,
    pub ensemble_size: usize,
    pub carry_marginal_between_legs: bool,
    pub posterior_marginal_clamp_mode: PosteriorMarginalClampMode,
    pub posterior_marginal_abs_threshold: f64,
    pub seed: u64,
    pub collect_iteration_metric: bool,
}

impl Default for AdaptiveRelayDecoderConfig {
    fn default() -> Self {
        Self {
            initial_gamma: 0.125,
            gamma_min: -0.24,
            gamma_max: 0.66,
            tau: 1.0,
            beta: 0.9,
            pre_decoding: false,
            pre_iteration: 80,
            maximum_iteration: 600,
            iter_per_leg: 60,
            update_mode: AdaptiveRelayUpdateMode::PerIteration,
            perturbation_mode: AdaptiveRelayPerturbationMode::Uniform,
            perturbation_interval: (0.0, 0.0),
            perturbation_sigma: 0.0,
            ensemble_size: 1,
            carry_marginal_between_legs: true,
            posterior_marginal_clamp_mode: PosteriorMarginalClampMode::NoClamp,
            posterior_marginal_abs_threshold: 1e10,
            seed: 0,
            collect_iteration_metric: false,
        }
    }
}

#[derive(Clone)]
struct AdaptiveRelayState {
    rng: StdRng,
    uniform: Option<Uniform<f64>>,
    fixed_uniform: Option<f64>,
}

#[derive(Clone)]
pub struct AdaptiveRelayDecoder<N: PartialEq + Default + Clone + Copy> {
    bp_decoder: MinSumBPDecoder<N>,
    config: Arc<AdaptiveRelayDecoderConfig>,
    state: AdaptiveRelayState,
}

#[derive(Clone)]
struct MemberRunTrace {
    success: bool,
    discovery_iteration: Option<usize>,
    total_iterations: usize,
    leg_iterations: Vec<usize>,
    gamma_history: Vec<f64>,
    clamp_applied_count: usize,
}

impl<N> AdaptiveRelayDecoder<N>
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
        config: Arc<AdaptiveRelayDecoderConfig>,
    ) -> AdaptiveRelayDecoder<N> {
        let low = config.perturbation_interval.0;
        let high = config.perturbation_interval.1;
        let (uniform, fixed_uniform) = if low == high {
            (None, Some(low))
        } else {
            (Some(Uniform::new(low, high)), None)
        };

        AdaptiveRelayDecoder {
            bp_decoder: MinSumBPDecoder::new(check_matrix, min_sum_config),
            config: config.clone(),
            state: AdaptiveRelayState {
                rng: StdRng::seed_from_u64(config.seed),
                uniform,
                fixed_uniform,
            },
        }
    }

    fn sample_standard_normal(&mut self) -> f64 {
        let u1 = self.state.rng.gen_range(f64::EPSILON..1.0);
        let u2 = self.state.rng.gen_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn sample_perturbation(&mut self) -> f64 {
        match self.config.perturbation_mode {
            AdaptiveRelayPerturbationMode::Uniform => {
                if let Some(v) = self.state.fixed_uniform {
                    return v;
                }
                self.state
                    .uniform
                    .as_ref()
                    .map(|u| u.sample(&mut self.state.rng))
                    .unwrap_or(0.0)
            }
            AdaptiveRelayPerturbationMode::Gaussian => {
                let sigma = self.config.perturbation_sigma.max(0.0);
                if sigma == 0.0 {
                    0.0
                } else {
                    sigma * self.sample_standard_normal()
                }
            }
        }
    }

    fn compute_target_gamma(&self, llr_abs: f64) -> f64 {
        let tau = self.config.tau.max(1e-12);
        self.config.gamma_min
            + (self.config.gamma_max - self.config.gamma_min) * (llr_abs / tau).tanh()
    }

    fn update_gamma_from_posterior(
        &mut self,
        posterior: &Array1<f64>,
        mu: &mut Array1<f64>,
    ) -> Array1<f64> {
        let beta = self.config.beta.clamp(0.0, 1.0);
        let mut gamma = Array1::<f64>::zeros(posterior.len());
        for i in 0..posterior.len() {
            let target = self.compute_target_gamma(posterior[i].abs());
            mu[i] = beta * mu[i] + (1.0 - beta) * target;
            gamma[i] = mu[i] + self.sample_perturbation();
        }
        gamma
    }

    fn apply_gamma(&mut self, gamma: &Array1<f64>) {
        self.bp_decoder.set_memory_strengths_f64(gamma.clone());
    }

    fn maybe_clamp_posterior(&mut self) -> (Array1<f64>, usize) {
        let posterior = self.bp_decoder.posterior_ratios_f64();
        if self.config.posterior_marginal_clamp_mode == PosteriorMarginalClampMode::NoClamp {
            return (posterior, 0);
        }

        let threshold = self.config.posterior_marginal_abs_threshold;
        if !(threshold.is_finite() && threshold > 0.0) {
            return (posterior, 0);
        }

        let mut clamped = posterior.clone();
        let mut changes = 0usize;
        for v in clamped.iter_mut() {
            let bounded = (*v).clamp(-threshold, threshold);
            if bounded != *v {
                changes += 1;
                *v = bounded;
            }
        }

        if changes > 0 {
            self.bp_decoder.set_posterior_ratios_f64(clamped.clone());
            self.bp_decoder.recompute_hard_decision();
        }

        (clamped, changes)
    }

    fn run_single_iteration(&mut self, detectors: ArrayView1<Bit>) -> (bool, Array1<Bit>, Array1<f64>, usize) {
        self.bp_decoder.run_iteration(detectors);
        let (_, clamp_count) = self.maybe_clamp_posterior();
        let decoding = self.bp_decoder.current_decoding().clone();
        let decoded_detectors = self.bp_decoder.compute_decoded_detectors();
        let success = self
            .bp_decoder
            .check_convergence(detectors, decoded_detectors.view());
        if !success {
            self.bp_decoder.current_iteration += 1;
        }
        let posterior = self.bp_decoder.posterior_ratios_f64();
        (success, decoding, posterior, clamp_count)
    }

    fn prepare_new_leg(&mut self, carry_marginal: bool) {
        self.bp_decoder.current_iteration = 0;
        if !carry_marginal {
            self.bp_decoder.set_posterior_ratios_to_priors();
        }
        self.bp_decoder.initialize_check_to_variable();
        self.bp_decoder.initialize_variable_to_check();
    }

    fn run_member(
        &mut self,
        detectors: ArrayView1<Bit>,
    ) -> (DecodeResult, MemberRunTrace) {
        let n_vars = self.check_matrix().cols();
        let max_total = self.config.maximum_iteration.max(1);
        let mut total_iters = 0usize;
        let mut first_success_iteration: Option<usize> = None;
        let mut leg_iterations = Vec::<usize>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut clamp_applied_count = 0usize;

        self.bp_decoder.initialize_decoder();

        let mut mu = Array1::<f64>::from_elem(n_vars, self.config.initial_gamma);
        let mut gamma = Array1::<f64>::from_elem(n_vars, self.config.initial_gamma);
        self.apply_gamma(&gamma);
        gamma_history.push(gamma.iter().copied().sum::<f64>() / (gamma.len().max(1) as f64));

        let mut latest_decoding = self.bp_decoder.current_decoding().clone();
        let mut latest_decoded_detectors = self.bp_decoder.compute_decoded_detectors();
        let mut latest_success = false;

        let pre_iters = if self.config.pre_decoding {
            self.config.pre_iteration.min(max_total)
        } else {
            0
        };

        let mut pre_leg_count = 0usize;
        for _ in 0..pre_iters {
            let (success, decoding, _posterior, clamp_count) = self.run_single_iteration(detectors);
            total_iters += 1;
            pre_leg_count += 1;
            clamp_applied_count += clamp_count;
            latest_success = success;
            latest_decoding = decoding;
            latest_decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            if success {
                first_success_iteration = Some(total_iters);
                break;
            }
        }
        if pre_leg_count > 0 {
            leg_iterations.push(pre_leg_count);
        }

        if first_success_iteration.is_none() {
            match self.config.update_mode {
                AdaptiveRelayUpdateMode::PerIteration => {
                    while total_iters < max_total {
                        let (success, decoding, posterior, clamp_count) =
                            self.run_single_iteration(detectors);
                        total_iters += 1;
                        clamp_applied_count += clamp_count;
                        latest_success = success;
                        latest_decoding = decoding;
                        latest_decoded_detectors = self.bp_decoder.compute_decoded_detectors();
                        leg_iterations.push(1);

                        if success {
                            first_success_iteration = Some(total_iters);
                            break;
                        }

                        gamma = self.update_gamma_from_posterior(&posterior, &mut mu);
                        self.apply_gamma(&gamma);
                        gamma_history.push(
                            gamma.iter().copied().sum::<f64>() / (gamma.len().max(1) as f64),
                        );
                    }
                }
                AdaptiveRelayUpdateMode::PerLeg => {
                    while total_iters < max_total {
                        let remaining = max_total - total_iters;
                        let leg_steps = self.config.iter_per_leg.max(1).min(remaining);
                        self.prepare_new_leg(self.config.carry_marginal_between_legs);
                        self.apply_gamma(&gamma);
                        gamma_history.push(
                            gamma.iter().copied().sum::<f64>() / (gamma.len().max(1) as f64),
                        );

                        let mut this_leg_iters = 0usize;
                        for _ in 0..leg_steps {
                            let (success, decoding, _posterior, clamp_count) =
                                self.run_single_iteration(detectors);
                            total_iters += 1;
                            this_leg_iters += 1;
                            clamp_applied_count += clamp_count;
                            latest_success = success;
                            latest_decoding = decoding;
                            latest_decoded_detectors = self.bp_decoder.compute_decoded_detectors();

                            if success {
                                first_success_iteration = Some(total_iters);
                                break;
                            }
                        }
                        leg_iterations.push(this_leg_iters);

                        if first_success_iteration.is_some() {
                            break;
                        }

                        let posterior = self.bp_decoder.posterior_ratios_f64();
                        gamma = self.update_gamma_from_posterior(&posterior, &mut mu);
                    }
                }
            }
        }

        let mut result = self
            .bp_decoder
            .build_result(latest_success, latest_decoded_detectors, max_total);
        result.decoding = latest_decoding;
        result.success = first_success_iteration.is_some();
        result.iterations = first_success_iteration.unwrap_or(total_iters);

        let trace = MemberRunTrace {
            success: result.success,
            discovery_iteration: first_success_iteration,
            total_iterations: total_iters,
            leg_iterations,
            gamma_history,
            clamp_applied_count,
        };

        (result, trace)
    }
}

impl<N> Decoder for AdaptiveRelayDecoder<N>
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
        let ensemble_size = self.config.ensemble_size.max(1);

        let mut member_success = Vec::<bool>::with_capacity(ensemble_size);
        let mut member_discovery_iterations = Vec::<Option<usize>>::with_capacity(ensemble_size);
        let mut member_total_iterations = Vec::<usize>::with_capacity(ensemble_size);

        let mut winner_member_index: Option<usize> = None;
        let mut winner_result: Option<DecodeResult> = None;
        let mut winner_trace: Option<MemberRunTrace> = None;

        for member_idx in 0..ensemble_size {
            let (result, trace) = self.run_member(detectors);
            member_success.push(trace.success);
            member_discovery_iterations.push(trace.discovery_iteration);
            member_total_iterations.push(trace.total_iterations);

            if trace.success {
                winner_member_index = Some(member_idx);
                winner_result = Some(result);
                winner_trace = Some(trace);
                break;
            }

            if winner_result.is_none() {
                winner_result = Some(result);
                winner_trace = Some(trace);
            }
        }

        let mut final_result = winner_result.unwrap();
        let final_trace = winner_trace.unwrap();

        final_result.extra = BPExtraResult::AdaptiveRelayTrace {
            executed_members: member_success.len(),
            winner_member_index,
            member_success,
            member_discovery_iterations,
            member_total_iterations,
            leg_iterations: final_trace.leg_iterations,
            gamma_history: final_trace.gamma_history,
            clamp_applied_count: final_trace.clamp_applied_count,
        };

        final_result
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.bp_decoder.get_decoding_quality(errors)
    }
}

impl<N> DecoderRunner for AdaptiveRelayDecoder<N>
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
}
