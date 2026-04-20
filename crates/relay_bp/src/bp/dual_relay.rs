use super::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use super::relay::StoppingCriterion;
use crate::decoder::{BPExtraResult, Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix};

use ndarray::{Array1, ArrayView1};
use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualRelayMixMode {
    NaiveAverage,
    WeightedFast,
}

#[derive(Clone, Debug)]
pub struct DualRelayDecoderConfig {
    pub pre_iter: usize,
    pub maximum_leg: usize,
    pub iteration_per_leg: usize,
    pub initial_gamma_slow: f64,
    pub initial_gamma_fast: f64,
    pub gamma_interval_slow: (f64, f64),
    pub gamma_interval_fast: (f64, f64),
    pub mix_mode: DualRelayMixMode,
    pub eta: f64,
    pub delta: f64,
    pub use_previous_message: bool,
    pub beta: f64,
    pub ensemble_mode: bool,
    pub ensemble_size: usize,
    pub ensemble_gamma_interval: (f64, f64),
    pub num_pre_iteration_instance: usize,
    pub initial_gamma: Vec<f64>,
    pub n_solutions: usize,
    pub stopping_criterion: StoppingCriterion,
    pub seed: u64,
    pub collect_iteration_metric: bool,
}

impl Default for DualRelayDecoderConfig {
    fn default() -> Self {
        Self {
            pre_iter: 80,
            maximum_leg: 100,
            iteration_per_leg: 60,
            initial_gamma_slow: 0.125,
            initial_gamma_fast: 0.125,
            gamma_interval_slow: (0.1, 0.66),
            gamma_interval_fast: (0.1, 0.66),
            mix_mode: DualRelayMixMode::NaiveAverage,
            eta: 0.5,
            delta: 1.0,
            use_previous_message: false,
            beta: 0.0,
            ensemble_mode: false,
            ensemble_size: 2,
            ensemble_gamma_interval: (0.1, 0.66),
            num_pre_iteration_instance: 1,
            initial_gamma: vec![0.125],
            n_solutions: 1,
            stopping_criterion: StoppingCriterion::NConv { stop_after: 1 },
            seed: 0,
            collect_iteration_metric: true,
        }
    }
}

#[derive(Clone)]
pub struct DualRelayDecoder<N: PartialEq + Default + Clone + Copy> {
    bp_decoder: MinSumBPDecoder<N>,
    config: Arc<DualRelayDecoderConfig>,
    rng: StdRng,
}

impl<N> DualRelayDecoder<N>
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
        config: Arc<DualRelayDecoderConfig>,
    ) -> Self {
        let mut inner_cfg = (*min_sum_config).clone();
        inner_cfg.max_iter = config.pre_iter
            + config
                .maximum_leg
                .saturating_sub(1)
                .saturating_mul(config.iteration_per_leg);
        inner_cfg.gamma0 = Some(config.initial_gamma_slow);
        inner_cfg.enable_variable_message_drop = false;
        inner_cfg.drop_probability = 0.0;
        inner_cfg.drop_llr_threshold = 0.0;

        Self {
            bp_decoder: MinSumBPDecoder::new(check_matrix, Arc::new(inner_cfg)),
            config: config.clone(),
            rng: StdRng::seed_from_u64(config.seed),
        }
    }

    fn sample_gamma_vector(&mut self, interval: (f64, f64), variable_count: usize) -> Array1<f64> {
        let lo = interval.0.min(interval.1);
        let hi = interval.0.max(interval.1);
        if (hi - lo).abs() < f64::EPSILON {
            return Array1::from_elem(variable_count, lo);
        }
        let dist = Uniform::new_inclusive(lo, hi);
        Array1::from_iter((0..variable_count).map(|_| dist.sample(&mut self.rng)))
    }

    fn gammas_for_leg(&mut self, leg_idx: usize, variable_count: usize) -> (Array1<f64>, Array1<f64>) {
        if leg_idx == 0 {
            return (
                Array1::from_elem(variable_count, self.config.initial_gamma_slow),
                Array1::from_elem(variable_count, self.config.initial_gamma_fast),
            );
        }

        (
            self.sample_gamma_vector(self.config.gamma_interval_slow, variable_count),
            self.sample_gamma_vector(self.config.gamma_interval_fast, variable_count),
        )
    }

    fn sample_gamma_scalar(&mut self, interval: (f64, f64)) -> f64 {
        let lo = interval.0.min(interval.1);
        let hi = interval.0.max(interval.1);
        if (hi - lo).abs() < f64::EPSILON {
            return lo;
        }
        let dist = Uniform::new_inclusive(lo, hi);
        dist.sample(&mut self.rng)
    }

    fn ensemble_gammas_for_iteration(
        &mut self,
        leg_idx: usize,
        variable_count: usize,
    ) -> Vec<Array1<f64>> {
        let count = if leg_idx == 0 {
            self.config.num_pre_iteration_instance.max(1)
        } else {
            self.config.ensemble_size.max(1)
        };

        if leg_idx == 0 {
            let init = if self.config.initial_gamma.is_empty() {
                vec![self.config.initial_gamma_slow]
            } else {
                self.config.initial_gamma.clone()
            };
            return (0..count)
                .map(|i| {
                    let gamma = if i < init.len() {
                        init[i]
                    } else {
                        *init.last().unwrap_or(&self.config.initial_gamma_slow)
                    };
                    Array1::from_elem(variable_count, gamma)
                })
                .collect();
        }

        (0..count)
            .map(|_| {
                let gamma = self.sample_gamma_scalar(self.config.ensemble_gamma_interval);
                Array1::from_elem(variable_count, gamma)
            })
            .collect()
    }

    fn same_sign(a: f64, b: f64) -> bool {
        (a >= 0.0 && b >= 0.0) || (a < 0.0 && b < 0.0)
    }

    fn mix_posteriors(
        &self,
        posterior_slow: &Array1<f64>,
        posterior_fast: &Array1<f64>,
    ) -> (Array1<f64>, bool) {
        let eta = self.config.eta;
        let delta = self.config.delta;
        let mut used_fast_weighted_disagreement = false;

        let mixed = Array1::from_iter(
            posterior_slow
                .iter()
                .zip(posterior_fast.iter())
                .map(|(slow, fast)| match self.config.mix_mode {
                    DualRelayMixMode::NaiveAverage => (1.0 - eta) * slow + eta * fast,
                    DualRelayMixMode::WeightedFast => {
                        if Self::same_sign(*slow, *fast) {
                            (1.0 - eta) * slow + eta * fast
                        } else {
                            used_fast_weighted_disagreement = true;
                            delta * fast + slow
                        }
                    }
                }),
        );

        (mixed, used_fast_weighted_disagreement)
    }

    fn update_best_result(best_result: &mut Option<DecodeResult>, candidate: &DecodeResult) {
        match best_result {
            Some(current_best) => {
                if candidate.decoding_quality < current_best.decoding_quality {
                    *best_result = Some(candidate.clone());
                }
            }
            None => {
                *best_result = Some(candidate.clone());
            }
        }
    }
}

impl<N> Decoder for DualRelayDecoder<N>
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.bp_decoder.check_matrix()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.bp_decoder.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let variable_count = self.check_matrix().cols();
        let maximum_leg = self.config.maximum_leg.max(1);
        let pre_iter = self.config.pre_iter.max(1);
        let iter_per_leg = self.config.iteration_per_leg.max(1);
        let max_iter_total = pre_iter + maximum_leg.saturating_sub(1).saturating_mul(iter_per_leg);
        let required_unique = self.config.n_solutions.max(1);

        let mut total_iterations = 0usize;
        let mut num_converged_legs = 0usize;
        let mut unique_decodings: HashSet<Vec<Bit>> = HashSet::new();
        let mut best_result: Option<DecodeResult> = None;
        let mut last_result: Option<DecodeResult> = None;

        let mut leg_success: Vec<bool> = Vec::with_capacity(maximum_leg);
        let mut leg_iterations: Vec<usize> = Vec::with_capacity(maximum_leg);
        let mut leg_negative_llr_counts: Vec<usize> = Vec::with_capacity(maximum_leg);
        let mut leg_decodings: Vec<Array1<Bit>> = Vec::with_capacity(maximum_leg);
        let mut leg_posteriors: Vec<Array1<f64>> = Vec::with_capacity(maximum_leg);

        let mut slow_success_count = 0usize;
        let mut fast_success_count = 0usize;
        let mut joint_success_count = 0usize;
        let mut naive_average_applied_count = 0usize;
        let mut weighted_fast_disagreement_applied_count = 0usize;

        self.bp_decoder.initialize_decoder();

        'legs: for leg_idx in 0..maximum_leg {
            let iter_budget = if leg_idx == 0 { pre_iter } else { iter_per_leg };
            let (gammas_slow, gammas_fast) = self.gammas_for_leg(leg_idx, variable_count);

            self.bp_decoder.current_iteration = 0;
            self.bp_decoder.initialize_check_to_variable();
            self.bp_decoder.initialize_variable_to_check();

            let mut current_leg_success = false;
            let mut current_leg_iterations = 0usize;
            let mut decoded_detectors = Array1::default(detectors.dim());
            let mut final_mode = "none".to_string();
            let mut final_mode_score = f64::MAX;

            for _ in 0..iter_budget {
                let previous_posterior = self.bp_decoder.posterior_ratios_f64();
                let previous_check_message_sum = if self.config.use_previous_message {
                    Some(self.bp_decoder.sum_check_to_variable_by_variable_f64())
                } else {
                    None
                };

                self.bp_decoder.run_check_to_variable_update(detectors);

                if self.config.ensemble_mode {
                    let gamma_ensemble = self.ensemble_gammas_for_iteration(leg_idx, variable_count);
                    let mut ensemble_posteriors: Vec<Array1<f64>> = Vec::with_capacity(gamma_ensemble.len());
                    let mut best_success: Option<(Array1<f64>, Array1<Bit>, f64)> = None;

                    for gammas in &gamma_ensemble {
                        let posterior = self
                            .bp_decoder
                            .predict_posterior_from_previous_and_memory_f64(
                                &previous_posterior,
                                gammas,
                                previous_check_message_sum.as_ref(),
                                self.config.beta,
                            );
                        self.bp_decoder.set_posterior_ratios_f64(posterior.clone());
                        self.bp_decoder.recompute_hard_decision();
                        let decoding = self.bp_decoder.current_decoding().clone();
                        let decoded_detectors_candidate = self.bp_decoder.compute_decoded_detectors();
                        let success = self
                            .bp_decoder
                            .check_convergence(detectors, decoded_detectors_candidate.view());
                        let score = self.bp_decoder.get_decoding_quality(decoding.view());

                        if success {
                            match &best_success {
                                Some((_, _, best_score)) if score > *best_score => {}
                                _ => {
                                    best_success = Some((
                                        posterior.clone(),
                                        decoded_detectors_candidate.clone(),
                                        score,
                                    ));
                                }
                            }
                        }

                        ensemble_posteriors.push(posterior);
                    }

                    if let Some((best_posterior, best_decoded_detectors, best_score)) = best_success {
                        self.bp_decoder
                            .set_posterior_and_rebuild_variable_to_check_f64(best_posterior);
                        decoded_detectors = best_decoded_detectors;
                        final_mode = "ensemble_member".to_string();
                        final_mode_score = best_score;
                        current_leg_success = true;
                    } else {
                        let ensemble_count = ensemble_posteriors.len();
                        let mut median_posterior = Array1::<f64>::zeros(variable_count);
                        if ensemble_count > 0 {
                            let mut values_buffer = vec![0.0; ensemble_posteriors.len()];
                            for v in 0..variable_count {
                                for i in 0..ensemble_count {
                                    values_buffer[i] = ensemble_posteriors[i as usize][v];
                                }
                                values_buffer.sort_unstable_by(|a, b| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                median_posterior[v] = if ensemble_count as usize % 2 == 1 {
                                    values_buffer[ensemble_count as usize / 2]
                                } else {
                                    let mid = ensemble_count as usize / 2;
                                    (values_buffer[mid - 1] + values_buffer[mid]) / 2.0
                                };
                            }
                        }

                        // let mut averaged_posterior = Array1::<f64>::zeros(variable_count);
                        // for posterior in &ensemble_posteriors {
                        //     averaged_posterior += posterior;
                        // }
                        // averaged_posterior /= ensemble_count;

                        self.bp_decoder
                            .set_posterior_and_rebuild_variable_to_check_f64(median_posterior);
                        decoded_detectors = self.bp_decoder.compute_decoded_detectors();
                        current_leg_success = self
                            .bp_decoder
                            .check_convergence(detectors, decoded_detectors.view());
                        final_mode = "ensemble_median".to_string();
                        let decoding_joint = self.bp_decoder.current_decoding().clone();
                        final_mode_score = self.bp_decoder.get_decoding_quality(decoding_joint.view());
                        joint_success_count += usize::from(current_leg_success);
                    }

                    self.bp_decoder.current_iteration += 1;
                    current_leg_iterations += 1;
                    total_iterations += 1;

                    if current_leg_success {
                        break;
                    }

                    continue;
                }

                let posterior_slow = self
                    .bp_decoder
                    .predict_posterior_from_previous_and_memory_f64(
                        &previous_posterior,
                        &gammas_slow,
                        previous_check_message_sum.as_ref(),
                        self.config.beta,
                    );
                self.bp_decoder.set_posterior_ratios_f64(posterior_slow.clone());
                self.bp_decoder.recompute_hard_decision();
                let decoding_slow = self.bp_decoder.current_decoding().clone();
                let decoded_detectors_slow = self.bp_decoder.compute_decoded_detectors();
                let success_slow = self
                    .bp_decoder
                    .check_convergence(detectors, decoded_detectors_slow.view());
                let score_slow = self.bp_decoder.get_decoding_quality(decoding_slow.view());

                let posterior_fast = self
                    .bp_decoder
                    .predict_posterior_from_previous_and_memory_f64(
                        &previous_posterior,
                        &gammas_fast,
                        previous_check_message_sum.as_ref(),
                        self.config.beta,
                    );
                self.bp_decoder.set_posterior_ratios_f64(posterior_fast.clone());
                self.bp_decoder.recompute_hard_decision();
                let decoding_fast = self.bp_decoder.current_decoding().clone();
                let decoded_detectors_fast = self.bp_decoder.compute_decoded_detectors();
                let success_fast = self
                    .bp_decoder
                    .check_convergence(detectors, decoded_detectors_fast.view());
                let score_fast = self.bp_decoder.get_decoding_quality(decoding_fast.view());

                if success_slow || success_fast {
                    if success_slow && (!success_fast || score_slow <= score_fast) {
                        self.bp_decoder
                            .set_posterior_and_rebuild_variable_to_check_f64(
                                posterior_slow.clone(),
                            );
                        decoded_detectors = decoded_detectors_slow;
                        final_mode = "slow".to_string();
                        final_mode_score = score_slow;
                        slow_success_count += 1;
                    } else {
                        self.bp_decoder
                            .set_posterior_and_rebuild_variable_to_check_f64(
                                posterior_fast.clone(),
                            );
                        decoded_detectors = decoded_detectors_fast;
                        final_mode = "fast".to_string();
                        final_mode_score = score_fast;
                        fast_success_count += 1;
                    }
                    current_leg_success = true;
                } else {
                    let (mixed_posterior, used_disagreement_weighting) =
                        self.mix_posteriors(&posterior_slow, &posterior_fast);
                    self.bp_decoder
                        .set_posterior_and_rebuild_variable_to_check_f64(mixed_posterior);
                    decoded_detectors = self.bp_decoder.compute_decoded_detectors();
                    current_leg_success = self.bp_decoder.check_convergence(detectors, decoded_detectors.view());
                    final_mode = "joint".to_string();
                    let decoding_joint = self.bp_decoder.current_decoding().clone();
                    final_mode_score = self.bp_decoder.get_decoding_quality(decoding_joint.view());
                    joint_success_count += usize::from(current_leg_success);

                    match self.config.mix_mode {
                        DualRelayMixMode::NaiveAverage => {
                            naive_average_applied_count += 1;
                        }
                        DualRelayMixMode::WeightedFast => {
                            if used_disagreement_weighting {
                                weighted_fast_disagreement_applied_count += 1;
                            }
                        }
                    }
                }

                self.bp_decoder.current_iteration += 1;
                current_leg_iterations += 1;
                total_iterations += 1;

                if current_leg_success {
                    break;
                }
            }

            let mut result =
                self.bp_decoder
                    .build_result(current_leg_success, decoded_detectors, max_iter_total);
            result.iterations = total_iterations;
            result.max_iter = max_iter_total;

            leg_success.push(current_leg_success);
            leg_iterations.push(current_leg_iterations);
            leg_negative_llr_counts.push(result.posterior_ratios.iter().filter(|x| **x < 0.0).count());
            leg_decodings.push(result.decoding.clone());
            leg_posteriors.push(result.posterior_ratios.clone());

            if current_leg_success {
                num_converged_legs += 1;
                let decode_key = result.decoding.iter().copied().collect::<Vec<_>>();
                unique_decodings.insert(decode_key);
                Self::update_best_result(&mut best_result, &result);

                let stop_by_unique = unique_decodings.len() >= required_unique;
                let stop_by_criterion = match self.config.stopping_criterion {
                    StoppingCriterion::PreIter => leg_idx == 0,
                    StoppingCriterion::NConv { stop_after } => num_converged_legs >= stop_after,
                    StoppingCriterion::All => false,
                };

                if stop_by_unique || stop_by_criterion {
                    last_result = Some(result);
                    break 'legs;
                }
            }

            let _ = final_mode;
            let _ = final_mode_score;
            last_result = Some(result);
        }

        let mut final_result = best_result
            .or(last_result)
            .unwrap_or_else(|| {
                let decoded_detectors = self.bp_decoder.compute_decoded_detectors();
                self.bp_decoder
                    .build_result(false, decoded_detectors, max_iter_total)
            });

        final_result.iterations = total_iterations;
        final_result.max_iter = max_iter_total;
        final_result.extra = BPExtraResult::DualRelayTrace {
            leg_success,
            leg_iterations,
            leg_negative_llr_counts,
            leg_decodings,
            leg_posteriors,
            slow_success_count,
            fast_success_count,
            joint_success_count,
            naive_average_applied_count,
            weighted_fast_disagreement_applied_count,
            unique_solution_count: unique_decodings.len(),
        };

        final_result
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.bp_decoder.get_decoding_quality(errors)
    }
}

impl<N> DecoderRunner for DualRelayDecoder<N>
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
