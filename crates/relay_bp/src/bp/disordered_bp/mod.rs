use ndarray::{Array1, ArrayView1};
use num_traits::{Bounded, FromPrimitive, Signed, ToPrimitive};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;
use std::sync::Arc;

use crate::bp::disordered_bp::config::{BiasApplyMode, DisorderedBPDecoderConfig, SamplingMode};
use crate::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use crate::bp::disordered_bp::trace::DisorderedBPLegTrace;
use crate::decoder::{BPExtraResult, Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix};

pub mod config;
pub mod trace;

#[derive(Clone)]
pub struct DisorderedBPDecoder<N: PartialEq + Default + Clone + Copy> {
    bp_decoder: MinSumBPDecoder<N>,
    config: Arc<DisorderedBPDecoderConfig>,
    rng: StdRng,
}

impl<N> DisorderedBPDecoder<N>
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
        config: Arc<DisorderedBPDecoderConfig>,
    ) -> Self {
        let mut inner_cfg = (*min_sum_config).clone();
        inner_cfg.max_iter = config.t_0 + config.maximum_leg.saturating_sub(1) * config.iteration_per_leg;
        inner_cfg.alpha = Some(config.alpha_fixed);
        inner_cfg.gamma0 = Some(config.initial_gamma);
        inner_cfg.enable_variable_message_drop = false;
        inner_cfg.drop_probability = 0.0;
        inner_cfg.drop_llr_threshold = 0.0;
        inner_cfg.rng_seed = Some(config.seed);

        Self {
            bp_decoder: MinSumBPDecoder::new(check_matrix, Arc::new(inner_cfg)),
            config: config.clone(),
            rng: StdRng::seed_from_u64(config.seed),
        }
    }

    fn decode_inner_leg(
        &mut self,
        detectors: ArrayView1<Bit>,
        iter_budget: usize,
        bias_values: &Array1<f64>,
        bias_mask: &[bool],
        apply_bias: bool,
    ) -> DecodeResult {
        let mut success = false;
        let mut decoded_detectors = Array1::default(detectors.dim());

        for _ in 0..iter_budget {
            self.bp_decoder.run_iteration(detectors);
            if apply_bias {
                self.apply_bias_to_posterior(bias_values, bias_mask);
            }

            self.bp_decoder.current_iteration += 1;
            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors, decoded_detectors.view());
            if success {
                break;
            }
        }

        self.bp_decoder.build_result(success, decoded_detectors, iter_budget)
    }

    fn apply_bias_to_posterior(&mut self, bias_values: &Array1<f64>, bias_mask: &[bool]) {
        let mut posterior = self.bp_decoder.posterior_ratios_f64();
        for (idx, posterior_val) in posterior.iter_mut().enumerate() {
            if bias_mask[idx] {
                *posterior_val += bias_values[idx];
            }
        }
        self.bp_decoder.set_posterior_ratios_f64(posterior);
        self.bp_decoder.recompute_hard_decision();
    }

    fn sample_vector(&mut self, mode: SamplingMode, fixed: f64, interval: (f64, f64), len: usize) -> Array1<f64> {
        match mode {
            SamplingMode::Fixed => Array1::from_elem(len, fixed),
            SamplingMode::IntervalRandom => {
                let lo = interval.0.min(interval.1);
                let hi = interval.0.max(interval.1);
                if (hi - lo).abs() < f64::EPSILON {
                    return Array1::from_elem(len, lo);
                }
                let dist = Uniform::new_inclusive(lo, hi);
                Array1::from_iter((0..len).map(|_| dist.sample(&mut self.rng)))
            }
        }
    }

    fn sample_alpha_vector_for_leg(&mut self, leg_idx: usize, variable_count: usize) -> Array1<f64> {
        if leg_idx == 0 {
            return Array1::from_elem(variable_count, self.config.initial_alpha);
        }
        self.sample_vector(
            self.config.alpha_mode,
            self.config.alpha_fixed,
            self.config.alpha_interval,
            variable_count,
        )
    }

    fn sample_gamma_vector_for_leg(&mut self, leg_idx: usize, variable_count: usize) -> Array1<f64> {
        if leg_idx == 0 {
            return Array1::from_elem(variable_count, self.config.initial_gamma);
        }
        self.sample_vector(
            self.config.gamma_mode,
            self.config.gamma_fixed,
            self.config.gamma_interval,
            variable_count,
        )
    }

    fn sample_bias_for_leg(
        &mut self,
        leg_idx: usize,
        variable_count: usize,
        previous_leg_posterior: &Array1<f64>,
    ) -> (Array1<f64>, Vec<bool>) {
        if leg_idx == 0 {
            return (Array1::zeros(variable_count), vec![false; variable_count]);
        }

        let mut bias_values = self.sample_vector(
            self.config.bias_mode,
            self.config.bias_fixed,
            self.config.bias_interval,
            variable_count,
        );

        let flip_prob = self.config.negative_sign_prob.clamp(0.0, 1.0);
        for val in bias_values.iter_mut() {
            if self.rng.gen_bool(flip_prob) {
                *val = -*val;
            }
        }

        let bias_mask = match self.config.bias_apply_mode {
            BiasApplyMode::All => vec![true; variable_count],
            BiasApplyMode::Filter => {
                let threshold = self.config.bias_filter_threshold.abs();
                previous_leg_posterior
                    .iter()
                    .map(|v| v.abs() <= threshold)
                    .collect()
            }
        };

        (bias_values, bias_mask)
    }

    fn build_leg_trace(
        &self,
        result: &DecodeResult,
        alpha_values: &Array1<f64>,
        gamma_values: &Array1<f64>,
        bias_values: &Array1<f64>,
        bias_mask: &[bool],
    ) -> DisorderedBPLegTrace {
        let alpha_mean = if alpha_values.is_empty() {
            0.0
        } else {
            alpha_values.iter().sum::<f64>() / alpha_values.len() as f64
        };

        let gamma_mean = if gamma_values.is_empty() {
            0.0
        } else {
            gamma_values.iter().sum::<f64>() / gamma_values.len() as f64
        };

        let mut bias_sum = 0.0;
        let mut bias_count = 0usize;
        for (idx, value) in bias_values.iter().enumerate() {
            if bias_mask[idx] {
                bias_sum += value.abs();
                bias_count += 1;
            }
        }
        let bias_mean = if bias_count == 0 {
            0.0
        } else {
            bias_sum / bias_count as f64
        };

        DisorderedBPLegTrace {
            success: result.success,
            iterations: result.iterations,
            negative_llr_count: result.posterior_ratios.iter().filter(|x| **x < 0.0).count(),
            decoding: result.decoding.clone(),
            posterior: result.posterior_ratios.clone(),
            alpha_mean,
            gamma_mean,
            bias_mean,
            bias_applied_count: bias_count,
        }
    }
}

impl<N> Decoder for DisorderedBPDecoder<N>
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
        let carry_factor = self.config.carry_marginal_factor.clamp(0.0, 1.0);
        let maximum_leg = self.config.maximum_leg.max(1);
        let total_max_iter = self.config.t_0 + maximum_leg.saturating_sub(1) * self.config.iteration_per_leg;

        let mut total_iterations = 0usize;
        let base_prior_llr = self.log_prior_ratios();
        let mut initial_posterior_for_leg = base_prior_llr.clone();
        let mut previous_leg_posterior = base_prior_llr.clone();
        let mut final_result: Option<DecodeResult> = None;

        let mut leg_traces: Vec<DisorderedBPLegTrace> = Vec::with_capacity(maximum_leg);

        for leg_idx in 0..maximum_leg {
            let iter_budget = if leg_idx == 0 {
                self.config.t_0.max(1)
            } else {
                self.config.iteration_per_leg.max(1)
            };

            let alpha_values = self.sample_alpha_vector_for_leg(leg_idx, variable_count);
            let gamma_values = self.sample_gamma_vector_for_leg(leg_idx, variable_count);
            let (bias_values, bias_mask) =
                self.sample_bias_for_leg(leg_idx, variable_count, &previous_leg_posterior);
            let apply_bias = leg_idx > 0;

            self.bp_decoder.current_iteration = 0;
            self.bp_decoder
                .set_variable_alphas_f64(Some(alpha_values.clone()));
            self.bp_decoder.set_memory_strengths_f64(gamma_values.clone());
            self.bp_decoder
                .set_log_prior_ratio_f64(base_prior_llr.clone());
            self.bp_decoder
                .set_posterior_ratios_f64(initial_posterior_for_leg.clone());
            self.bp_decoder.initialize_check_to_variable();
            self.bp_decoder.initialize_variable_to_check();

            let mut leg_result = self.decode_inner_leg(
                detectors,
                iter_budget,
                &bias_values,
                &bias_mask,
                apply_bias,
            );
            total_iterations += leg_result.iterations;

            let leg_trace = self.build_leg_trace(
                &leg_result,
                &alpha_values,
                &gamma_values,
                &bias_values,
                &bias_mask,
            );
            leg_traces.push(leg_trace);

            previous_leg_posterior = self.bp_decoder.posterior_ratios_f64();

            leg_result.max_iter = total_max_iter;
            leg_result.iterations = total_iterations;
            final_result = Some(leg_result.clone());

            if leg_result.success {
                break;
            }

            initial_posterior_for_leg = previous_leg_posterior.mapv(|v| carry_factor * v);
        }

        self.bp_decoder.clear_variable_alphas();

        let mut result = final_result.unwrap_or_else(|| {
            let decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            self.bp_decoder
                .build_result(false, decoded_detectors, total_max_iter)
        });
        result.max_iter = total_max_iter;
        result.iterations = total_iterations;
        result.extra = BPExtraResult::DisorderedBPTrace {
            leg_success: leg_traces.iter().map(|t| t.success).collect(),
            leg_iterations: leg_traces.iter().map(|t| t.iterations).collect(),
            leg_negative_llr_counts: leg_traces.iter().map(|t| t.negative_llr_count).collect(),
            leg_decodings: leg_traces.iter().map(|t| t.decoding.clone()).collect(),
            leg_posteriors: leg_traces.iter().map(|t| t.posterior.clone()).collect(),
            leg_alpha_means: leg_traces.iter().map(|t| t.alpha_mean).collect(),
            leg_gamma_means: leg_traces.iter().map(|t| t.gamma_mean).collect(),
            leg_bias_means: leg_traces.iter().map(|t| t.bias_mean).collect(),
            leg_bias_applied_counts: leg_traces.iter().map(|t| t.bias_applied_count).collect(),
        };
        result
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.bp_decoder.get_decoding_quality(errors)
    }
}

impl<N> DecoderRunner for DisorderedBPDecoder<N>
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
