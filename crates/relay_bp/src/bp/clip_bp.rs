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
use crate::decoder::{BPExtraResult, Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix};

use ndarray::{Array1, ArrayView1};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum SignUpdateMode {
    Naive,
    Hysteresis,
    NeighborOnly,
}

#[derive(Clone, Debug)]
pub struct ClipBpConfig {
    pub min_llr: f64,
    pub sign_mode: SignUpdateMode,
    pub gamma_first: f64,
    pub gamma_center: f64,
    pub gamma_width: f64,
    pub max_legs: usize,
    pub max_iter_first: usize,
    pub max_iter: usize,
    pub max_solutions: usize,
    pub seed: u64,
}

#[derive(Clone)]
pub struct ClipBpDecoder {
    bp_decoder: MinSumBPDecoder<f64>,
    min_sum_config: Arc<MinSumDecoderConfig>,
    clip_config: Arc<ClipBpConfig>,
    check_matrix: Arc<SparseBitMatrix>,
    rng: StdRng,
    current_gammas: Array1<f64>,
    m_tilde_prev: Array1<f64>,
    hysteresis_latch: Vec<i8>,
}

impl ClipBpDecoder {
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        clip_config: Arc<ClipBpConfig>,
    ) -> Self {
        let n = check_matrix.cols();
        let rng = StdRng::seed_from_u64(clip_config.seed);
        let bp_decoder =
            MinSumBPDecoder::new(Arc::clone(&check_matrix), Arc::clone(&min_sum_config));

        Self {
            bp_decoder,
            min_sum_config,
            clip_config,
            check_matrix,
            rng,
            current_gammas: Array1::zeros(n),
            m_tilde_prev: Array1::zeros(n),
            hysteresis_latch: vec![1_i8; n],
        }
    }

    fn sample_gammas(&mut self) -> Array1<f64> {
        let n = self.check_matrix.cols();
        let half = self.clip_config.gamma_width / 2.0;
        let lo = self.clip_config.gamma_center - half;
        let hi = self.clip_config.gamma_center + half;

        if lo >= hi {
            Array1::from_elem(n, self.clip_config.gamma_center)
        } else {
            let dist = Uniform::new(lo, hi);
            Array1::from_shape_fn(n, |_| dist.sample(&mut self.rng))
        }
    }

    fn compute_clipped(&mut self, raw: &Array1<f64>, prior_llr: &Array1<f64>) -> Array1<f64> {
        let min_llr = self.clip_config.min_llr.abs();
        let mut out = Array1::zeros(raw.len());

        for j in 0..raw.len() {
            let m_j = raw[j];
            let sign = match &self.clip_config.sign_mode {
                SignUpdateMode::Naive => {
                    if m_j > 0.0 {
                        1.0
                    } else if m_j < 0.0 {
                        -1.0
                    } else if self.m_tilde_prev[j] >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                SignUpdateMode::Hysteresis => {
                    if m_j > min_llr {
                        self.hysteresis_latch[j] = 1;
                    } else if m_j < -min_llr {
                        self.hysteresis_latch[j] = -1;
                    }
                    self.hysteresis_latch[j] as f64
                }
                SignUpdateMode::NeighborOnly => {
                    let raw_j = m_j + self.current_gammas[j] * (prior_llr[j] - self.m_tilde_prev[j]);
                    if raw_j >= 0.0 { 1.0 } else { -1.0 }
                }
            };

            out[j] = sign * m_j.abs().max(min_llr);
        }

        out
    }

    fn run_leg(
        &mut self,
        detectors: ArrayView1<Bit>,
        max_iter: usize,
        prior_llr: &Array1<f64>,
    ) -> DecodeResult {
        self.hysteresis_latch.fill(1_i8);

        let mut success = false;
        let mut decoded_detectors = Array1::default(detectors.dim());

        for _ in 0..max_iter {
            self.bp_decoder.run_iteration(detectors.view());
            self.bp_decoder.current_iteration += 1;

            decoded_detectors = self.bp_decoder.compute_decoded_detectors();
            success = self
                .bp_decoder
                .check_convergence(detectors.view(), decoded_detectors.view());

            let result =
                self.bp_decoder
                    .build_result(success, decoded_detectors.clone(), max_iter);
            let m_tilde = self.compute_clipped(&result.posterior_ratios, prior_llr);

            self.bp_decoder.set_posterior_ratios_f64(m_tilde.clone());
            self.m_tilde_prev = m_tilde;

            if success {
                break;
            }
        }

        self.bp_decoder
            .build_result(success, decoded_detectors, max_iter)
    }
}


impl Decoder for ClipBpDecoder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        Arc::clone(&self.check_matrix)
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.bp_decoder.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let n = self.check_matrix.cols();
        let prior_llr = self.min_sum_config.log_prior_ratios();
        let max_legs = self.clip_config.max_legs.max(1);

        let mut leg_success = Vec::with_capacity(max_legs);
        let mut leg_iterations = Vec::with_capacity(max_legs);
        let mut leg_negative_llr_counts = Vec::with_capacity(max_legs);
        let mut leg_decodings = Vec::with_capacity(max_legs);
        let mut leg_posteriors = Vec::with_capacity(max_legs);

        let mut total_iterations = 0_usize;
        let mut num_solutions = 0_usize;
        let mut min_pm = f64::MAX;
        let mut best_result: Option<DecodeResult> = None;

        self.bp_decoder.initialize_decoder();
        self.bp_decoder
            .set_memory_strengths_f64(Array1::from_elem(n, self.clip_config.gamma_first));
        self.current_gammas.fill(self.clip_config.gamma_first);
        self.m_tilde_prev = prior_llr.clone();

        let first_result =
            self.run_leg(detectors.view(), self.clip_config.max_iter_first, &prior_llr);
        total_iterations += first_result.iterations;

        if first_result.success {
            num_solutions += 1;
            let pm = first_result.decoding_quality;
            if pm < min_pm {
                min_pm = pm;
                best_result = Some(first_result.clone());
            }
        }

        leg_success.push(first_result.success);
        leg_iterations.push(first_result.iterations);
        leg_negative_llr_counts.push(first_result.posterior_ratios.iter().filter(|&&x| x < 0.0).count());
        leg_decodings.push(first_result.decoding.clone());
        leg_posteriors.push(first_result.posterior_ratios.clone());
        let mut last_result = first_result.clone();

        for _leg in 1..max_legs {
            if num_solutions >= self.clip_config.max_solutions {
                break;
            }

            let gammas = self.sample_gammas();
            self.current_gammas.assign(&gammas);
            self.bp_decoder.set_memory_strengths_f64(gammas);
            self.bp_decoder.current_iteration = 0;
            self.bp_decoder.initialize_check_to_variable();
            self.bp_decoder.initialize_variable_to_check();

            let result = self.run_leg(detectors.view(), self.clip_config.max_iter, &prior_llr);
            total_iterations += result.iterations;

            if result.success {
                num_solutions += 1;
                let pm = result.decoding_quality;
                if pm < min_pm {
                    min_pm = pm;
                    best_result = Some(result.clone());
                }
            }

            leg_success.push(result.success);
            leg_iterations.push(result.iterations);
            leg_negative_llr_counts.push(result.posterior_ratios.iter().filter(|&&x| x < 0.0).count());
            leg_decodings.push(result.decoding.clone());
            leg_posteriors.push(result.posterior_ratios.clone());
            last_result = result;
        }

        let mut final_result = best_result.unwrap_or(last_result);

        final_result.iterations = total_iterations;
        final_result.extra = BPExtraResult::RelayTrace {
            leg_success,
            leg_iterations,
            leg_negative_llr_counts,
            leg_decodings,
            leg_posteriors,
            relay_parallel_avg_iter_seconds: None,
            dyn_phase_avg_iter_seconds: None,
        };
        final_result
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        self.bp_decoder.get_decoding_quality(errors)
    }
}

impl DecoderRunner for ClipBpDecoder {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::bipartite_graph::{BipartiteGraph, SparseBipartiteGraph};
    use ndarray::array;

    fn build_decoder(sign_mode: SignUpdateMode) -> ClipBpDecoder {
        let check_matrix = array![[1_u8, 1, 0], [0, 1, 1],];
        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);

        let min_sum_config = Arc::new(MinSumDecoderConfig {
            error_priors: array![0.05, 0.05, 0.05],
            max_iter: 12,
            alpha: Some(1.0),
            alpha_iteration_scaling_factor: 1.0,
            gamma0: Some(0.125),
            ..Default::default()
        });

        let clip_config = Arc::new(ClipBpConfig {
            min_llr: 0.3,
            sign_mode,
            gamma_first: 0.125,
            gamma_center: 0.21,
            gamma_width: 0.9,
            max_legs: 3,
            max_iter_first: 8,
            max_iter: 6,
            max_solutions: 1,
            seed: 0,
        });

        ClipBpDecoder::new(Arc::new(check_matrix), min_sum_config, clip_config)
    }

    #[test]
    fn zero_syndrome_converges_for_all_sign_modes() {
        let detectors = array![0_u8, 0_u8];

        for mode in [
            SignUpdateMode::Naive,
            SignUpdateMode::Hysteresis,
            SignUpdateMode::NeighborOnly,
        ] {
            let mut decoder = build_decoder(mode);
            let result = decoder.decode_detailed(detectors.view());

            assert!(result.success);
            assert_eq!(result.decoded_detectors, detectors);
            assert_eq!(result.decoding.len(), 3);
        }
    }
}