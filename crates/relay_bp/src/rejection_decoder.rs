// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bp::min_sum::MinSumDecoderConfig;
use crate::bp::relay::{RelayDecoder, RelayDecoderConfig};
use crate::decoder::{Bit, BPExtraResult, DecodeResult, Decoder, Mod2Mul, RejectionExtraResult, SparseBitMatrix};

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub enum ReweightingMode {
    Full,
    Partial,
}

impl Default for ReweightingMode {
    fn default() -> Self {
        ReweightingMode::Full
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReweightingSelection {
    Random,
    Prior,
}

impl Default for ReweightingSelection {
    fn default() -> Self {
        ReweightingSelection::Random
    }
}

#[derive(Clone, Debug)]
pub struct RejectionConfig {
    pub mode: ReweightingMode,
    pub selection: ReweightingSelection,
    pub k: f64,
    pub b: f64,
    pub seed: u64,
}

impl Default for RejectionConfig {
    fn default() -> Self {
        Self {
            mode: ReweightingMode::Full,
            selection: ReweightingSelection::Random,
            k: 0.5,
            b: 2.0,
            seed: 0,
        }
    }
}

#[derive(Clone)]
pub struct RejectionDecoder {
    check_matrix: Arc<SparseBitMatrix>,
    observable_matrix: Arc<SparseBitMatrix>,
    min_sum_config: Arc<MinSumDecoderConfig>,
    relay_config: Arc<RelayDecoderConfig>,
    original_priors: Arc<Array1<f64>>,
    original_log_priors: Arc<Array1<f64>>,
    config: RejectionConfig,
    rng: StdRng,
}

impl RejectionDecoder {
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        observable_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        relay_config: Arc<RelayDecoderConfig>,
        rejection_config: RejectionConfig,
    ) -> Self {
        let original_priors = min_sum_config.error_priors.clone();
        let original_log_priors = min_sum_config.log_prior_ratios();
        let seed = if rejection_config.seed == 0 {
            rand::random::<u64>()
        } else {
            rejection_config.seed
        };
        let rng = StdRng::seed_from_u64(seed);
        Self {
            check_matrix,
            observable_matrix,
            min_sum_config,
            relay_config,
            original_priors: Arc::new(original_priors),
            original_log_priors: Arc::new(original_log_priors),
            config: RejectionConfig { seed, ..rejection_config },
            rng,
        }
    }

    fn build_decoder_with_priors(&self, priors: Array1<f64>) -> RelayDecoder<f64> {
        let mut min_sum_config = (*self.min_sum_config).clone();
        min_sum_config.error_priors = priors;
        let min_sum_config = Arc::new(min_sum_config);
        RelayDecoder::new(self.check_matrix.clone(), min_sum_config, self.relay_config.clone())
    }

    fn compute_weight(&self, decoding: &Array1<Bit>) -> f64 {
        decoding
            .iter()
            .zip(self.original_log_priors.iter())
            .filter(|(&bit, &llr)| bit == 1 && llr.is_finite())
            .map(|(_, &llr)| llr)
            .sum::<f64>()
    }

    fn same_coset(&self, a: &Array1<Bit>, b: &Array1<Bit>) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| x == y)
    }

    fn select_reweight_indices(&mut self, decoding: &Array1<Bit>) -> Vec<usize> {
        let mut indices: Vec<usize> = decoding
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| if val == 1 { Some(idx) } else { None })
            .collect();

        if indices.is_empty() {
            return indices;
        }

        match self.config.mode {
            ReweightingMode::Full => indices,
            ReweightingMode::Partial => {
                let k = self.config.k.clamp(0.0, 1.0);
                let mut count = (k * indices.len() as f64).ceil() as usize;
                count = count.clamp(1, indices.len());

                match self.config.selection {
                    ReweightingSelection::Random => {
                        indices.shuffle(&mut self.rng);
                        indices.truncate(count);
                    }
                    ReweightingSelection::Prior => {
                        let priors = self.original_priors.clone();
                        indices.sort_by(|&a, &b| {
                            let pa = priors[a];
                            let pb = priors[b];
                            let cmp = pb.partial_cmp(&pa).unwrap_or(Ordering::Equal);
                            if cmp == Ordering::Equal {
                                a.cmp(&b)
                            } else {
                                cmp
                            }
                        });
                        indices.truncate(count);
                    }
                }
                indices
            }
        }
    }

    fn apply_reweighting(&mut self, decoding: &Array1<Bit>) -> Array1<f64> {
        let mut priors = (*self.original_priors).clone();
        let indices = self.select_reweight_indices(decoding);

        if indices.is_empty() {
            return priors;
        }

        let b = if self.config.b.is_finite() && self.config.b > 0.0 {
            self.config.b
        } else {
            1.0
        };
        for idx in indices {
            let p = priors[idx];
            let p_new = p.powf(b).clamp(1e-15, 1.0 - 1e-15);
            priors[idx] = p_new;
        }
        priors
    }
}

impl Decoder for RejectionDecoder {
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.check_matrix.clone()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        (*self.original_log_priors).clone()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let mut decoder_stage1 = self.build_decoder_with_priors((*self.original_priors).clone());
        let mut result = decoder_stage1.decode_detailed(detectors);

        let c1 = result.decoding.clone();
        let coset1 = self.observable_matrix.mul_mod2(&c1);
        let weight1 = self.compute_weight(&c1);

        let mut r2_iter = f64::NAN;
        let mut r3_iter = f64::NAN;
        let mut r2_class = None;
        let mut r3_class = None;
        let mut reject = 0u8;
        let mut rejection_gap = f64::NAN;

        let priors_stage2 = self.apply_reweighting(&c1);
        let mut decoder_stage2 = self.build_decoder_with_priors(priors_stage2);
        let result2 = decoder_stage2.decode_detailed(detectors);
        let c2 = result2.decoding.clone();
        let coset2 = self.observable_matrix.mul_mod2(&c2);

        r2_iter = if result2.success {
            result2.iterations as f64
        } else {
            f64::INFINITY
        };

        let r2_same = self.same_coset(&coset1, &coset2);
        r2_class = Some(r2_same);

        if !r2_same {
            reject = 2;
            let weight2 = self.compute_weight(&c2);
            rejection_gap = weight2 - weight1;
        } else {
            let priors_stage3 = self.apply_reweighting(&c2);
            let mut decoder_stage3 = self.build_decoder_with_priors(priors_stage3);
            let result3 = decoder_stage3.decode_detailed(detectors);
            let c3 = result3.decoding.clone();
            let coset3 = self.observable_matrix.mul_mod2(&c3);

            r3_iter = if result3.success {
                result3.iterations as f64
            } else {
                f64::INFINITY
            };

            let r3_same = self.same_coset(&coset2, &coset3);
            r3_class = Some(r3_same);
            if !r3_same {
                reject = 3;
                let weight3 = self.compute_weight(&c3);
                rejection_gap = weight3 - weight1;
            }
        }

        result.extra = BPExtraResult::Rejection(RejectionExtraResult {
            r2_iter: Some(r2_iter),
            r3_iter: Some(r3_iter),
            r2_class,
            r3_class,
            reject: Some(reject),
            rejection_gap: Some(rejection_gap),
        });

        result
    }
}
