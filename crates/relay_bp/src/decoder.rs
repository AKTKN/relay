// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::bipartite_graph::SparseBipartiteGraph;
use ndarray::{stack, Array1, Array2, ArrayView1, ArrayView2, Axis};

use indicatif::{ParallelProgressIterator, ProgressFinish, ProgressIterator, ProgressStyle};
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub type Bit = u8;
pub type SparseBitMatrix = SparseBipartiteGraph<Bit>;

use dyn_clone::DynClone;

use std::sync::Arc;
use crate::bp::slg_mbp::trace::{SLGMBPDetailedDynamicsTrace, SLGMBPScoreSpikeTrace};

pub trait Mod2Mul<Rhs = Self> {
    type Output;

    // Required method
    fn mul_mod2(&self, rhs: Rhs) -> Self::Output;
}

impl Mod2Mul<&Array1<Bit>> for SparseBitMatrix {
    type Output = Array1<Bit>;

    fn mul_mod2(&self, rhs: &Array1<Bit>) -> Self::Output {
        let mut detectors_u8 = self * rhs;
        detectors_u8.map_inplace(|x| *x %= 2);
        detectors_u8
    }
}

pub trait Decoder: DynClone + Sync {
    fn check_matrix(&self) -> Arc<SparseBitMatrix>;
    fn log_prior_ratios(&mut self) -> Array1<f64>;
    /// Decode a single input problem
    fn decode(&mut self, detectors: ArrayView1<Bit>) -> Array1<Bit> {
        self.decode_detailed(detectors).decoding
    }
    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult;
    fn decode_detailed_dynamics(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        self.decode_detailed(detectors)
    }
    fn decode_batch(&mut self, detectors: ArrayView2<Bit>) -> Array2<Bit> {
        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .map(|row| self.decode(row))
            .collect();

        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }
    fn decode_detailed_batch(&mut self, detectors: ArrayView2<Bit>) -> Vec<DecodeResult> {
        detectors
            .axis_iter(Axis(0))
            .map(|row| self.decode_detailed(row))
            .collect()
    }
    // Compute detectors from errors
    fn get_detectors(&self, errors: ArrayView1<Bit>) -> Array1<Bit> {
        self.check_matrix().mul_mod2(&errors.to_owned())
    }

    fn get_detectors_batch(&self, errors: ArrayView2<Bit>) -> Array2<Bit> {
        let check_matrix = self.check_matrix();
        let detectors: Vec<Array1<Bit>> = errors
            .axis_iter(Axis(0))
            .map(|row| check_matrix.mul_mod2(&row.to_owned()))
            .collect();
        stack(
            Axis(0),
            &detectors.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn get_decoding_quality(&mut self, errors: ArrayView1<u8>) -> f64 {
        let log_prior_ratios = self.log_prior_ratios();
        let mut decoding_quality: f64 = 0.0;
        for i in 0..errors.len() {
            if errors[i] == 1 && f64::is_finite(log_prior_ratios[i]) {
                decoding_quality += log_prior_ratios[i];
            }
        }
        decoding_quality
    }
}

dyn_clone::clone_trait_object!(Decoder);

pub trait DecoderRunner: Decoder + Clone + Sync {
    fn par_decode_batch(&mut self, detectors: ArrayView2<Bit>) -> Array2<Bit> {
        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .map_with(|| self.clone(), |decoder, row| decoder().decode(row))
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    fn par_decode_detailed_batch(&mut self, detectors: ArrayView2<Bit>) -> Vec<DecodeResult> {
        detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .map_with(
                || self.clone(),
                |decoder, row| decoder().decode_detailed(row),
            )
            .collect()
    }

    /// Decode a batch displaying a progress bar
    fn decode_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Array2<Bit> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map(|row| self.decode(row))
            .collect();
        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    /// Decode a batch displaying a progress bar
    fn decode_detailed_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<DecodeResult> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        detectors
            .axis_iter(Axis(0))
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map(|row| self.decode_detailed(row))
            .collect()
    }

    fn par_decode_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Array2<Bit> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        let arrs: Vec<Array1<Bit>> = detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map_with(|| self.clone(), |decoder, row| decoder().decode(row))
            .collect();

        stack(Axis(0), &arrs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    fn par_decode_detailed_batch_progress_bar(
        &mut self,
        detectors: ArrayView2<Bit>,
        leave_progress_bar_on_finish: bool,
    ) -> Vec<DecodeResult> {
        let finish_mode = match leave_progress_bar_on_finish {
            true => ProgressFinish::AndLeave,
            false => ProgressFinish::AndClear,
        };

        detectors
            .axis_iter(Axis(0))
            .into_par_iter()
            .progress_with_style(self.get_progress_bar_style())
            .with_finish(finish_mode)
            .map_with(
                || self.clone(),
                |decoder, row| decoder().decode_detailed(row),
            )
            .collect()
    }

    fn get_progress_bar_style(&self) -> ProgressStyle {
        ProgressStyle::default_bar().template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({per_sec}, {eta})").unwrap()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DecodeResult {
    pub decoding: Array1<Bit>,
    pub decoded_detectors: Array1<Bit>,
    pub posterior_ratios: Array1<f64>,
    pub success: bool,
    pub decoding_quality: f64,
    pub iterations: usize,
    pub max_iter: usize,
    pub extra: BPExtraResult,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BPExtraResult {
    None,
    LBFTrace {
        converged: bool,
    },
    DisorderedBPTrace {
        leg_success: Vec<bool>,
        leg_iterations: Vec<usize>,
        leg_negative_llr_counts: Vec<usize>,
        leg_decodings: Vec<Array1<Bit>>,
        leg_posteriors: Vec<Array1<f64>>,
        leg_alpha_means: Vec<f64>,
        leg_gamma_means: Vec<f64>,
        leg_bias_means: Vec<f64>,
        leg_bias_applied_counts: Vec<usize>,
    },
    RelayTrace {
        leg_success: Vec<bool>,
        leg_iterations: Vec<usize>,
        leg_negative_llr_counts: Vec<usize>,
        leg_decodings: Vec<Array1<Bit>>,
        leg_posteriors: Vec<Array1<f64>>,
        relay_parallel_avg_iter_seconds: Option<f64>,
        dyn_phase_avg_iter_seconds: Option<f64>,
    },
    AdaptiveRelayTrace {
        executed_members: usize,
        winner_member_index: Option<usize>,
        member_success: Vec<bool>,
        member_discovery_iterations: Vec<Option<usize>>,
        member_total_iterations: Vec<usize>,
        leg_iterations: Vec<usize>,
        gamma_history: Vec<f64>,
        clamp_applied_count: usize,
    },
    SLGMBPTrace {
        phase1_converged: bool,
        phase1_iterations: usize,
        total_iterations: usize,
        generation_count: usize,
        generation_best_fitness: Vec<f64>,
        selected_solution_posterior: Option<Array1<f64>>,
        accepted_solution_decodings: Vec<Array1<Bit>>,
        accepted_solution_weights: Vec<f64>,
        accepted_solution_discovery_iterations: Vec<usize>,
        residual_weight_history: Vec<usize>,
        gamma_history: Vec<f64>,
        detailed_dynamics: Option<SLGMBPDetailedDynamicsTrace>,
        score_spike_trace: Option<SLGMBPScoreSpikeTrace>,
    },
    DualRelayTrace {
        leg_success: Vec<bool>,
        leg_iterations: Vec<usize>,
        leg_negative_llr_counts: Vec<usize>,
        leg_decodings: Vec<Array1<Bit>>,
        leg_posteriors: Vec<Array1<f64>>,
        slow_success_count: usize,
        fast_success_count: usize,
        joint_success_count: usize,
        naive_average_applied_count: usize,
        weighted_fast_disagreement_applied_count: usize,
        unique_solution_count: usize,
    },
}
