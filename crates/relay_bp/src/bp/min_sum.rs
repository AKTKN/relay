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
use crate::decoder::{BPExtraResult, DecodeResult, Decoder, DecoderRunner};
use crate::decoder::{Bit, SparseBitMatrix};
use itertools::izip;
use log::debug;
use ndarray::{Array1, ArrayView1};
use num_traits::FromPrimitive;
use num_traits::{Bounded, Signed, ToPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sprs::CsMatView;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Step1Metrics {
    pub variable_count: usize,
    pub w_sigma: Vec<u32>,
    pub delta_hd: Vec<u32>,
    pub delta_m_l2: Vec<f64>,
    pub m_norm_l2: Vec<f64>,
    pub m_bar: Vec<f64>,
    pub f_low_0: Vec<f64>,
    pub f_low_1: Vec<f64>,
    pub f_endpoint: Vec<u32>,
    pub v_osc: Vec<u32>,
    pub t_stag: Option<usize>,
    /// Packed sign bits (1 when M<0, else 0), shape (iterations, ceil(n_vars/8)).
    /// Bit-order within each byte is big-endian (same as numpy.packbits default).
    pub sign_trajectory_packed: Option<Vec<u8>>,
    pub sign_bytes_per_iter: usize,
    /// Absolute-marginal snapshots, flattened shape (n_snap, n_vars), row-major.
    pub abs_m_snapshots: Option<Vec<f32>>,
    pub snapshot_times: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct MinSumDecoderConfig {
    pub error_priors: Array1<f64>,
    pub max_iter: usize,
    pub alpha: Option<f64>,
    pub alpha_iteration_scaling_factor: f64,
    pub gamma0: Option<f64>,
    pub data_scale_value: Option<f64>,
    pub max_data_value: Option<f64>,
    pub int_bits: Option<isize>,
    pub frac_bits: Option<isize>,
    pub enable_variable_message_drop: bool,
    pub drop_probability: f64,
    pub drop_llr_threshold: f64,
    pub rng_seed: Option<u64>,
}

impl Default for MinSumDecoderConfig {
    fn default() -> Self {
        Self {
            error_priors: Default::default(),
            max_iter: 200,
            alpha: None,
            alpha_iteration_scaling_factor: 1.,
            gamma0: None,
            data_scale_value: None,
            max_data_value: None,
            int_bits: None,
            frac_bits: None,
            enable_variable_message_drop: false,
            drop_probability: 0.0,
            drop_llr_threshold: 0.0,
            rng_seed: None,
        }
    }
}

impl MinSumDecoderConfig {
    pub fn prior_ratios(&self) -> Array1<f64> {
        // A funky way of making (1-p)/p handle left to right type inference for arithmetic
        (1.0 - &self.error_priors) / &self.error_priors
    }

    pub fn log_prior_ratios(&self) -> Array1<f64> {
        self.prior_ratios().ln()
    }

    pub fn set_max_iter(&mut self, iterations: usize) {
        self.max_iter = iterations;
    }

    pub fn set_fixed(&mut self, int_bits: isize, frac_bits: isize) {
        self.int_bits = Some(int_bits);
        self.frac_bits = Some(frac_bits);
        self.max_data_value = Some((1 << (int_bits - 1)) as f64);
    }
}

/// A fast min-sum implementation of BP implemented internally
/// using a sparse bipartite graph.
#[derive(Clone)]
pub struct MinSumBPDecoder<N: PartialEq + Default + Clone + Copy> {
    check_matrix: Arc<SparseBitMatrix>,
    pub config: Arc<MinSumDecoderConfig>,
    log_prior_ratios: Array1<N>,
    check_to_variable: SparseBipartiteGraph<N>,
    variable_to_check: SparseBipartiteGraph<N>,
    // A cache of check to variable data mappings.
    check_to_variable_nnz_map: Vec<usize>,
    // A cache of variable to check data mappings.
    variable_to_check_nnz_map: Vec<usize>,
    posterior_ratios: Array1<N>,
    memory_strengths: Array1<N>,
    variable_alphas: Option<Array1<N>>,
    decoding: Array1<Bit>,
    max_data_value: Option<N>,
    data_scale_value: Option<N>,
    message_drop_rng: Option<StdRng>,
    pub current_iteration: usize,
}

impl<N> MinSumBPDecoder<N>
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
        config: Arc<MinSumDecoderConfig>,
    ) -> MinSumBPDecoder<N> {
        let check_to_variable = MinSumBPDecoder::build_check_to_variable(check_matrix.clone());
        let variable_to_check = MinSumBPDecoder::build_variable_to_check(check_matrix.clone());
        let (check_to_variable_nnz_map, variable_to_check_nnz_map) =
            MinSumBPDecoder::build_nnz_maps(check_to_variable.view(), variable_to_check.view());

        let max_data_value = match config.max_data_value {
            Some(val) => N::from_f64(val),
            None => None,
        };

        let data_scale_value = match config.data_scale_value {
            Some(val) => N::from_f64(val),
            None => None,
        };

        let log_prior_ratios = config.log_prior_ratios().mapv_into_any(|val| {
            let updated_val = match val {
                f64::INFINITY => N::max_value(),
                _ => {
                    // Apply optional data scaling
                    let prior = match config.data_scale_value {
                        Some(scale_value) => scale_value * val,
                        None => val,
                    };
                    N::from_f64(prior).unwrap()
                }
            };

            // Bound prior values if necessary
            match max_data_value {
                Some(max_val) => Self::bound_value_magnitude(updated_val, max_val),
                None => updated_val,
            }
        });

        let memory_strengths = Array1::from_elem(check_matrix.cols(), N::zero());

        let posterior_ratios = if config.gamma0.is_some() {
            log_prior_ratios.clone()
        } else {
            Array1::zeros(check_matrix.cols())
        };

        let decoding = Array1::zeros(check_matrix.cols());
        let message_drop_rng = if config.enable_variable_message_drop && config.drop_probability > 0.0 {
            config.rng_seed.map(StdRng::seed_from_u64)
        } else {
            None
        };

        MinSumBPDecoder::<N> {
            check_matrix,
            config,
            log_prior_ratios,
            check_to_variable,
            variable_to_check,
            check_to_variable_nnz_map,
            variable_to_check_nnz_map,
            posterior_ratios,
            memory_strengths,
            variable_alphas: None,
            decoding,
            max_data_value,
            data_scale_value,
            message_drop_rng,
            current_iteration: 0,
        }
    }

    pub fn set_log_prior_ratio(&mut self, mut log_prior_ratios: Array1<N>) {
        self.log_prior_ratios = match self.data_scale_value {
            Some(scale_val) => {
                log_prior_ratios.iter_mut().for_each(|v| *v *= scale_val);
                log_prior_ratios
            }
            None => log_prior_ratios,
        };
    }

    pub fn set_log_prior_ratio_f64(&mut self, log_prior_ratios: Array1<f64>) {
        self.log_prior_ratios = match self.config.data_scale_value {
            Some(scale_val) => {
                log_prior_ratios.mapv_into_any(|v| N::from_f64(scale_val * v).unwrap())
            }
            None => log_prior_ratios.mapv_into_any(|v| N::from_f64(v).unwrap()),
        };
    }

    pub fn set_posterior_ratios_to_priors(&mut self) {
        self.posterior_ratios = self.log_prior_ratios.clone();
    }

    pub fn set_posterior_ratios_f64(&mut self, posterior_ratios: Array1<f64>) {
        self.posterior_ratios = match self.config.data_scale_value {
            Some(scale_val) => {
                posterior_ratios.mapv_into_any(|v| N::from_f64(scale_val * v).unwrap())
            }
            None => posterior_ratios.mapv_into_any(|v| N::from_f64(v).unwrap()),
        };
    }

    /// Predict posterior marginals from the current check-to-variable messages,
    /// using externally supplied previous marginals and memory strengths.
    pub fn predict_posterior_from_previous_and_memory_f64(
        &self,
        previous_posterior: &Array1<f64>,
        memory_strengths: &Array1<f64>,
        previous_check_message_sum: Option<&Array1<f64>>,
        previous_check_message_beta: f64,
    ) -> Array1<f64> {
        let variable_count = self.check_to_variable.outer_dims();
        let prior_llr = self.config.log_prior_ratios();
        let mut predicted = Array1::<f64>::zeros(variable_count);

        for var_idx in 0..variable_count {
            let gamma = memory_strengths[var_idx];
            let variable_prior =
                (1.0 - gamma) * prior_llr[var_idx] + gamma * previous_posterior[var_idx];

            let check_sum = self
                .check_to_variable
                .outer_view(var_idx)
                .map(|col_vec| {
                    col_vec
                        .iter()
                        .map(|(_, val)| {
                            let msg = N::to_f64(val).unwrap_or(0.0);
                            match self.config.data_scale_value {
                                Some(scale_val) => msg / scale_val,
                                None => msg,
                            }
                        })
                        .sum::<f64>()
                })
                .unwrap_or(0.0);

            let previous_check_sum = previous_check_message_sum
                .map(|sums| sums[var_idx])
                .unwrap_or(0.0);

            predicted[var_idx] =
                variable_prior + check_sum + previous_check_message_beta * previous_check_sum;
        }

        predicted
    }

    /// Return per-variable sums of current check-to-variable messages in f64 scale.
    pub fn sum_check_to_variable_by_variable_f64(&self) -> Array1<f64> {
        let variable_count = self.check_to_variable.outer_dims();
        let mut sums = Array1::<f64>::zeros(variable_count);

        for var_idx in 0..variable_count {
            sums[var_idx] = self
                .check_to_variable
                .outer_view(var_idx)
                .map(|col_vec| {
                    col_vec
                        .iter()
                        .map(|(_, val)| {
                            let msg = N::to_f64(val).unwrap_or(0.0);
                            match self.config.data_scale_value {
                                Some(scale_val) => msg / scale_val,
                                None => msg,
                            }
                        })
                        .sum::<f64>()
                })
                .unwrap_or(0.0);
        }

        sums
    }

    /// Apply externally selected posterior marginals and rebuild a single
    /// variable-to-check message state from them.
    pub fn set_posterior_and_rebuild_variable_to_check_f64(&mut self, posterior: Array1<f64>) {
        self.set_posterior_ratios_f64(posterior);

        for var_idx in 0..self.check_to_variable.outer_dims() {
            let data_range = self.check_to_variable.indptr().outer_inds(var_idx);
            for (ind, check_to_var_msg) in izip!(
                data_range.clone(),
                &self.check_to_variable.data()[data_range.clone()]
            ) {
                let map_ind = self.check_to_variable_nnz_map[ind];
                self.variable_to_check.data_mut()[map_ind] =
                    self.posterior_ratios[var_idx] + check_to_var_msg.neg();
            }
        }

        self.bound_magnitudes();
        self.compute_hard_decision();
    }

    /// Set external memory strengths from f64. Applies scaling if needed.
    pub fn set_memory_strengths_f64(&mut self, memory_strengths: Array1<f64>) {
        self.memory_strengths = match self.config.data_scale_value {
            Some(scale_val) => {
                memory_strengths.mapv_into_any(|v| N::from_f64(scale_val * v).unwrap())
            }
            None => memory_strengths.mapv_into_any(|v| N::from_f64(v).unwrap()),
        };
    }

    /// Set external memory strengths from N. Applies scaling if needed.
    pub fn set_memory_strengths(&mut self, mut memory_strengths: Array1<N>) {
        self.memory_strengths = match self.data_scale_value {
            Some(scale_val) => {
                memory_strengths.iter_mut().for_each(|v| *v *= scale_val);
                memory_strengths
            }
            None => memory_strengths,
        };
    }

    /// Set optional per-variable alpha values used in check-to-variable updates.
    /// When None, the scalar alpha schedule from config is used.
    pub fn set_variable_alphas_f64(&mut self, variable_alphas: Option<Array1<f64>>) {
        self.variable_alphas = variable_alphas.map(|alphas| match self.config.data_scale_value {
            Some(scale_val) => alphas.mapv_into_any(|v| N::from_f64(scale_val * v).unwrap()),
            None => alphas.mapv_into_any(|v| N::from_f64(v).unwrap()),
        });
    }

    pub fn clear_variable_alphas(&mut self) {
        self.variable_alphas = None;
    }

    // Construct a new check message graph
    fn build_check_to_variable(check_matrix: Arc<SparseBitMatrix>) -> SparseBipartiteGraph<N> {
        let check_matrix_csc = check_matrix.to_csc();

        let default_messages: Vec<_> = vec![N::default(); check_matrix_csc.nnz()];

        SparseBipartiteGraph::new_csc(
            check_matrix_csc.shape(),
            check_matrix_csc.indptr().raw_storage().to_vec(),
            check_matrix_csc.indices().to_vec(),
            default_messages,
        )
    }

    // Construct a new variable message graph
    fn build_variable_to_check(check_matrix: Arc<SparseBitMatrix>) -> SparseBipartiteGraph<N> {
        let check_matrix_csr = check_matrix.to_csr();

        let default_messages: Vec<_> = vec![N::default(); check_matrix_csr.nnz()];

        SparseBipartiteGraph::new(
            check_matrix_csr.shape(),
            check_matrix_csr.indptr().raw_storage().to_vec(),
            check_matrix_csr.indices().to_vec(),
            default_messages,
        )
    }

    fn build_nnz_maps(
        check_to_variable: CsMatView<N>,
        variable_to_check: CsMatView<N>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut check_to_variable_nnz_map: Vec<usize> = Vec::with_capacity(check_to_variable.nnz());
        for (_, (row, col)) in check_to_variable.view().iter() {
            check_to_variable_nnz_map.push(variable_to_check.nnz_index(row, col).unwrap().0);
        }
        let mut variable_to_check_nnz_map: Vec<usize> = Vec::with_capacity(variable_to_check.nnz());
        for (_, (row, col)) in variable_to_check.view().iter() {
            variable_to_check_nnz_map.push(check_to_variable.nnz_index(row, col).unwrap().0);
        }
        (check_to_variable_nnz_map, variable_to_check_nnz_map)
    }

    /// Initialize variable message state to the prior
    pub fn initialize_variable_to_check(&mut self) {
        for mut row_vec in self.variable_to_check.outer_iterator_mut() {
            row_vec
                .iter_mut()
                .for_each(|(col_ind, val)| *val = self.log_prior_ratios[col_ind]);
        }
    }

    pub fn initialize_check_to_variable(&mut self) {
        for mut col_vec in self.check_to_variable.outer_iterator_mut() {
            col_vec
                .iter_mut()
                .for_each(|(_row_ind, val)| *val = N::zero());
        }
    }

    pub fn initialize_memory_strengths(&mut self) {
        // Initialize Mem-BP and posteriors
        let ewa_factor_float = self.config.gamma0.unwrap_or(0.);
        let ewa_factor = N::from_f64(match self.config.data_scale_value {
            Some(scale_value) => scale_value * ewa_factor_float,
            None => ewa_factor_float,
        })
        .unwrap();
        self.memory_strengths.fill(ewa_factor);
    }

    pub fn initialize_decoder(&mut self) {
        self.current_iteration = 0;
        self.initialize_memory_strengths();
        self.initialize_check_to_variable();
        self.initialize_variable_to_check();
        // Initialize posteriors if needed for mem-BP
        if self.config.gamma0.is_some() {
            self.set_posterior_ratios_to_priors();
        };
    }

    fn alpha(&self) -> N {
        let mut alpha = match self.config.alpha {
            Some(0.) => {
                let iteration = (self.current_iteration + 1) as f64;
                1.0 - (2_f64).powf(-(iteration / self.config.alpha_iteration_scaling_factor))
            }
            Some(val) => val,
            None => 1.0,
        };

        // Handle case of alpha < 0 defaulting to 1.
        // This aligns with integer case of ldpc-simulation
        if alpha < 0. {
            alpha = 1.
        }
        // Scale if needed
        alpha = match self.config.data_scale_value {
            Some(scale_val) => scale_val * alpha,
            None => alpha,
        };

        N::from_f64(alpha).unwrap()
    }

    /// Compute check to bit message iteration
    fn compute_check_to_variable(
        &mut self,
        detectors: ArrayView1<Bit>,
    ) -> &mut SparseBipartiteGraph<N> {
        let alpha = self.alpha();
        let variable_alphas = self.variable_alphas.as_ref();

        for (var_check_row_ind, var_check_row_vec) in
            self.variable_to_check.outer_iterator().enumerate()
        {
            let row_sign = if detectors[var_check_row_ind] == 1 {
                N::one().neg()
            } else {
                N::one()
            };
            let mut accumulated_sign = row_sign.is_negative();
            let mut min_ind: usize = 0;
            // True min message
            let mut min_message = N::max_value();
            // Next lowest min message to be used for self-exlusive min value
            let mut second_min_message = N::max_value();

            for (var_check_col_ind, var_check_col_val) in var_check_row_vec.iter() {
                accumulated_sign ^= var_check_col_val.is_negative();
                let abs_msg = var_check_col_val.abs();
                if abs_msg <= min_message {
                    second_min_message = min_message;
                    min_message = abs_msg;
                    min_ind = var_check_col_ind;
                } else if abs_msg <= second_min_message {
                    second_min_message = abs_msg
                }
            }

            debug!("Variable messages for row {var_check_row_ind:?}: {var_check_row_vec:?}");

            // Iterate over the row's storage indices
            let data_range = self
                .variable_to_check
                .indptr()
                .outer_inds(var_check_row_ind);

            for (ind, var_check_col_ind, var_check_col_val) in izip!(
                data_range.clone(),
                &self.variable_to_check.indices()[data_range.clone()],
                &self.variable_to_check.data()[data_range.clone()]
            ) {
                // Extract the sign from the accumulated sign.
                let check_to_variable_sign = accumulated_sign ^ var_check_col_val.is_negative();
                let check_to_variable_min: N = if *var_check_col_ind != min_ind {
                    min_message
                } else {
                    second_min_message
                };
                let alpha_for_variable = match variable_alphas {
                    Some(alphas) => alphas[*var_check_col_ind],
                    None => alpha,
                };
                // Copy the sign to the variable. check_to_variable_min is guranteed to be positive.
                let mut check_to_variable = alpha_for_variable * check_to_variable_min;
                if check_to_variable_sign {
                    check_to_variable = check_to_variable.neg();
                }

                // We directly manipulate the indicies of the check_to_variable_matrix using
                // the cached value map to avoid the need for a logarithmic insert
                self.check_to_variable.data_mut()[self.variable_to_check_nnz_map[ind]] =
                    check_to_variable;
            }
        }

        if let Some(scale_val) = self.data_scale_value {
            self.check_to_variable /= scale_val
        }

        &mut self.check_to_variable
    }

    fn compute_variable_prior(&self, variable: usize) -> N {
        // Apply membp
        if self.config.gamma0.is_some() {
            if self.log_prior_ratios[variable] == N::max_value() {
                return self.log_prior_ratios[variable];
            }
            let scaled_one = self.data_scale_value.unwrap_or(N::one());
            // First divide through denominator before numerator to avoid overflow
            let prior_component = (self.log_prior_ratios[variable] / scaled_one)
                * (scaled_one - self.memory_strengths[variable]);
            let posterior_component =
                (self.posterior_ratios[variable] / scaled_one) * self.memory_strengths[variable];
            return prior_component + posterior_component;
        }
        self.log_prior_ratios[variable]
    }

    // Blend a BP marginal with the previous marginal using memory strength.
    // This implements: M_t = (1 - gamma) * M'_t + gamma * M_{t-1}.
    fn blend_marginal_with_memory(&self, bp_marginal: N, previous_marginal: N, variable: usize) -> N {
        if self.config.gamma0.is_none() {
            return bp_marginal;
        }

        if bp_marginal == N::max_value() {
            return bp_marginal;
        }

        let scaled_one = self.data_scale_value.unwrap_or(N::one());
        let gamma = self.memory_strengths[variable];
        let bp_component = (bp_marginal / scaled_one) * (scaled_one - gamma);
        let memory_component = (previous_marginal / scaled_one) * gamma;
        bp_component + memory_component
    }

    fn variable_message_drop_enabled(&self) -> bool {
        self.config.enable_variable_message_drop && self.config.drop_probability > 0.0
    }

    fn should_drop_variable_messages(&mut self, variable: usize) -> bool {
        if !self.variable_message_drop_enabled() {
            return false;
        }

        let llr_abs = self.posterior_ratio_f64(variable).abs();
        if !llr_abs.is_finite() || llr_abs < self.config.drop_llr_threshold.abs() {
            return false;
        }

        let Some(rng) = self.message_drop_rng.as_mut() else {
            return false;
        };

        rng.gen_bool(self.config.drop_probability.clamp(0.0, 1.0))
    }

    fn zero_variable_to_check_messages<I>(&mut self, indices: I)
    where
        I: IntoIterator<Item = usize>,
    {
        for ind in indices {
            let map_ind = self.check_to_variable_nnz_map[ind];
            self.variable_to_check.data_mut()[map_ind] = N::zero();
        }
    }

    /// Compute bit to check message iteration
    fn compute_variable_to_check(&mut self) -> &mut SparseBipartiteGraph<N> {
        for check_var_col_ind in 0..self.check_to_variable.outer_dims() {
            let drop_indices = {
                let check_var_col_vec = self.check_to_variable.outer_view(check_var_col_ind).unwrap();
                // Accumulate messages
                let mut check_to_var_row_sum = self.compute_variable_prior(check_var_col_ind);

                debug!("Check messages for col {check_var_col_ind:?}: {check_var_col_vec:?}");

                let data_range = self
                    .check_to_variable
                    .indptr()
                    .outer_inds(check_var_col_ind);
                let drop_indices: Vec<usize> = data_range.clone().collect();

                // Perform iteration in the forward direction to accumulate left to right
                for (ind, check_var_row_val) in izip!(
                    data_range.clone(),
                    &self.check_to_variable.data()[data_range.clone()]
                ) {
                    self.variable_to_check.data_mut()[self.check_to_variable_nnz_map[ind]] =
                        check_to_var_row_sum;
                    check_to_var_row_sum += *check_var_row_val;
                }

                self.posterior_ratios[check_var_col_ind] = check_to_var_row_sum;

                // Now perform iteration in the reverse direction to accumulate right to left
                check_to_var_row_sum = N::zero();
                // Remove each messages contribution
                for (ind, check_var_row_val) in izip!(
                    data_range.clone(),
                    &self.check_to_variable.data()[data_range.clone()]
                )
                .rev()
                {
                    let map_ind = self.check_to_variable_nnz_map[ind];
                    self.variable_to_check.data_mut()[map_ind] += check_to_var_row_sum;
                    check_to_var_row_sum += *check_var_row_val;

                    // We directly manipulate the indicies of the variable_to_check matrix using
                    // the cached value map to avoid the need for a logarithmic insert
                    debug!(
                        "location ({:?}, {:?}), variable_to_check: {:.32}",
                        self.check_to_variable.indices()[ind],
                        check_var_col_ind,
                        self.variable_to_check.data_mut()[self.check_to_variable_nnz_map[ind]]
                    );
                }

                drop_indices
            };

            if self.should_drop_variable_messages(check_var_col_ind) {
                self.zero_variable_to_check_messages(drop_indices);
            }
        }

        self.bound_magnitudes();

        &mut self.variable_to_check
    }

    /// Compute bit-to-check messages for LRBP legs.
    ///
    /// This first computes the standard BP marginal M'_t = prior + incoming check messages,
    /// then applies memory blending M_t = (1-gamma) * M'_t + gamma * M_{t-1}.
    fn compute_variable_to_check_lrbp(&mut self) -> &mut SparseBipartiteGraph<N> {
        for check_var_col_ind in 0..self.check_to_variable.outer_dims() {
            let drop_indices = {
                let check_var_col_vec = self.check_to_variable.outer_view(check_var_col_ind).unwrap();
                // Start from the plain BP prior (without pre-mixing with memory).
                let mut check_to_var_row_sum = self.log_prior_ratios[check_var_col_ind];

                debug!("Check messages for col {check_var_col_ind:?}: {check_var_col_vec:?}");

                let data_range = self
                    .check_to_variable
                    .indptr()
                    .outer_inds(check_var_col_ind);
                let drop_indices: Vec<usize> = data_range.clone().collect();

                // Perform iteration in the forward direction to accumulate left to right.
                for (ind, check_var_row_val) in izip!(
                    data_range.clone(),
                    &self.check_to_variable.data()[data_range.clone()]
                ) {
                    self.variable_to_check.data_mut()[self.check_to_variable_nnz_map[ind]] =
                        check_to_var_row_sum;
                    check_to_var_row_sum += *check_var_row_val;
                }

                let bp_marginal = check_to_var_row_sum;
                let previous_marginal = self.posterior_ratios[check_var_col_ind];
                self.posterior_ratios[check_var_col_ind] = self.blend_marginal_with_memory(
                    bp_marginal,
                    previous_marginal,
                    check_var_col_ind,
                );

                // Now perform iteration in the reverse direction to accumulate right to left.
                check_to_var_row_sum = N::zero();
                // Remove each message contribution.
                for (ind, check_var_row_val) in izip!(
                    data_range.clone(),
                    &self.check_to_variable.data()[data_range.clone()]
                )
                .rev()
                {
                    let map_ind = self.check_to_variable_nnz_map[ind];
                    self.variable_to_check.data_mut()[map_ind] += check_to_var_row_sum;
                    check_to_var_row_sum += *check_var_row_val;

                    debug!(
                        "location ({:?}, {:?}), variable_to_check: {:.32}",
                        self.check_to_variable.indices()[ind],
                        check_var_col_ind,
                        self.variable_to_check.data_mut()[self.check_to_variable_nnz_map[ind]]
                    );
                }

                drop_indices
            };

            if self.should_drop_variable_messages(check_var_col_ind) {
                self.zero_variable_to_check_messages(drop_indices);
            }
        }

        self.bound_magnitudes();

        &mut self.variable_to_check
    }

    pub fn run_iteration(&mut self, detectors: ArrayView1<Bit>) {
        debug!("Iteration {:?} start", self.current_iteration);
        self.compute_check_to_variable(detectors);
        // Now compute variable to check messages
        self.compute_variable_to_check();

        self.compute_hard_decision();
        debug!("Iteration {:?} end", self.current_iteration);
    }

    pub fn run_check_to_variable_update(&mut self, detectors: ArrayView1<Bit>) {
        self.compute_check_to_variable(detectors);
    }

    pub fn run_check_to_variable_update_for_check(
        &mut self,
        detectors: ArrayView1<Bit>,
        check_idx: usize,
    ) {
        let Some(var_check_row_vec) = self.variable_to_check.outer_view(check_idx) else {
            return;
        };

        let alpha = self.alpha();
        let variable_alphas = self.variable_alphas.as_ref();
        let row_sign = if detectors[check_idx] == 1 {
            N::one().neg()
        } else {
            N::one()
        };
        let mut accumulated_sign = row_sign.is_negative();
        let mut min_ind: usize = 0;
        let mut min_message = N::max_value();
        let mut second_min_message = N::max_value();

        for (var_idx, msg) in var_check_row_vec.iter() {
            accumulated_sign ^= msg.is_negative();
            let abs_msg = msg.abs();
            if abs_msg <= min_message {
                second_min_message = min_message;
                min_message = abs_msg;
                min_ind = var_idx;
            } else if abs_msg <= second_min_message {
                second_min_message = abs_msg;
            }
        }

        let scale = self.data_scale_value;
        for (var_idx, msg) in var_check_row_vec.iter() {
            let check_to_variable_sign = accumulated_sign ^ msg.is_negative();
            let check_to_variable_min: N = if var_idx != min_ind {
                min_message
            } else {
                second_min_message
            };

            let alpha_for_variable = match variable_alphas {
                Some(alphas) => alphas[var_idx],
                None => alpha,
            };

            let mut check_to_variable = alpha_for_variable * check_to_variable_min;
            if check_to_variable_sign {
                check_to_variable = check_to_variable.neg();
            }
            if let Some(scale_val) = scale {
                check_to_variable /= scale_val;
            }

            if let Some(nnz_idx) = self.check_to_variable.nnz_index(check_idx, var_idx) {
                self.check_to_variable.data_mut()[nnz_idx.0] = check_to_variable;
            }
        }
    }

    pub fn run_variable_to_check_update(&mut self) {
        self.compute_variable_to_check();
        self.compute_hard_decision();
    }

    pub fn run_variable_to_check_update_for_variable(&mut self, variable_idx: usize) {
        let check_neighbors = self.check_neighbors(variable_idx);
        if check_neighbors.is_empty() {
            return;
        }

        let mut sum = self.compute_variable_prior(variable_idx);
        for check_idx in &check_neighbors {
            if let Some(c2v_nnz_idx) = self.check_to_variable.nnz_index(*check_idx, variable_idx)
            {
                if let Some(v2c_nnz_idx) = self.variable_to_check.nnz_index(*check_idx, variable_idx)
                {
                    self.variable_to_check.data_mut()[v2c_nnz_idx.0] = sum;
                    sum += self.check_to_variable.data()[c2v_nnz_idx.0];
                }
            }
        }

        self.posterior_ratios[variable_idx] = sum;

        let mut reverse_sum = N::zero();
        for check_idx in check_neighbors.iter().rev() {
            if let Some(c2v_nnz_idx) = self.check_to_variable.nnz_index(*check_idx, variable_idx)
            {
                if let Some(v2c_nnz_idx) = self.variable_to_check.nnz_index(*check_idx, variable_idx)
                {
                    self.variable_to_check.data_mut()[v2c_nnz_idx.0] += reverse_sum;
                    reverse_sum += self.check_to_variable.data()[c2v_nnz_idx.0];
                }
            }
        }

        if self.should_drop_variable_messages(variable_idx) {
            let drop_indices: Vec<usize> = check_neighbors.iter().filter_map(|check_idx| {
                self.check_to_variable.nnz_index(*check_idx, variable_idx)
                    .map(|nnz_idx| nnz_idx.0)
            }).collect();
            self.zero_variable_to_check_messages(drop_indices);
        }

        self.bound_magnitudes();
    }

    pub fn run_lrbp_variable_to_check_update(&mut self) {
        self.compute_variable_to_check_lrbp();
        self.compute_hard_decision();
    }

    pub fn recompute_hard_decision(&mut self) {
        self.compute_hard_decision();
    }

    pub fn check_neighbors(&self, variable_idx: usize) -> Vec<usize> {
        let Some(check_col_vec) = self.check_to_variable.outer_view(variable_idx) else {
            return Vec::new();
        };
        check_col_vec.indices().to_vec()
    }

    pub fn variable_neighbors(&self, check_idx: usize) -> Vec<usize> {
        let Some(var_row_vec) = self.variable_to_check.outer_view(check_idx) else {
            return Vec::new();
        };
        var_row_vec.indices().to_vec()
    }

    pub fn posterior_ratios_f64(&self) -> Array1<f64> {
        self.posterior_ratios.clone().mapv_into_any(|val| {
            let posterior = N::to_f64(&val).unwrap_or(0.0);
            match self.config.data_scale_value {
                Some(scale_val) => posterior / scale_val,
                None => posterior,
            }
        })
    }

    pub fn posterior_ratio_f64(&self, variable_idx: usize) -> f64 {
        let posterior = N::to_f64(&self.posterior_ratios[variable_idx]).unwrap_or(0.0);
        match self.config.data_scale_value {
            Some(scale_val) => posterior / scale_val,
            None => posterior,
        }
    }

    pub fn check_to_variable_data(&self) -> &[N] {
        self.check_to_variable.data()
    }

    pub fn check_to_variable_data_mut(&mut self) -> &mut [N] {
        self.check_to_variable.data_mut()
    }

    pub fn check_to_variable_indices(&self) -> &[usize] {
        self.check_to_variable.indices()
    }

    pub fn current_decoding(&self) -> &Array1<Bit> {
        &self.decoding
    }

    pub fn build_result(
        &mut self,
        success: bool,
        decoded_detectors: Array1<Bit>,
        max_iter: usize,
    ) -> DecodeResult {
        DecodeResult {
            decoding: self.decoding.clone(),
            decoded_detectors,
            posterior_ratios: self.posterior_ratios.clone().mapv_into_any(|val| {
                let posterior = N::to_f64(&val).unwrap();
                match self.config.data_scale_value {
                    Some(scale_val) => posterior / scale_val,
                    None => posterior,
                }
            }),
            success,
            decoding_quality: if success {
                self.get_decoding_quality(self.decoding.clone().view())
            } else {
                f64::MAX
            },
            iterations: self.current_iteration,
            max_iter,
            extra: BPExtraResult::None,
        }
    }
    fn bound_magnitudes(&mut self) {
        // Bound magnitudes
        if self.max_data_value.is_some() {
            let max_val = self.max_data_value.unwrap();
            self.variable_to_check
                .data_mut()
                .iter_mut()
                .for_each(|v| *v = Self::bound_value_magnitude(*v, max_val));
            self.posterior_ratios
                .iter_mut()
                .for_each(|v| *v = Self::bound_value_magnitude(*v, max_val));
        }
    }

    fn bound_value_magnitude(value: N, max_val: N) -> N
    where
        N: std::ops::Add,
    {
        if value < max_val.neg() {
            max_val.neg()
        } else if value > max_val {
            max_val
        } else {
            value
        }
    }

    fn compute_hard_decision(&mut self) {
        for (idx, posterior) in self.posterior_ratios.iter().enumerate() {
            self.decoding[idx] = Bit::from((*posterior) <= N::zero());
        }
        debug!("Posteriors: {:?}", self.posterior_ratios);
        debug!("Hard decision: {:?}", self.decoding);
    }

    pub fn compute_decoded_detectors(&self) -> Array1<Bit> {
        self.get_detectors(self.decoding.view())
    }

    // Check the convergence of the problem instance
    pub fn check_convergence(
        &self,
        detectors: ArrayView1<Bit>,
        decoded_detectors: ArrayView1<Bit>,
    ) -> bool {
        detectors == decoded_detectors
    }

    /// Decode while collecting per-iteration metrics used by the Step1 stagnation detector.
    ///
    /// This method does not change the underlying BP update logic; it only observes
    /// decoder state after each iteration.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_detailed_step1_metrics(
        &mut self,
        detectors: ArrayView1<Bit>,
        k_default: usize,
        t_warmup: usize,
        theta_r_default: f64,
        n_min_default: usize,
        low_threshold_0: f64,
        low_threshold_1: f64,
        endpoint_k: usize,
        collect_sign_trajectory: bool,
        snapshot_times: Option<&[usize]>,
    ) -> (DecodeResult, Step1Metrics) {
        // Initialize probability ratios.
        self.initialize_decoder();

        let n_vars = self.check_matrix.cols();
        let scale = self.config.data_scale_value.unwrap_or(1.0);
        let k_default = k_default.max(1);
        let endpoint_k = endpoint_k.max(1);
        let detection_start_t = t_warmup.saturating_add(k_default);
        let theta_r_default = theta_r_default.clamp(0.0, 1.0);
        let low_threshold_0 = low_threshold_0.abs();
        let low_threshold_1 = low_threshold_1.abs();

        // Define t=0 using the variable prior only (no incoming check messages).
        let mut prev_hat_e: Vec<u8> = vec![0; n_vars];
        let mut prev_m: Vec<f64> = vec![0.0; n_vars];
        let mut prev_sign: Vec<u8> = vec![0; n_vars];
        for j in 0..n_vars {
            let m0 = N::to_f64(&self.log_prior_ratios[j]).unwrap_or(0.0) / scale;
            prev_m[j] = m0;
            prev_hat_e[j] = Bit::from(m0 <= 0.0);
            // sgn(0) := +1, so treat m==0 as non-negative.
            prev_sign[j] = u8::from(m0 < 0.0);
        }

        // Endpoint sign history for F_{endpoint_k}: store t=0 then roll.
        let endpoint_ring_len = endpoint_k + 1;
        let mut sign_endpoint_ring: Vec<u8> = vec![0; endpoint_ring_len * n_vars];
        sign_endpoint_ring[0..n_vars].copy_from_slice(&prev_sign);

        // Flip-rate window state for r_j(t; K_default).
        let mut flip_ring: Vec<u8> = vec![0; k_default * n_vars];
        let mut flip_sum: Vec<u16> = vec![0; n_vars];
        let mut post_warmup_flip_count: usize = 0;

        let mut w_sigma: Vec<u32> = Vec::with_capacity(self.config.max_iter);
        let mut delta_hd: Vec<u32> = Vec::with_capacity(self.config.max_iter);
        let mut delta_m_l2: Vec<f64> = Vec::with_capacity(self.config.max_iter);
        let mut m_norm_l2: Vec<f64> = Vec::with_capacity(self.config.max_iter);
        let mut m_bar: Vec<f64> = Vec::with_capacity(self.config.max_iter);
        let mut f_low_0: Vec<f64> = Vec::with_capacity(self.config.max_iter);
        let mut f_low_1: Vec<f64> = Vec::with_capacity(self.config.max_iter);
        let mut f_endpoint: Vec<u32> = Vec::with_capacity(self.config.max_iter);
        let mut v_osc: Vec<u32> = Vec::with_capacity(self.config.max_iter);

        let sign_bytes_per_iter = (n_vars + 7) / 8;
        let mut sign_trajectory_packed: Option<Vec<u8>> = if collect_sign_trajectory {
            Some(Vec::with_capacity(self.config.max_iter * sign_bytes_per_iter))
        } else {
            None
        };

        let mut snapshot_times_sorted: Vec<usize> = snapshot_times
            .map(|times| {
                let mut v = times.to_vec();
                v.sort_unstable();
                v.dedup();
                v
            })
            .unwrap_or_default();
        // Ignore t=0 snapshots for now (spec uses t>=1).
        snapshot_times_sorted.retain(|t| *t > 0);
        let mut abs_m_snapshots: Option<Vec<f32>> = if snapshot_times_sorted.is_empty() {
            None
        } else {
            Some(vec![0.0; snapshot_times_sorted.len() * n_vars])
        };
        let mut next_snapshot_index: usize = 0;

        let mut success: bool = false;
        let mut decoded_detectors = Array1::default(detectors.dim());
        let mut t_stag: Option<usize> = None;

        for _ in 0..self.config.max_iter {
            self.run_iteration(detectors);
            self.current_iteration += 1;
            let t = self.current_iteration;

            decoded_detectors = self.compute_decoded_detectors();
            let mut w: u32 = 0;
            for i in 0..detectors.len() {
                if detectors[i] != decoded_detectors[i] {
                    w += 1;
                }
            }
            success = w == 0;

            // Compute per-variable metrics.
            let endpoint_store_pos = t % endpoint_ring_len;
            let endpoint_old_pos = if t >= endpoint_k {
                (t - endpoint_k) % endpoint_ring_len
            } else {
                0
            };
            let use_in_window = t > t_warmup;
            let window_ready = use_in_window && (post_warmup_flip_count + 1 >= k_default);
            let flip_store_offset = if use_in_window {
                (post_warmup_flip_count % k_default) * n_vars
            } else {
                0
            };

            let mut delta_hd_count: u32 = 0;
            let mut delta_m_sum_sq: f64 = 0.0;
            let mut m_norm_sum_sq: f64 = 0.0;
            let mut abs_sum: f64 = 0.0;
            let mut low0_count: u32 = 0;
            let mut low1_count: u32 = 0;
            let mut f_endpoint_count: u32 = 0;
            let mut v_osc_count: u32 = 0;

            let mut packed_row: Vec<u8> = if collect_sign_trajectory {
                vec![0u8; sign_bytes_per_iter]
            } else {
                Vec::new()
            };

            for j in 0..n_vars {
                let m = N::to_f64(&self.posterior_ratios[j]).unwrap_or(0.0) / scale;
                let sign = u8::from(m < 0.0);

                if t >= endpoint_k {
                    let old_sign = sign_endpoint_ring[endpoint_old_pos * n_vars + j];
                    f_endpoint_count += u32::from((sign ^ old_sign) != 0);
                }
                sign_endpoint_ring[endpoint_store_pos * n_vars + j] = sign;

                // flip_s = sgn(M(t)) XOR sgn(M(t-1))
                let flip = sign ^ prev_sign[j];
                let old_flip = if use_in_window && window_ready {
                    flip_ring[flip_store_offset + j]
                } else {
                    0
                };
                if use_in_window {
                    flip_ring[flip_store_offset + j] = flip;
                    let updated_sum = (flip_sum[j] as i32) + (flip as i32) - (old_flip as i32);
                    flip_sum[j] = updated_sum.max(0) as u16;
                }
                prev_sign[j] = sign;

                if window_ready {
                    if (flip_sum[j] as f64) > theta_r_default * (k_default as f64) {
                        v_osc_count += 1;
                    }
                }

                let hat = self.decoding[j];
                delta_hd_count += u32::from((hat ^ prev_hat_e[j]) != 0);
                prev_hat_e[j] = hat;

                let dm = m - prev_m[j];
                delta_m_sum_sq += dm * dm;
                prev_m[j] = m;

                m_norm_sum_sq += m * m;
                let abs_m = m.abs();
                abs_sum += abs_m;
                if abs_m < low_threshold_0 {
                    low0_count += 1;
                }
                if abs_m < low_threshold_1 {
                    low1_count += 1;
                }

                if collect_sign_trajectory && sign != 0 {
                    let byte_index = j / 8;
                    let bit_in_byte = 7 - (j % 8);
                    packed_row[byte_index] |= 1u8 << bit_in_byte;
                }
            }

            if use_in_window {
                post_warmup_flip_count += 1;
            }

            if collect_sign_trajectory {
                if let Some(buf) = sign_trajectory_packed.as_mut() {
                    buf.extend_from_slice(&packed_row);
                }
            }

            // Snapshots of |M| at selected times.
            if let Some(snaps) = abs_m_snapshots.as_mut() {
                while next_snapshot_index < snapshot_times_sorted.len()
                    && snapshot_times_sorted[next_snapshot_index] == t
                {
                    let base = next_snapshot_index * n_vars;
                    for j in 0..n_vars {
                        snaps[base + j] = prev_m[j].abs() as f32;
                    }
                    next_snapshot_index += 1;
                }
            }

            w_sigma.push(w);
            delta_hd.push(delta_hd_count);
            delta_m_l2.push(delta_m_sum_sq.sqrt());
            m_norm_l2.push(m_norm_sum_sq.sqrt());
            m_bar.push(abs_sum / (n_vars as f64));
            f_low_0.push((low0_count as f64) / (n_vars as f64));
            f_low_1.push((low1_count as f64) / (n_vars as f64));
            f_endpoint.push(f_endpoint_count);
            v_osc.push(v_osc_count);

            if t_stag.is_none()
                && t >= detection_start_t
                && w > 0
                && (v_osc_count as usize) >= n_min_default
            {
                t_stag = Some(t);
            }

            if success {
                debug!("Succeeded on iteration {:?}", self.current_iteration);
                break;
            }
        }

        // If we exited early, fill any remaining requested snapshots with the final |M|.
        if let Some(snaps) = abs_m_snapshots.as_mut() {
            while next_snapshot_index < snapshot_times_sorted.len() {
                let base = next_snapshot_index * n_vars;
                for j in 0..n_vars {
                    snaps[base + j] = prev_m[j].abs() as f32;
                }
                next_snapshot_index += 1;
            }
        }

        let decode_result = self.build_result(success, decoded_detectors, self.config.max_iter);
        let metrics = Step1Metrics {
            variable_count: n_vars,
            w_sigma,
            delta_hd,
            delta_m_l2,
            m_norm_l2,
            m_bar,
            f_low_0,
            f_low_1,
            f_endpoint,
            v_osc,
            t_stag,
            sign_trajectory_packed,
            sign_bytes_per_iter,
            abs_m_snapshots,
            snapshot_times: snapshot_times_sorted,
        };

        (decode_result, metrics)
    }
}

impl<N> Decoder for MinSumBPDecoder<N>
where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + FromPrimitive
        + ToPrimitive
        + Signed
        + Bounded
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + std::ops::DivAssign
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
        self.check_matrix.clone()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.config.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        // Initialize probability ratios
        self.initialize_decoder();
        let mut success: bool = false;
        let mut decoded_detectors = Array1::default(detectors.dim());

        for _ in 0..self.config.max_iter {
            self.run_iteration(detectors);
            self.current_iteration += 1;
            decoded_detectors = self.compute_decoded_detectors();
            success = self.check_convergence(detectors, decoded_detectors.view());

            // If we have converged may now exit
            if success {
                debug!("Succeeded on iteration {:?}", self.current_iteration);
                break;
            }
        }

        self.build_result(success, decoded_detectors, self.config.max_iter)
    }
}

impl<N> DecoderRunner for MinSumBPDecoder<N> where
    N: PartialEq
        + Debug
        + Default
        + Clone
        + Copy
        + FromPrimitive
        + ToPrimitive
        + Signed
        + Bounded
        + std::cmp::PartialOrd
        + std::ops::Add
        + std::ops::AddAssign
        + std::ops::Mul<N>
        + std::ops::MulAssign
        + std::ops::DivAssign
        + Send
        + Sync
        + std::fmt::Display
        + 'static
{
}

#[cfg(test)]
mod tests {
    use crate::bipartite_graph::BipartiteGraph;

    use super::*;
    use env_logger;
    use ndarray::prelude::*;

    use crate::dem::DetectorErrorModel;
    use crate::utilities::test::get_test_data_path;
    use ndarray::Array2;
    use ndarray_npy::read_npy;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn decode_detailed_repetition_code() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];

        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let iterations = 10;
        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            ..Default::default()
        };
        let arc_bp_config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(arc_check_matrix, arc_bp_config);

        let error = array![0, 0, 0];
        let detectors: Array1<Bit> = array![0, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![1, 0, 0];
        let detectors: Array1<Bit> = array![1, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 1, 0];
        let detectors: Array1<Bit> = array![1, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 0, 1];
        let detectors: Array1<Bit> = array![0, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);
    }

    #[test]
    fn decode_detailed_step1_metrics_matches_decode_detailed() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];
        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let iterations = 10;
        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            ..Default::default()
        };
        let arc_bp_config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<f64> = MinSumBPDecoder::new(arc_check_matrix, arc_bp_config);
        let detectors: Array1<Bit> = array![0, 0];

        let expected = decoder.clone().decode_detailed(detectors.view());
        let (actual, metrics) = decoder.decode_detailed_step1_metrics(
            detectors.view(),
            20,
            0,
            0.4,
            3,
            0.1,
            0.5,
            20,
            false,
            None,
        );

        assert_eq!(expected.success, actual.success);
        assert_eq!(expected.iterations, actual.iterations);
        assert_eq!(expected.decoding, actual.decoding);
        assert_eq!(expected.decoded_detectors, actual.decoded_detectors);

        let iters = actual.iterations;
        assert_eq!(metrics.w_sigma.len(), iters);
        assert_eq!(metrics.delta_hd.len(), iters);
        assert_eq!(metrics.delta_m_l2.len(), iters);
        assert_eq!(metrics.m_norm_l2.len(), iters);
        assert_eq!(metrics.m_bar.len(), iters);
        assert_eq!(metrics.f_low_0.len(), iters);
        assert_eq!(metrics.f_low_1.len(), iters);
        assert_eq!(metrics.f_endpoint.len(), iters);
        assert_eq!(metrics.v_osc.len(), iters);
        assert!(metrics.sign_trajectory_packed.is_none());
        assert!(metrics.abs_m_snapshots.is_none());
        assert!(metrics.snapshot_times.is_empty());
    }

    #[test]
    fn decode_detailed_step1_metrics_respects_warmup_window_start() {
        init();

        let check_matrix = array![[1, 1, 0], [0, 1, 1],];
        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let iterations = 25;
        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            ..Default::default()
        };
        let arc_bp_config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<f64> = MinSumBPDecoder::new(arc_check_matrix, arc_bp_config);
        let detectors: Array1<Bit> = array![1, 0];

        let k_default = 3usize;
        let t_warmup = 4usize;
        let first_possible_t = t_warmup + k_default;

        let (_actual, metrics) = decoder.decode_detailed_step1_metrics(
            detectors.view(),
            k_default,
            t_warmup,
            0.0,
            0,
            0.1,
            0.5,
            20,
            false,
            None,
        );

        let prefix_len = metrics
            .v_osc
            .len()
            .min(first_possible_t.saturating_sub(1));
        for idx in 0..prefix_len {
            assert_eq!(
                metrics.v_osc[idx],
                0,
                "v_osc must be zero before warmup+window; t={} threshold={}",
                idx + 1,
                first_possible_t,
            );
        }

        if let Some(t_stag) = metrics.t_stag {
            assert!(
                t_stag >= first_possible_t,
                "t_stag={} should be >= warmup+window={}",
                t_stag,
                first_possible_t,
            );
        }
    }

    #[test]
    fn decode_detailed_repetition_code_int() {
        init();

        // Build 3, 2 qubit repetition code with weight 2 checks
        let check_matrix = array![[1, 1, 0], [0, 1, 1],];

        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let iterations = 10;

        let bits = 7;
        let scale = 4.0;

        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.003, 0.003, 0.003],
            max_iter: iterations,
            max_data_value: Some(((1 << bits) - 1) as f64),
            data_scale_value: Some(scale),
            ..Default::default()
        };
        let arc_bp_config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<isize> =
            MinSumBPDecoder::new(arc_check_matrix, arc_bp_config);

        let error = array![0, 0, 0];
        let detectors: Array1<Bit> = array![0, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![1, 0, 0];
        let detectors: Array1<Bit> = array![1, 0];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 1, 0];
        let detectors: Array1<Bit> = array![1, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);

        let error = array![0, 0, 1];
        let detectors: Array1<Bit> = array![0, 1];

        let result = decoder.decode_detailed(detectors.view());

        assert_eq!(result.decoding, error);
        assert_eq!(result.decoded_detectors, detectors);
        assert_eq!(result.max_iter, iterations);
        assert!(result.success);
    }

    #[test]
    fn decode_detailed_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            alpha: Some(0.),
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                >= (detectors_slice.shape()[0] as f64) * 0.93
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    #[test]
    fn decode_144_12_12() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);

        let results = decoder_144_12_12.par_decode_batch(detectors_slice);

        let results_detailed = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        for i in 0..results.shape()[0] {
            assert!(results.row(i) == results_detailed[i].decoding)
        }
    }

    #[test]
    fn decode_detailed_144_12_12_membp() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);
        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            gamma0: Some(0.15),
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<f64> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let par_results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);
        let results = decoder_144_12_12.decode_detailed_batch(detectors_slice);
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                == par_results
                    .iter()
                    .map(|x| x.success as usize)
                    .sum::<usize>() as f64
        );
        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                >= (detectors_slice.shape()[0] as f64) * 0.93
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    #[test]
    fn decode_detailed_144_12_12_int() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open file");
        let check_matrix = Arc::new(code_144_12_12.detector_error_matrix);

        let bits = 16;
        let scale = 8.0;

        let bp_config_144_12_12 = MinSumDecoderConfig {
            error_priors: code_144_12_12.error_priors,
            max_data_value: Some(((1 << bits) - 1) as f64),
            data_scale_value: Some(scale),
            alpha: Some(0.),
            ..Default::default()
        };
        let config = Arc::new(bp_config_144_12_12);

        let mut decoder_144_12_12: MinSumBPDecoder<isize> =
            MinSumBPDecoder::new(check_matrix, config);
        let num_errors = 100;
        let detectors_slice = detectors_144_12_12.slice(s![..num_errors, ..]);
        let results = decoder_144_12_12.par_decode_detailed_batch(detectors_slice);

        assert!(
            results.iter().map(|x| x.success as usize).sum::<usize>() as f64
                >= (detectors_slice.shape()[0] as f64) * 0.93
        );

        assert_eq!(results[0].decoding.len(), 8785);
    }

    #[test]
    fn variable_message_drop_zeros_selected_variable_edges() {
        init();

        let check_matrix = array![[1, 1, 0], [0, 1, 1],];
        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.1, 0.45, 0.45],
            enable_variable_message_drop: true,
            drop_probability: 1.0,
            drop_llr_threshold: 1.0,
            rng_seed: Some(7),
            ..Default::default()
        };
        let config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<f64> = MinSumBPDecoder::new(arc_check_matrix, config);
        decoder.initialize_check_to_variable();
        decoder.run_variable_to_check_update();

        let dropped_edge = decoder.variable_to_check.nnz_index(0, 0).unwrap().0;
        let kept_edge_a = decoder.variable_to_check.nnz_index(0, 1).unwrap().0;
        let kept_edge_b = decoder.variable_to_check.nnz_index(1, 1).unwrap().0;

        assert_eq!(decoder.variable_to_check.data()[dropped_edge], 0.0);
        assert_ne!(decoder.variable_to_check.data()[kept_edge_a], 0.0);
        assert_ne!(decoder.variable_to_check.data()[kept_edge_b], 0.0);
    }

    #[test]
    fn variable_message_drop_respects_threshold() {
        init();

        let check_matrix = array![[1, 1, 0], [0, 1, 1],];
        let check_matrix: SparseBipartiteGraph<_> = SparseBipartiteGraph::from_dense(check_matrix);
        let arc_check_matrix = Arc::new(check_matrix);

        let bp_config = MinSumDecoderConfig {
            error_priors: array![0.1, 0.45, 0.45],
            enable_variable_message_drop: true,
            drop_probability: 1.0,
            drop_llr_threshold: 3.0,
            rng_seed: Some(7),
            ..Default::default()
        };
        let config = Arc::new(bp_config);

        let mut decoder: MinSumBPDecoder<f64> = MinSumBPDecoder::new(arc_check_matrix, config);
        decoder.initialize_check_to_variable();
        decoder.run_variable_to_check_update();

        let edge = decoder.variable_to_check.nnz_index(0, 0).unwrap().0;
        assert_ne!(decoder.variable_to_check.data()[edge], 0.0);
    }
}
