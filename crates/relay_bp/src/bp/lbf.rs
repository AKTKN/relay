use crate::decoder::{BPExtraResult, Bit, DecodeResult, Decoder, SparseBitMatrix};
use log::warn;
use ndarray::{Array1, ArrayView1};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct LbfDecoderConfig {
    pub error_priors: Array1<f64>,
    pub max_iter: usize,
    pub k_step: usize,
}

impl Default for LbfDecoderConfig {
    fn default() -> Self {
        Self {
            error_priors: Default::default(),
            max_iter: 200,
            k_step: 2,
        }
    }
}

#[derive(Clone)]
pub struct LbfDecoder {
    check_matrix: Arc<SparseBitMatrix>,
    config: Arc<LbfDecoderConfig>,
    variable_to_checks: Vec<Vec<usize>>,
    check_to_variables: Vec<Vec<usize>>,
    variable_neighborhoods: Vec<Vec<usize>>,
    current_iteration: usize,
}

impl LbfDecoder {
    pub fn new(check_matrix: Arc<SparseBitMatrix>, config: Arc<LbfDecoderConfig>) -> Self {
        assert!(config.k_step >= 2 && config.k_step % 2 == 0);

        let csc = check_matrix.to_csc();
        let csr = check_matrix.to_csr();

        let m = check_matrix.rows();
        let n = check_matrix.cols();

        let mut variable_to_checks = vec![vec![]; n];
        for v in 0..n {
            if let Some(col) = csc.outer_view(v) {
                variable_to_checks[v] = col.indices().to_vec();
                variable_to_checks[v].sort_unstable();
            }
        }

        let mut check_to_variables = vec![vec![]; m];
        for c in 0..m {
            if let Some(row) = csr.outer_view(c) {
                check_to_variables[c] = row.indices().to_vec();
                check_to_variables[c].sort_unstable();
            }
        }

        let variable_neighborhoods =
            compute_variable_neighborhoods(&variable_to_checks, &check_to_variables, config.k_step);

        if config.error_priors.len() != n {
            warn!("error_priors length mismatch, ignored.");
        }

        Self {
            check_matrix,
            config,
            variable_to_checks,
            check_to_variables,
            variable_neighborhoods,
            current_iteration: 0,
        }
    }

    fn build_result(
        &mut self,
        _detectors: ArrayView1<Bit>,
        success: bool,
        converged: bool,
        decoding: Array1<Bit>,
    ) -> DecodeResult {
        let decoded_detectors = self.get_detectors(decoding.view());
        let decoding_len = decoding.len();
        let decoding_quality = self.get_decoding_quality(decoding.view());

        DecodeResult {
            decoding,
            decoded_detectors,
            posterior_ratios: Array1::zeros(decoding_len),
            success,
            decoding_quality,
            iterations: self.current_iteration,
            max_iter: self.config.max_iter,
            extra: BPExtraResult::LBFTrace { converged },
        }
    }

    /// Compute CN total parity:
    /// s_j ^ XOR(all connected variables)
    fn compute_check_parities(
        &self,
        decoding: &Array1<Bit>,
        syndrome: &Array1<Bit>,
    ) -> Vec<Bit> {
        let m = self.check_to_variables.len();
        let mut parity = vec![0; m];

        for c in 0..m {
            let mut p = syndrome[c] & 1;
            for &v in &self.check_to_variables[c] {
                p ^= decoding[v] & 1;
            }
            parity[c] = p;
        }

        parity
    }

    /// CN -> VN message using fast formula:
    /// m_{c->v} = total_parity[c] ^ decoding[v]
    #[inline]
    fn cn_to_vn_message_fast(
        total_parity: &[Bit],
        decoding: &Array1<Bit>,
        c: usize,
        v: usize,
    ) -> usize {
        (total_parity[c] ^ decoding[v]) as usize
    }

    /// score = (#flip) - (#keep)
    fn variable_score(
        &self,
        v: usize,
        decoding: &Array1<Bit>,
        total_parity: &[Bit],
    ) -> isize {
        let mut flip_votes = 0usize;
        let mut keep_votes = 0usize;

        for &c in &self.variable_to_checks[v] {
            let desired = Self::cn_to_vn_message_fast(total_parity, decoding, c, v) as Bit;
            let should_flip = desired ^ (decoding[v] & 1);

            if should_flip == 1 {
                flip_votes += 1;
            } else {
                keep_votes += 1;
            }
        }

        flip_votes as isize - keep_votes as isize
    }

    fn is_better_candidate(
        &self,
        a: usize,
        b: usize,
        scores: &[isize],
        degrees: &[usize],
    ) -> bool {
        if scores[a] != scores[b] {
            return scores[a] > scores[b];
        }
        if degrees[a] != degrees[b] {
            return degrees[a] < degrees[b];
        }
        a < b
    }
}

impl Decoder for LbfDecoder {
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
        Array1::zeros(self.check_matrix.cols())
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let n = self.check_matrix.cols();
        let m = self.check_matrix.rows();

        if detectors.len() != m {
            panic!("detectors length mismatch");
        }

        self.current_iteration = 0;

        let syndrome = detectors.to_owned();
        let mut decoding = Array1::<Bit>::zeros(n);

        let degrees: Vec<usize> = self.variable_to_checks.iter().map(|v| v.len()).collect();

        for _ in 0..self.config.max_iter {
            // Compute check parities
            let total_parity = self.compute_check_parities(&decoding, &syndrome);

            // Check success
            if total_parity.iter().all(|&x| x == 0) {
                return self.build_result(detectors, true, true, decoding);
            }

            // Compute scores
            let mut scores = vec![0isize; n];
            for v in 0..n {
                scores[v] = self.variable_score(v, &decoding, &total_parity);
            }

            // Select flips
            let mut flip_set = Vec::new();

            for v in 0..n {
                if scores[v] <= -1 {
                    continue;
                }

                let mut is_best = true;
                for &u in &self.variable_neighborhoods[v] {
                    if self.is_better_candidate(u, v, &scores, &degrees) {
                        is_best = false;
                        break;
                    }
                }

                if is_best {
                    flip_set.push(v);
                }
            }

            if flip_set.is_empty() {
                break;
            }

            // Flip
            for &v in &flip_set {
                decoding[v] ^= 1;
            }

            self.current_iteration += 1;
        }

        let converged = self.current_iteration < self.config.max_iter;
        self.build_result(detectors, false, converged, decoding)
    }
}

fn compute_variable_neighborhoods(
    variable_to_checks: &[Vec<usize>],
    check_to_variables: &[Vec<usize>],
    k_step: usize,
) -> Vec<Vec<usize>> {
    let n = variable_to_checks.len();
    let m = check_to_variables.len();

    let mut neighborhoods = Vec::with_capacity(n);

    for start_v in 0..n {
        let mut visited_v = vec![false; n];
        let mut visited_c = vec![false; m];
        visited_v[start_v] = true;

        let mut q = VecDeque::new();
        q.push_back((true, start_v, 0));

        let mut neigh = Vec::new();

        while let Some((is_var, idx, depth)) = q.pop_front() {
            if depth >= k_step {
                continue;
            }

            if is_var {
                for &c in &variable_to_checks[idx] {
                    if !visited_c[c] {
                        visited_c[c] = true;
                        q.push_back((false, c, depth + 1));
                    }
                }
            } else {
                for &v in &check_to_variables[idx] {
                    if !visited_v[v] {
                        visited_v[v] = true;
                        let nd = depth + 1;
                        q.push_back((true, v, nd));

                        if nd % 2 == 0 {
                            neigh.push(v);
                        }
                    }
                }
            }
        }

        neigh.sort_unstable();
        neigh.dedup();
        neighborhoods.push(neigh);
    }

    neighborhoods
}