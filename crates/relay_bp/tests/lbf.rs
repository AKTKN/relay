// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::{array, Array1};
use relay_bp::bipartite_graph::BipartiteGraph;
use relay_bp::bp::lbf::{LbfDecoder, LbfDecoderConfig};
use relay_bp::decoder::{Bit, Decoder, Mod2Mul, SparseBitMatrix};
use std::sync::Arc;

fn make_lbf(check_matrix: SparseBitMatrix, max_iter: usize, weight: usize, k_step: usize) -> LbfDecoder {
    let n = check_matrix.cols();
    let cfg = LbfDecoderConfig {
        error_priors: Array1::from_elem(n, 0.003),
        max_iter,
        weight,
        k_step,
    };
    LbfDecoder::new(Arc::new(check_matrix), Arc::new(cfg))
}

#[test]
fn lbf_repetition_code_trivial_cases_succeed() {
    // H = [[1,1,0],[0,1,1]]
    let h_dense = array![[1u8, 1, 0], [0u8, 1, 1]];
    let h = SparseBitMatrix::from_dense(h_dense);

    let mut dec = make_lbf(h, 20, 1000, 2);

    let syndromes: Vec<Array1<Bit>> = vec![
        array![1u8, 0u8],
        array![1u8, 1u8],
        array![0u8, 1u8],
    ];

    for s in syndromes {
        let result = dec.decode_detailed(s.view());
        assert!(result.success, "expected success for syndrome {s:?}");
        assert_eq!(result.decoded_detectors, s);
        assert!(result.iterations <= 20);
    }
}

#[test]
fn lbf_steane_code_single_error_syndromes_succeed() {
    // Standard (3x7) parity-check used for the classical [7,4,3] Hamming code.
    // Columns are unique non-zero 3-bit vectors.
    let h_dense = array![
        [1u8, 0, 0, 1, 0, 1, 1],
        [0u8, 1, 0, 1, 1, 0, 1],
        [0u8, 0, 1, 0, 1, 1, 1]
    ];
    let h = SparseBitMatrix::from_dense(h_dense);

    let mut dec = make_lbf(h.clone(), 50, 1000, 2);

    // Test degree-1 columns (0,1,2) specifically; LBF should flip these cleanly.
    for j in 0..3 {
        let mut e = Array1::<Bit>::zeros(7);
        e[j] = 1;
        let s = h.mul_mod2(&e);
        let result = dec.decode_detailed(s.view());
        assert!(result.success, "expected success for single-error at col {j}");
        assert_eq!(result.decoded_detectors, s);
    }

    // Also verify a denser syndrome still converges (not necessarily to a weight-1 solution).
    let mut e6 = Array1::<Bit>::zeros(7);
    e6[6] = 1;
    let s6 = h.mul_mod2(&e6);
    let result6 = dec.decode_detailed(s6.view());
    assert!(result6.success, "expected success for syndrome of col 6");
    assert_eq!(result6.decoded_detectors, s6);
 }
