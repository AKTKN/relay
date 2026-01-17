// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::decoder::{Bit, LsdResult, SparseBitMatrix};
use ndarray::Array1;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LsdMethod {
    LSD_0,
    LSD_E,
    LSD_CS,
}

impl LsdMethod {
    pub fn from_str(s: &str) -> Self {
        match s {
            "LSD_E" => LsdMethod::LSD_E,
            "LSD_CS" => LsdMethod::LSD_CS,
            _ => LsdMethod::LSD_0,
        }
    }
}

/// A cluster used in Localized Statistics Decoding
struct LsdCluster {
    id: usize,
    active: bool,
    valid: bool,
    bit_nodes: HashSet<usize>,
    check_nodes: HashSet<usize>,
    /// Boundary checks are checks in the cluster that are connected to bits outside the cluster
    boundary_check_nodes: HashSet<usize>,
    /// Enclosed syndromes are unsatisfied checks that originated this cluster or were merged in
    enclosed_syndromes: HashSet<usize>,
    /// Local solution vector (ordered by bit_nodes iteration order or explicit mapping?)
    /// We need to be careful about ordering. We'll store it relative to sorted bit_nodes or just map it later.
    /// Simplest: store map bit_idx -> val? Or just use the same order as when solving.
    /// When solving we use `bits: Vec<usize>`. We should store that order or the full solution mapped.
    /// Let's store `solution: HashMap<usize, u8>`.
    solution: HashMap<usize, u8>,
}

impl LsdCluster {
    fn new(id: usize, syndrome_idx: usize) -> Self {
        let mut c = Self {
            id,
            active: true,
            valid: false,
            bit_nodes: HashSet::new(),
            check_nodes: HashSet::new(),
            boundary_check_nodes: HashSet::new(),
            enclosed_syndromes: HashSet::new(),
            solution: HashMap::new(),
        };
        // Initialize with the seed syndrome
        c.check_nodes.insert(syndrome_idx);
        c.boundary_check_nodes.insert(syndrome_idx);
        c.enclosed_syndromes.insert(syndrome_idx);
        c
    }
}

/// Gaussian Elimination solver for GF(2)
/// Returns Some(solution) if solvable, None otherwise
fn solve_system_rows(matrix_rows: &Vec<Vec<u8>>, target: &Vec<u8>) -> Option<Vec<u8>> {
    let rows = matrix_rows.len();
    if rows == 0 {
         // If target contains non-zero, impossible with 0 rows?
         // If target is empty, consistent.
         if target.iter().any(|&x| x != 0) { return None; }
         return Some(vec![]); // Length ambiguous, but implies 0 cols effectively or don't care
    }
    let cols = matrix_rows[0].len();
    if cols == 0 {
        if target.iter().any(|&x| x != 0) { return None; }
        return Some(vec![]);
    }
    
    let mut m = matrix_rows.clone(); // (rows x cols)
    let mut t = target.clone();      // (rows)
    
    let mut pivot_row = 0;
    let mut pivot_col = 0;
    let mut pivot_map = Vec::new(); // (row, col)

    while pivot_row < rows && pivot_col < cols {
        // Find pivot in current column
        let mut pivot = pivot_row;
        while pivot < rows && m[pivot][pivot_col] == 0 {
            pivot += 1;
        }
        
        if pivot < rows {
            // Swap rows
            m.swap(pivot_row, pivot);
            let tmp = t[pivot_row]; t[pivot_row] = t[pivot]; t[pivot] = tmp;
            
            pivot_map.push((pivot_row, pivot_col));

            // Eliminate
            for i in 0..rows {
                if i != pivot_row && m[i][pivot_col] == 1 {
                    for j in pivot_col..cols {
                        m[i][j] ^= m[pivot_row][j];
                    }
                    t[i] ^= t[pivot_row];
                }
            }
            pivot_row += 1;
        }
        pivot_col += 1;
    }
    
    // Check for inconsistency
    for i in pivot_row..rows {
        if t[i] != 0 {
            // Row is 0=1
            return None; 
        }
    }
    
    // Back substitution (solution extraction)
    // Pivot vars are determined by non-pivot vars.
    // Set free vars (non-pivot) to 0.
    let mut x = vec![0u8; cols];
    
    // Reverse order of pivots to back-substitute
    for &(r, c) in pivot_map.iter().rev() {
        // x[c] = t[r] - sum(m[r][k]*x[k] for k > c)
        let mut val = t[r];
        for k in (c+1)..cols {
            if m[r][k] == 1 {
                val ^= x[k];
            }
        }
        x[c] = val;
    }
    
    Some(x)
}

fn find(parent: &mut Vec<usize>, i: usize) -> usize {
    if parent[i] != i {
        parent[i] = find(parent, parent[i]);
    }
    parent[i]
}

fn union(parent: &mut Vec<usize>, i: usize, j: usize) {
    let root_i = find(parent, i);
    let root_j = find(parent, j);
    if root_i != root_j {
        parent[root_i] = root_j;
    }
}

pub fn run_lsd(
    check_matrix: &SparseBitMatrix,
    syndrome: &Array1<Bit>,
    posterior_llrs: &Array1<f64>,
    prior_llrs: &Array1<f64>,
    lsd_order: usize,
    lsd_method: &str
) -> LsdResult {
    let start_time = Instant::now();
    let num_checks = check_matrix.rows();
    let num_bits = check_matrix.cols();

    let is_csr = check_matrix.is_csr();
    let check_matrix_csr = if is_csr {
        None
    } else {
        Some(check_matrix.to_csr())
    };
    
    // Precompute Variable to Check adjacency (simple vec of vecs)
    let mut var_to_checks: Vec<Vec<usize>> = vec![Vec::new(); num_bits];
    
    if check_matrix.is_csr() {
        for (check_idx, row) in check_matrix.outer_iterator().enumerate() {
            for &bit_idx in row.indices() {
                var_to_checks[bit_idx].push(check_idx);
            }
        }
    } else {
        // CSC: outer iterator is over columns (bits)
        for (bit_idx, col) in check_matrix.outer_iterator().enumerate() {
            for &check_idx in col.indices() {
                var_to_checks[bit_idx].push(check_idx);
            }
        }
    }

    // 1. Initialize clusters
    // We only care about syndromes that are 1
    let mut clusters: Vec<LsdCluster> = Vec::new();
    let mut check_to_cluster: Vec<Option<usize>> = vec![None; num_checks];
    let mut bit_to_cluster: Vec<Option<usize>> = vec![None; num_bits];
    
    for (i, &s) in syndrome.iter().enumerate() {
        if s == 1 {
            let id = clusters.len();
            let c = LsdCluster::new(id, i);
            clusters.push(c);
            check_to_cluster[i] = Some(id);
        }
    }
    
    let method = LsdMethod::from_str(lsd_method);
    let _ = method; // suppress unused warning for now if logic is generic
    
    // Use simple Union-Find-like logic for merging.
    let mut parent: Vec<usize> = (0..clusters.len()).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize {
        if p[i] != i {
            p[i] = find(p, p[i]);
        }
        p[i]
    }
    fn union(p: &mut Vec<usize>, i: usize, j: usize) {
        let root_i = find(p, i);
        let root_j = find(p, j);
        if root_i != root_j {
            p[root_i] = root_j; // merge i into j
        }
    }

    // Main Loop
    let mut active = true;
    let mut iter = 0;
    while active && iter < 100 { // Safety limit
        active = false;
        iter += 1;
        
        // Growth Step
        for cid in 0..clusters.len() {
            // Only process if this is a root cluster and active and not valid
            if parent[cid] == cid {
                 if !clusters[cid].active || clusters[cid].valid {
                     continue;
                 }
            } else {
                continue;
            }

            // Find candidates
            let mut candidates: Vec<usize> = Vec::new();
            // Iterate over all boundary checks
            for &chk in &clusters[cid].boundary_check_nodes {
                 // get neib bits
                 let row_view = if is_csr {
                     check_matrix.outer_view(chk)
                 } else {
                     check_matrix_csr.as_ref().and_then(|m| m.outer_view(chk))
                 };
                 if let Some(row_vec) = row_view {
                     for &bit in row_vec.indices() {
                         // Check if bit is already in THIS cluster
                         let bit_owner = bit_to_cluster[bit].map(|id| find(&mut parent, id));
                         if bit_owner != Some(cid) {
                             candidates.push(bit);
                         }
                     }
                 }
            }
            
            if candidates.is_empty() {
                 clusters[cid].active = false;
                 continue;
            }
            
            // Heuristic: Sort by signed LLR ascending. Using POSTERIOR LLRs.
            candidates.sort_by(|&a, &b| posterior_llrs[a].partial_cmp(&posterior_llrs[b]).unwrap_or(std::cmp::Ordering::Equal));
            candidates.dedup();
            
            // Add bits
            let num_to_add = if lsd_order > 0 { 1 } else { 1 };
            let mut added_any = false;

            for i in 0..std::cmp::min(candidates.len(), num_to_add) {
                let bit = candidates[i];
                
                // Add bit to current cluster
                clusters[cid].bit_nodes.insert(bit);
                
                // If bit belonged to another cluster, we need to merge THAT cluster into THIS one?
                if let Some(other_id) = bit_to_cluster[bit] {
                    let root_other = find(&mut parent, other_id);
                    if root_other != cid {
                        union(&mut parent, root_other, cid);
                    }
                }
                bit_to_cluster[bit] = Some(cid); // Mark as owned by cid (root)
                
                // Add incident checks
                for &chk in &var_to_checks[bit] {
                    if !clusters[cid].check_nodes.contains(&chk) {
                         // New check
                         // If check belongs to another cluster, merge
                         if let Some(other_id) = check_to_cluster[chk] {
                             let root_other = find(&mut parent, other_id);
                             if root_other != cid {
                                 union(&mut parent, root_other, cid);
                             }

                         }
                         
                         clusters[cid].check_nodes.insert(chk);
                         clusters[cid].boundary_check_nodes.insert(chk);
                         check_to_cluster[chk] = Some(cid);
                    }
                }
                added_any = true;
            }
            
            if added_any {
                active = true;
            } else {
                clusters[cid].active = false;
            }
        }
        
        // Consolidate Merges
        for i in 0..clusters.len() {
            let root = find(&mut parent, i);
            if root != i {
                // i is merged into root.
                
                // We clone content of i and add to root.
                let bit_nodes: Vec<usize> = clusters[i].bit_nodes.iter().cloned().collect();
                let check_nodes: Vec<usize> = clusters[i].check_nodes.iter().cloned().collect();
                let enclosed: Vec<usize> = clusters[i].enclosed_syndromes.iter().cloned().collect();
                // Boundary checks also merge
                let boundary: Vec<usize> = clusters[i].boundary_check_nodes.iter().cloned().collect();

                {
                    let root_cluster = &mut clusters[root];
                    root_cluster.bit_nodes.extend(bit_nodes);
                    root_cluster.check_nodes.extend(check_nodes);
                    root_cluster.enclosed_syndromes.extend(enclosed);
                    root_cluster.boundary_check_nodes.extend(boundary);
                }
                
                // Mark i as empty/inactive
                clusters[i].bit_nodes.clear();
                clusters[i].check_nodes.clear();
                clusters[i].boundary_check_nodes.clear();
                clusters[i].enclosed_syndromes.clear();
                clusters[i].solution.clear();
                clusters[i].active = false;
                clusters[i].valid = true; // effectively "done"
            }
        }
        
        // Check Validity of Roots
        for cid in 0..clusters.len() {
            if parent[cid] == cid && clusters[cid].active && !clusters[cid].valid {
                 // Construct H_local
                 let checks: Vec<usize> = clusters[cid].check_nodes.iter().cloned().collect();
                 let bits: Vec<usize> = clusters[cid].bit_nodes.iter().cloned().collect();
                 
                 if bits.is_empty() { 
                      clusters[cid].active = false;
                      continue; 
                 }
                 
                 // Syndrome vector locally
                 let mut s_local = vec![0u8; checks.len()];
                 let mut has_syndrome = false;
                 for (row_i, &chk_idx) in checks.iter().enumerate() {
                     if syndrome[chk_idx] == 1 {
                         s_local[row_i] = 1;
                         has_syndrome = true;
                     }
                 }
                 
                 if !has_syndrome {
                     clusters[cid].valid = true;
                     clusters[cid].active = false;
                     continue;
                 }
                 
                 // Map bit_idx -> col_i (dense map for speed)
                 let mut bit_map: Vec<i32> = vec![-1; num_bits];
                 for (j, &b) in bits.iter().enumerate() {
                     bit_map[b] = j as i32;
                 }
                 
                 let mut h_local = vec![vec![0u8; bits.len()]; checks.len()];
                 for (row_i, &chk_idx) in checks.iter().enumerate() {
                     let row_view = if is_csr {
                         check_matrix.outer_view(chk_idx)
                     } else {
                         check_matrix_csr.as_ref().and_then(|m| m.outer_view(chk_idx))
                     };
                     if let Some(view) = row_view {
                         for &col_idx in view.indices() {
                             let col_j = bit_map[col_idx];
                             if col_j >= 0 {
                                 h_local[row_i][col_j as usize] = 1;
                             }
                         }
                     }
                 }
                 
                 if let Some(sol) = solve_system_rows(&h_local, &s_local) {
                     clusters[cid].valid = true;
                     clusters[cid].active = false; // Stop growing
                     
                     // Store solution mapping
                     for (j, &b) in bits.iter().enumerate() {
                         if sol[j] == 1 {
                             clusters[cid].solution.insert(b, 1);
                         }
                     }
                 }
            }
        }
    }
    
    // Build Result
    let mut cluster_sizes = Vec::new();
    let mut cluster_llrs = Vec::new();
    let mut cluster_ids = vec![0; num_bits];
    let mut lsd_correction = Array1::zeros(num_bits);
    
    let mut output_id = 1;
    for cid in 0..clusters.len() {
        if parent[cid] == cid { 
             if !clusters[cid].bit_nodes.is_empty() {
                 let size = clusters[cid].bit_nodes.len();
                 let mut llr_sum = 0.0;
                 for &b in &clusters[cid].bit_nodes {
                     llr_sum += prior_llrs[b].abs(); // Using PRIOR LLRs for metrics
                     cluster_ids[b] = output_id;
                 }
                 cluster_sizes.push(size);
                 cluster_llrs.push(llr_sum);
                 output_id += 1;
             }
             
             // Apply solution if present
             for (&bit, &val) in &clusters[cid].solution {
                 if val == 1 {
                     lsd_correction[bit] = 1;
                 }
             }
        }
    }
    
    LsdResult {
        cluster_sizes,
        cluster_llrs,
        cluster_ids,
        elapsed_time_micros: start_time.elapsed().as_micros() as u64,
        lsd_correction: Some(lsd_correction),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use crate::bipartite_graph::BipartiteGraph;
    use crate::decoder::SparseBitMatrix;

    #[test]
    fn test_lsd_basic_clustering() {
        // H = [1 1 0; 0 1 1; 1 0 1]
        let h_dense = arr2(&[
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1]
        ]);
        // SparseBitMatrix is CsMat<u8>, which implements BipartiteGraph
        let h: SparseBitMatrix = SparseBitMatrix::from_dense(h_dense);
        
        // Check 0 and 1 are unsatisfied.
        let syndrome = arr1(&[1, 1, 0]);
        // Bit 1 (col 1) is connected to check 0 and 1.
        // Make bit 1 weak (low magnitude LLR), others strong.
        let llrs = arr1(&[-2.0, -0.5, -2.0]);

        // LSD_CS is one of the methods handled in from_str
        let result = run_lsd(&h, &syndrome, &llrs, 0, "LSD_CS");

        // We expect checks 0 and 1 to merge because they share bit 1 chosen by LLR sorting.
        // Result should have non-empty clusters.
        assert!(!result.cluster_sizes.is_empty());
        
        // Bit 1 should be in a cluster
        assert!(result.cluster_ids[1] > 0);
        
        // Since bit 1 connects check 0 and 1, and it's the weakest, 
        // it serves as a bridge. They should likely merge into one cluster.
        let covered_bits: usize = result.cluster_ids.iter().filter(|&&x| x > 0).count();
        assert!(covered_bits >= 1);
    }

    #[test]
    fn test_lsd_merging() {
        // H = 
        // C0: 1 0 0  (connects B0)
        // C1: 1 1 0  (connects B0, B1)
        // C2: 0 1 0  (connects B1)
        let h_dense = arr2(&[
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]);
        let h: SparseBitMatrix = SparseBitMatrix::from_dense(h_dense);
        
        // C0 and C2 unsatisfied. C1 satisfied.
        let syndrome = arr1(&[1, 0, 1]);
        
        // B0 and B1 weak.
        let llrs = arr1(&[0.1, 0.1, 5.0]);
        
        // LSD_E involves expansion
        let result = run_lsd(&h, &syndrome, &llrs, 0, "LSD_E");
        
        // C0 expands B0. C2 expands B1.
        // B0 connects to C1. B1 connects to C1.
        // C1 becomes boundary for both.
        // They should merge via C1.
        
        assert!(!result.cluster_sizes.is_empty());
        
        // Check B0 and B1 are clustered
        assert!(result.cluster_ids[0] > 0);
        assert!(result.cluster_ids[1] > 0);
        
        // Check they are in the SAME cluster
        assert_eq!(result.cluster_ids[0], result.cluster_ids[1]);
    }
}

