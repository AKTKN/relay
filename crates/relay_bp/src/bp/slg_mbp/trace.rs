use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SLGMBPDynamicsEntry {
    pub stage: String,
    pub generation_index: usize,
    pub member_index: usize,
    pub residual_syndrome_count: usize,
    pub residual_adjacent_variable_count: usize,
    pub score: f64,
    pub residual_adjacent_variable_indices: Vec<usize>,
    pub residual_adjacent_variable_llrs: Vec<f32>,
    pub converged: bool,
    pub iteration_count: usize,
    pub estimated_error_weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SLGMBPDetailedDynamicsTrace {
    pub observed_syndrome_weight: usize,
    pub entries: Vec<SLGMBPDynamicsEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SLGMBPScoreSpikeGenerationSnapshot {
    pub generation_index: usize,
    pub posterior_llr_all_variables: Vec<f64>,
    pub memory_strength_all_variables: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SLGMBPScoreSpikeTrace {
    pub spike_generation_index: usize,
    pub previous_generation_index: usize,
    pub spike_prev_score: f64,
    pub spike_score: f64,
    pub spike_delta_score: f64,
    pub spike_delta_order_log10: Option<f64>,
    pub window_start_generation: usize,
    pub window_end_generation: usize,
    pub spike_residual_adjacent_variable_indices: Vec<usize>,
    pub snapshots: Vec<SLGMBPScoreSpikeGenerationSnapshot>,
}

#[derive(Clone, Debug)]
pub struct SLGMBPRuntimeTrace {
    pub phase1_converged: bool,
    pub phase1_iterations: usize,
    pub total_iterations: usize,
    pub generation_count: usize,
    pub generation_best_fitness: Vec<f64>,
    pub selected_solution_posterior: Option<Array1<f64>>,
    pub residual_weight_history: Vec<usize>,
    pub gamma_history: Vec<f64>,
    pub detailed_dynamics: Option<SLGMBPDetailedDynamicsTrace>,
    pub score_spike_trace: Option<SLGMBPScoreSpikeTrace>,
}
