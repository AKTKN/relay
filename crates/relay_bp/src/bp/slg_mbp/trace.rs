use ndarray::Array1;

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
}
