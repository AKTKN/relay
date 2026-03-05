use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;

use super::config::{SelectionMode, WeightedSelectionMode};

#[derive(Clone)]
pub struct PopulationMember {
    pub posterior: Array1<f64>,
    pub fitness: f64,
}

pub fn build_next_generation(
    population: &[PopulationMember],
    elite_count: usize,
    mutation_rate: f64,
    mutation_llr_abs_threshold: f64,
    selection_mode: SelectionMode,
    weighted_mode: WeightedSelectionMode,
    tournament_size: usize,
    rng: &mut StdRng,
) -> Vec<Array1<f64>> {
    if population.is_empty() {
        return Vec::new();
    }

    let m = population.len();
    let n = population[0].posterior.len();

    let mut ranked: Vec<usize> = (0..m).collect();
    ranked.sort_by(|a, b| {
        population[*b]
            .fitness
            .partial_cmp(&population[*a].fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut children = Vec::<Array1<f64>>::with_capacity(m);
    for idx in ranked.into_iter().take(elite_count.min(m)) {
        children.push(population[idx].posterior.clone());
    }

    while children.len() < m {
        let pa = select_parent(population, selection_mode, weighted_mode, tournament_size, rng);
        let pb = select_parent(population, selection_mode, weighted_mode, tournament_size, rng);

        let mut child = Array1::<f64>::zeros(n);
        for v in 0..n {
            child[v] = if rng.gen_bool(0.5) {
                pa.posterior[v]
            } else {
                pb.posterior[v]
            };
        }

        mutate_low_confidence(
            &mut child,
            mutation_rate,
            mutation_llr_abs_threshold,
            rng,
        );

        children.push(child);
    }

    children
}

fn select_parent<'a>(
    population: &'a [PopulationMember],
    selection_mode: SelectionMode,
    weighted_mode: WeightedSelectionMode,
    tournament_size: usize,
    rng: &mut StdRng,
) -> &'a PopulationMember {
    match selection_mode {
        SelectionMode::Weighted => weighted_select(population, weighted_mode, rng),
        SelectionMode::Tournament => tournament_select(population, tournament_size, rng),
    }
}

fn tournament_select<'a>(
    population: &'a [PopulationMember],
    tournament_size: usize,
    rng: &mut StdRng,
) -> &'a PopulationMember {
    let m = population.len();
    let mut best_idx = rng.gen_range(0..m);
    let mut best = population[best_idx].fitness;

    for _ in 1..tournament_size.max(2) {
        let idx = rng.gen_range(0..m);
        if population[idx].fitness > best {
            best = population[idx].fitness;
            best_idx = idx;
        }
    }

    &population[best_idx]
}

fn weighted_select<'a>(
    population: &'a [PopulationMember],
    weighted_mode: WeightedSelectionMode,
    rng: &mut StdRng,
) -> &'a PopulationMember {
    match weighted_mode {
        WeightedSelectionMode::Softmax => softmax_select(population, rng),
        WeightedSelectionMode::Rank => rank_select(population, rng),
    }
}

fn softmax_select<'a>(population: &'a [PopulationMember], rng: &mut StdRng) -> &'a PopulationMember {
    let max_fit = population
        .iter()
        .map(|p| p.fitness)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let mut weights = Vec::<f64>::with_capacity(population.len());
    let mut sum_w = 0.0;
    for member in population {
        let w = (member.fitness - max_fit).exp();
        let ww = if w.is_finite() { w.max(1e-12) } else { 1e-12 };
        weights.push(ww);
        sum_w += ww;
    }

    if !sum_w.is_finite() || sum_w <= 0.0 {
        return &population[rng.gen_range(0..population.len())];
    }

    let mut threshold = rng.gen_range(0.0..sum_w);
    for (idx, w) in weights.iter().enumerate() {
        threshold -= *w;
        if threshold <= 0.0 {
            return &population[idx];
        }
    }

    &population[population.len() - 1]
}

fn rank_select<'a>(population: &'a [PopulationMember], rng: &mut StdRng) -> &'a PopulationMember {
    let mut ranked: Vec<usize> = (0..population.len()).collect();
    ranked.sort_by(|a, b| {
        population[*b]
            .fitness
            .partial_cmp(&population[*a].fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let len = ranked.len();
    let mut sum_w = 0.0;
    let mut weights = vec![0.0; len];
    for (pos, _) in ranked.iter().enumerate() {
        let w = (len - pos) as f64;
        weights[pos] = w;
        sum_w += w;
    }

    let mut threshold = rng.gen_range(0.0..sum_w.max(1e-12));
    for (pos, w) in weights.iter().enumerate() {
        threshold -= *w;
        if threshold <= 0.0 {
            return &population[ranked[pos]];
        }
    }

    &population[ranked[len - 1]]
}

fn mutate_low_confidence(
    child: &mut Array1<f64>,
    mutation_rate: f64,
    abs_threshold: f64,
    rng: &mut StdRng,
) {
    let p = mutation_rate.clamp(0.0, 1.0);
    for val in child.iter_mut() {
        if val.abs() < abs_threshold && rng.gen_bool(p) {
            *val = -*val;
        }
    }
}
