use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

use crate::bp::min_sum::{MinSumBPDecoder, MinSumDecoderConfig};
use crate::decoder::{BPExtraResult, Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix};

pub mod config;
pub mod evaluate;
pub mod ga_ops;
pub mod init_population;
pub mod trace;

use config::{GammaMode, SLGMBPDecoderConfig};
use evaluate::{fitness_from_final_marginal, fitness_from_ms_cumsum, residual_weight};
use ga_ops::{build_next_generation, PopulationMember};
use init_population::initialize_llr_population;

#[derive(Clone)]
pub struct SLGMBPDecoder {
    check_matrix: Arc<SparseBitMatrix>,
    base_min_sum_config: Arc<MinSumDecoderConfig>,
    config: Arc<SLGMBPDecoderConfig>,
}

impl SLGMBPDecoder {
    pub fn new(
        check_matrix: Arc<SparseBitMatrix>,
        min_sum_config: Arc<MinSumDecoderConfig>,
        config: Arc<SLGMBPDecoderConfig>,
    ) -> Self {
        Self {
            check_matrix,
            base_min_sum_config: min_sum_config,
            config,
        }
    }

    fn prior_llr(&self) -> Array1<f64> {
        self.base_min_sum_config.log_prior_ratios()
    }

    fn run_min_sum_phase(
        &self,
        detectors: ArrayView1<'_, Bit>,
        init_llr: &Array1<f64>,
    ) -> (Array1<Bit>, Array1<f64>, usize, bool, f64) {
        let priors_prob = self.base_min_sum_config.error_priors.clone();
        let cfg = self
            .config
            .min_sum_template_from_priors(priors_prob, self.config.t_ms);

        let mut decoder = MinSumBPDecoder::<f64>::new(self.check_matrix.clone(), Arc::new(cfg));

        decoder.current_iteration = 0;
        decoder.set_log_prior_ratio_f64(init_llr.clone());
        decoder.set_posterior_ratios_f64(init_llr.clone());
        decoder.initialize_check_to_variable();
        decoder.initialize_variable_to_check();

        let mut decoded = decoder.compute_decoded_detectors();
        let mut success = false;
        let mut llr_cumsum_abs_sum = 0.0;

        for _ in 0..self.config.t_ms {
            decoder.run_iteration(detectors);
            decoder.current_iteration += 1;
            let posterior = decoder.posterior_ratios_f64();
            llr_cumsum_abs_sum += posterior.iter().sum::<f64>().abs();
            decoded = decoder.compute_decoded_detectors();
            success = decoder.check_convergence(detectors, decoded.view());
            if success {
                break;
            }
        }

        (
            decoder.current_decoding().clone(),
            decoder.posterior_ratios_f64(),
            decoder.current_iteration,
            success,
            llr_cumsum_abs_sum,
        )
    }

    fn run_mem_bp_phase(
        &self,
        detectors: ArrayView1<'_, Bit>,
        prior_llr: &Array1<f64>,
        child_llr: &Array1<f64>,
        gamma: f64,
    ) -> (Array1<Bit>, Array1<f64>, usize, bool) {
        let priors_prob = self.base_min_sum_config.error_priors.clone();
        let cfg = self
            .config
            .min_sum_template_from_priors(priors_prob, self.config.t_mem);
        let mut decoder = MinSumBPDecoder::<f64>::new(self.check_matrix.clone(), Arc::new(cfg));

        let mut prev_marginal = child_llr.mapv(|v| self.config.eta * v);

        decoder.current_iteration = 0;
        decoder.set_log_prior_ratio_f64(prev_marginal.clone());
        decoder.set_posterior_ratios_f64(prev_marginal.clone());
        decoder.initialize_check_to_variable();
        decoder.initialize_variable_to_check();

        let mut decoded = decoder.compute_decoded_detectors();
        let mut success = false;

        for _ in 0..self.config.t_mem {
            let bias = prior_llr.mapv(|p| (1.0 - gamma) * p) + prev_marginal.mapv(|m| gamma * m);
            decoder.set_log_prior_ratio_f64(bias);

            decoder.run_iteration(detectors);
            decoder.current_iteration += 1;

            prev_marginal = decoder.posterior_ratios_f64();
            decoded = decoder.compute_decoded_detectors();
            success = decoder.check_convergence(detectors, decoded.view());
            if success {
                break;
            }
        }

        (
            decoder.current_decoding().clone(),
            decoder.posterior_ratios_f64(),
            decoder.current_iteration,
            success,
        )
    }

    fn sample_generation_gamma(&self, rng: &mut StdRng) -> f64 {
        match self.config.gamma_mode {
            GammaMode::Fixed => self.config.gamma_fixed,
            // User requirement: sample once per generation, then keep fixed inside that generation.
            GammaMode::IntervalRandomPerGeneration => {
                let (a, b) = self.config.gamma_interval;
                let lo = a.min(b);
                let hi = a.max(b);
                if (hi - lo).abs() < f64::EPSILON {
                    lo
                } else {
                    rng.gen_range(lo..=hi)
                }
            }
        }
    }

    fn max_iter(&self) -> usize {
        self.config.t_ms + self.config.g_max * self.config.t_mem
    }
}

impl Decoder for SLGMBPDecoder {
    fn check_matrix(&self) -> Arc<SparseBitMatrix> {
        self.check_matrix.clone()
    }

    fn log_prior_ratios(&mut self) -> Array1<f64> {
        self.base_min_sum_config.log_prior_ratios()
    }

    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult {
        let prior_llr = self.prior_llr();
        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // Phase 1: diversity initialization + Min-Sum BP.
        let init_population = initialize_llr_population(
            &prior_llr,
            self.config.init_perturbation_mode,
            self.config.ensemble_size,
            self.config.sigma2,
            self.config.delta,
            &mut rng,
        );

        let mut members = Vec::<PopulationMember>::with_capacity(init_population.len());
        let mut best_decoding = Array1::<Bit>::zeros(self.check_matrix.cols());
        let mut best_posterior = prior_llr.clone();
        let mut best_fitness = f64::NEG_INFINITY;
        let mut generation_best_fitness = Vec::<f64>::new();
        let mut gamma_history = Vec::<f64>::new();
        let mut phase1_iterations = 0usize;
        // Store (iters, llr_cost, decoding, posterior) for tie-breaking.
        // For equal iteration counts, prefer the solution with smaller sum(bit_i * prior_llr_i).
        let mut phase1_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;

        for init_llr in &init_population {
            let (decoding, posterior, iters, success, cumsum_abs) =
                self.run_min_sum_phase(detectors, init_llr);
            // Ensemble members are treated as parallel within this layer.
            phase1_iterations = phase1_iterations.max(iters);
            let residual = residual_weight(&self.get_detectors(decoding.view()), detectors);
            let fitness = fitness_from_ms_cumsum(
                residual,
                cumsum_abs,
                self.config.fitness_alpha,
                self.config.fitness_beta,
            );

            if fitness > best_fitness {
                best_fitness = fitness;
                best_decoding = decoding.clone();
                best_posterior = posterior.clone();
            }

            if success {
                let solution_llr_cost = decoding
                    .iter()
                    .zip(prior_llr.iter())
                    .map(|(&bit, &llr)| (bit as f64) * llr)
                    .sum::<f64>();
                let replace = match phase1_success {
                    None => true,
                    Some((best_iter, best_llr_cost, _, _)) => {
                        iters < best_iter
                            || (iters == best_iter && solution_llr_cost < best_llr_cost)
                    }
                };
                if replace {
                    phase1_success = Some((
                        iters,
                        solution_llr_cost,
                        decoding.clone(),
                        posterior.clone(),
                    ));
                }
                // If a member converged in one iteration we already reached the minimum.
                if iters == 1 {
                    break;
                }
                continue;
            }

            members.push(PopulationMember { posterior, fitness });
        }

        if let Some((phase1_success_iters, _, phase1_decoding, phase1_posterior)) = phase1_success {
            return DecodeResult {
                decoding: phase1_decoding.clone(),
                decoded_detectors: self.get_detectors(phase1_decoding.view()),
                posterior_ratios: phase1_posterior,
                success: true,
                decoding_quality: self.get_decoding_quality(phase1_decoding.view()),
                iterations: phase1_success_iters,
                max_iter: self.max_iter(),
                extra: BPExtraResult::SLGMBPTrace {
                    phase1_converged: true,
                    phase1_iterations: phase1_success_iters,
                    total_iterations: phase1_success_iters,
                    generation_count: 0,
                    generation_best_fitness,
                    selected_solution_posterior: None,
                    residual_weight_history: Vec::new(),
                    gamma_history,
                },
            };
        }

        let mut total_iterations = phase1_iterations;

        // Phases 2-5: GA + Mem-BP generational search.
        let mut residual_weight_history = Vec::<usize>::new();

        for _gen in 0..self.config.g_max {
            let children = build_next_generation(
                &members,
                self.config.elite_count,
                self.config.mutation_rate,
                self.config.mutation_llr_abs_threshold,
                self.config.selection_mode,
                self.config.weighted_selection_mode,
                self.config.tournament_size,
                &mut rng,
            );

            let generation_gamma = self.sample_generation_gamma(&mut rng);
            gamma_history.push(generation_gamma);

            let mut next_members = Vec::<PopulationMember>::with_capacity(children.len());
            let mut gen_best = f64::NEG_INFINITY;
            let mut generation_iterations = 0usize;
            // Store (iters, llr_cost, decoding, posterior) for tie-breaking.
            let mut generation_success: Option<(usize, f64, Array1<Bit>, Array1<f64>)> = None;

            for child in &children {
                let (decoding, posterior, iters, success) =
                    self.run_mem_bp_phase(detectors, &prior_llr, child, generation_gamma);

                let decoded_detectors = self.get_detectors(decoding.view());
                let residual = residual_weight(&decoded_detectors, detectors);
                residual_weight_history.push(residual);
                let fitness = fitness_from_final_marginal(
                    residual,
                    &posterior,
                    self.config.fitness_alpha,
                    self.config.fitness_beta,
                );
                gen_best = gen_best.max(fitness);

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_decoding = decoding.clone();
                    best_posterior = posterior.clone();
                }

                if success {
                    let solution_llr_cost = decoding
                        .iter()
                        .zip(prior_llr.iter())
                        .map(|(&bit, &llr)| (bit as f64) * llr)
                        .sum::<f64>();
                    let replace = match generation_success {
                        None => true,
                        Some((best_iter, best_llr_cost, _, _)) => {
                            iters < best_iter
                                || (iters == best_iter && solution_llr_cost < best_llr_cost)
                        }
                    };
                    if replace {
                        generation_success = Some((
                            iters,
                            solution_llr_cost,
                            decoding.clone(),
                            posterior.clone(),
                        ));
                    }
                    // Minimum possible iteration for this layer is 1.
                    if iters == 1 {
                        break;
                    }
                    continue;
                }

                // No success yet in this generation: account the layer by max iters.
                generation_iterations = generation_iterations.max(iters);
                next_members.push(PopulationMember { posterior, fitness });
            }

            if let Some((gen_success_iters, _, gen_success_decoding, gen_success_posterior)) = generation_success {
                total_iterations += gen_success_iters;
                let gen_decoded_detectors = self.get_detectors(gen_success_decoding.view());
                return DecodeResult {
                    decoding: gen_success_decoding.clone(),
                    decoded_detectors: gen_decoded_detectors,
                    posterior_ratios: gen_success_posterior.clone(),
                    success: true,
                    decoding_quality: self.get_decoding_quality(gen_success_decoding.view()),
                    iterations: total_iterations,
                    max_iter: self.max_iter(),
                    extra: BPExtraResult::SLGMBPTrace {
                        phase1_converged: false,
                        phase1_iterations,
                        total_iterations,
                        generation_count: gamma_history.len(),
                        generation_best_fitness,
                        selected_solution_posterior: Some(gen_success_posterior),
                        residual_weight_history,
                        gamma_history,
                    },
                };
            }

            total_iterations += generation_iterations;
            generation_best_fitness.push(gen_best);
            members = next_members;
        }

        DecodeResult {
            decoding: best_decoding.clone(),
            decoded_detectors: self.get_detectors(best_decoding.view()),
            posterior_ratios: best_posterior.clone(),
            success: false,
            decoding_quality: self.get_decoding_quality(best_decoding.view()),
            iterations: total_iterations,
            max_iter: self.max_iter(),
            extra: BPExtraResult::SLGMBPTrace {
                phase1_converged: false,
                phase1_iterations,
                total_iterations,
                generation_count: gamma_history.len(),
                generation_best_fitness,
                selected_solution_posterior: Some(best_posterior),
                residual_weight_history,
                gamma_history,
            },
        }
    }
}

impl DecoderRunner for SLGMBPDecoder {}
