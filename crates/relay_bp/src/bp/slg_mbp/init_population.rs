use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;

use super::config::InitPerturbationMode;

#[derive(Clone, Debug, PartialEq)]
pub enum PerturbationState {
    None,
    AdditiveGaussian(Array1<f64>),
    MultiplicativeUniform(Array1<f64>),
}

pub fn initialize_llr_population(
    prior_llr: &Array1<f64>,
    mode: InitPerturbationMode,
    ensemble_size: usize,
    sigma2: f64,
    delta: f64,
    rng: &mut StdRng,
) -> Vec<(Array1<f64>, PerturbationState)> {
    let mut out = Vec::<(Array1<f64>, PerturbationState)>::with_capacity(ensemble_size);

    for _ in 0..ensemble_size {
        let state = sample_perturbation_state(prior_llr.len(), mode, sigma2, delta, rng);
        let perturbed = apply_perturbation_state(prior_llr, &state);
        out.push((perturbed, state));
    }

    out
}

pub fn sample_perturbation_state(
    n_variables: usize,
    mode: InitPerturbationMode,
    sigma2: f64,
    delta: f64,
    rng: &mut StdRng,
) -> PerturbationState {
    match mode {
        InitPerturbationMode::AdditiveGaussian => {
            let sigma = sigma2.max(0.0).sqrt();
            let noise = Array1::from_shape_simple_fn(n_variables, || {
                if sigma > 0.0 {
                    sigma * sample_standard_normal(rng)
                } else {
                    0.0
                }
            });
            PerturbationState::AdditiveGaussian(noise)
        }
        InitPerturbationMode::MultiplicativeUniform => {
            let lo = 1.0 - delta.abs();
            let hi = 1.0 + delta.abs();
            let scale = Array1::from_shape_simple_fn(n_variables, || rng.gen_range(lo..=hi));
            PerturbationState::MultiplicativeUniform(scale)
        }
    }
}

pub fn apply_perturbation_state(
    prior_llr: &Array1<f64>,
    state: &PerturbationState,
) -> Array1<f64> {
    match state {
        PerturbationState::None => prior_llr.clone(),
        PerturbationState::AdditiveGaussian(noise) => prior_llr + noise,
        PerturbationState::MultiplicativeUniform(scale) => prior_llr * scale,
    }
}

pub fn sample_perturbed_llr(
    prior_llr: &Array1<f64>,
    mode: InitPerturbationMode,
    sigma2: f64,
    delta: f64,
    rng: &mut StdRng,
) -> Array1<f64> {
    let state = sample_perturbation_state(prior_llr.len(), mode, sigma2, delta, rng);
    apply_perturbation_state(prior_llr, &state)
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    // Box-Muller transform.
    let u1 = rng.gen_range(f64::EPSILON..1.0);
    let u2 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::{apply_perturbation_state, PerturbationState};
    use ndarray::array;

    #[test]
    fn additive_state_applies_elementwise_noise() {
        let prior = array![1.0, -2.0, 0.5];
        let noise = array![0.25, -0.5, 0.75];
        let perturbed = apply_perturbation_state(&prior, &PerturbationState::AdditiveGaussian(noise));
        assert_eq!(perturbed, array![1.25, -2.5, 1.25]);
    }

    #[test]
    fn multiplicative_state_applies_elementwise_scale() {
        let prior = array![1.0, -2.0, 0.5];
        let scale = array![2.0, 0.5, -1.0];
        let perturbed = apply_perturbation_state(
            &prior,
            &PerturbationState::MultiplicativeUniform(scale),
        );
        assert_eq!(perturbed, array![2.0, -1.0, -0.5]);
    }
}
