use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;

use super::config::InitPerturbationMode;

pub fn initialize_llr_population(
    prior_llr: &Array1<f64>,
    mode: InitPerturbationMode,
    ensemble_size: usize,
    sigma2: f64,
    delta: f64,
    rng: &mut StdRng,
) -> Vec<Array1<f64>> {
    let mut out = Vec::<Array1<f64>>::with_capacity(ensemble_size);

    for _ in 0..ensemble_size {
        out.push(sample_perturbed_llr(prior_llr, mode, sigma2, delta, rng));
    }

    out
}

pub fn sample_perturbed_llr(
    prior_llr: &Array1<f64>,
    mode: InitPerturbationMode,
    sigma2: f64,
    delta: f64,
    rng: &mut StdRng,
) -> Array1<f64> {
    let mut llr = prior_llr.clone();
    let sigma = sigma2.max(0.0).sqrt();

    match mode {
        InitPerturbationMode::AdditiveGaussian => {
            for v in llr.iter_mut() {
                if sigma > 0.0 {
                    *v += sigma * sample_standard_normal(rng);
                }
            }
        }
        InitPerturbationMode::MultiplicativeUniform => {
            let lo = 1.0 - delta.abs();
            let hi = 1.0 + delta.abs();
            for v in llr.iter_mut() {
                let scale = rng.gen_range(lo..=hi);
                *v *= scale;
            }
        }
    }

    llr
}

fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    // Box-Muller transform.
    let u1 = rng.gen_range(f64::EPSILON..1.0);
    let u2 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
