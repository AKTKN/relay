// Import necessary components and modules
use crate::decoder::{Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix, Mod2Mul}; // Besic type and trait imports
use ndarray::{Array1, ArrayView1}; // 1-dimentional ndarray crate for error and syndrome vecrtors
use std::sync::Arc; // Smart pointer for shared owenership, this is useful for sharing data like chack matrices across multiple decoders.


#[derive(Clone, Debug, PartialEq)]
pub enum SelectionStrategy{
    MostLikely,
    MajorityVote,
}

impl Default for SelectionStrategy{
    fn default() -> Self {
        SelectionStrategy::MostLikely
    }
}

// EnsembleDecoder struct definition
// store child decoders in a `decoders` field, which is a vector of boxed `Decoder` trait objects.
// - Vec<T>: A growable array type provided by the Rust standard library.
// - Box<T>: A smart pointer for heap allcation, allowing for dynamic dispatch of trait objects.
// - dyn Decoder: A trait object representig any type that implements the `Decoder` trait.
// - + Send: A marker trait indicating that the type can be safely transfered across thread boundaries.
#[derive(Clone)]
pub struct EnsembleDecoder{
    decoders: Vec<Box<dyn Decoder + Send>>,
    strategy: SelectionStrategy,
    original_log_priors: Option<Arc<Array1<f64>>>,
    observable_matrix: Option<Arc<SparseBitMatrix>>,
}

// Implement the constructer for EnsembleDecoder
impl EnsembleDecoder{
    /// Create a new EnsembleDecoder 
    pub fn new(
        decoders: Vec<Box<dyn Decoder + Send>>,
        strategy: SelectionStrategy,
        original_log_priors: Option<Arc<Array1<f64>>>,
        observable_matrix: Option<Arc<SparseBitMatrix>>,
    ) -> Self{
        if decoders.is_empty(){
            panic!("EnsembleDecoder requires at least one decoder.");
        }
        // 戦略と必要な情報が一致しているか簡単なチェック
        if strategy == SelectionStrategy::MostLikely && original_log_priors.is_none() {
            panic!("'MostLikely' strategy requires original_log_priors.");
        }
        if strategy == SelectionStrategy::MajorityVote && observable_matrix.is_none() {
            panic!("'MajorityVote' strategy requires observable_matrix.");
        }

        Self { 
            decoders,
            strategy,
            original_log_priors,
            observable_matrix,
        } 
    }
}

// Implement the Decoder trait for EnsembleDecoder
impl Decoder for EnsembleDecoder{
    /// return check matrix
    // In the future, we may want to get automorphism group og the check matrix
    fn check_matrix(&self) -> Arc<SparseBitMatrix>{
        self.decoders[0].check_matrix()
    }

    /// return log prior ratios of prior error probabilities
    /// In the future, we may want to add some pertubation to the prior ratios
    fn log_prior_ratios(&mut self) -> Array1<f64>{
        self.decoders[0].log_prior_ratios()
    }

    /// Return final decoding result after running multiple decoders
    // execute decoding for each decoder in the ensemble and collect their results.
    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult{
        let results: Vec<DecodeResult> = self
            .decoders
            .iter_mut()
            .map(|decoder| decoder.decode_detailed(detectors.view()))
            .collect();

        // Here we decide final correction 
        match self.strategy{
            SelectionStrategy::MostLikely =>{
                let log_priors = self.original_log_priors.as_ref().unwrap();

                results.into_iter()
                    .min_by(|a, b| {
                        // LLR = sum_{i where c_i=1} log(p_i / (1-p_i))
                        let llr_a = a.decoding.iter().zip(log_priors.iter()).filter(|(&c, _)| c == 1).map(|(_, &llr)| llr).sum::<f64>();
                        let llr_b = b.decoding.iter().zip(log_priors.iter()).filter(|(&c, _)| c == 1).map(|(_, &llr)| llr).sum::<f64>();
                        llr_a.partial_cmp(&llr_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("No results to compare for MostLikely strategy.")
            }
            
            SelectionStrategy::MajorityVote => {
                use std::collections::HashMap;
                let obs_matrix = self.observable_matrix.as_ref().unwrap();

                // 各補正がどの論理エラーに対応するか計算し、グループ化
                let mut coset_votes: HashMap<Array1<Bit>, Vec<DecodeResult>> = HashMap::new();
                for result in results {
                    let logical_error = obs_matrix.mul_mod2(&result.decoding);
                    coset_votes.entry(logical_error).or_default().push(result);
                }

                // 最も投票数の多いコセットを見つける
                let winning_coset_results = coset_votes.into_values()
                    .max_by_key(|v| v.len())
                    .expect("No results to vote on for MajorityVote strategy.");

                // 勝利したコセットの中から、最もハミング重みが小さいものを最終結果として選択
                winning_coset_results.into_iter()
                    .min_by_key(|r| r.decoding.sum())
                    .expect("Winning coset was empty.")
            }
        }
    }
}

// Implement the DecoderRunner trait for EnsembleDecoder
impl DecoderRunner for EnsembleDecoder {}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::bp::min_sum::MinSumDecoderConfig;
    use crate::bp::relay::{RelayDecoder, RelayDecoderConfig};
    use crate::dem::DetectorErrorModel;
    use crate::observable_decoder::ObservableDecoderRunner;
    use crate::utilities::test::get_test_data_path;
    use ndarray::{Array1, Array2};
    use ndarray_npy::read_npy;
    use std::sync::Arc;

    #[test]
    fn test_ensemble_relay_decode_basic() {
        // 1. テストデータの準備
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open");

        let ensemble_size = 3;
        let mut child_decoders : Vec<Box<dyn Decoder + Send>> = Vec::new();

        for _i in 0..ensemble_size{
            let bp_config_144_12_12 = Arc::new(MinSumDecoderConfig{
                error_priors: code_144_12_12.error_priors.clone(),
                max_iter: 200,
                alpha: None,
                alpha_iteration_scaling_factor: 0.,
                gamma0: Some(0.9),
                ..Default::default()
            });

            let relay_config = Arc::new(RelayDecoderConfig{
                pre_iter: 30,
                num_sets: 5,
                set_max_iter: 20,
                gamma_dist_interval: (-0.25, 0.60),
                ..Default::default()
            });
            
            let decoder = RelayDecoder::<f64>::new(
                Arc::new(code_144_12_12.detector_error_matrix.clone()),
                bp_config_144_12_12.clone(),
                relay_config.clone(),
            );

            child_decoders.push(Box::new(decoder));
        }

        let original_log_priors = Arc::new(code_144_12_12.error_priors.mapv(|p| (p/(1.0-p)).ln()));

        let mut ensemble_decoder = EnsembleDecoder::new(
            child_decoders,
            SelectionStrategy::MostLikely,
            Some(original_log_priors),
            None,
        );

        let _decode_result = ensemble_decoder.decode_detailed(
            detectors_144_12_12.row(0)
        );

        println!("Ensemble Relay Decoder Test Passed!");
    }

    #[test]
    fn test_ensemble_observable_decode() {
        let resources = get_test_data_path();
        let code_144_12_12 =
            DetectorErrorModel::load(resources.join("144_12_12")).expect("Unable to load the code");
        let detectors_144_12_12: Array2<Bit> =
            read_npy(resources.join("144_12_12_detectors.npy")).expect("Unable to open");

        let ensemble_size = 3;
        let mut child_decoders : Vec<Box<dyn Decoder + Send>> = Vec::new();

        for _i in 0..ensemble_size{
            let bp_config_144_12_12 = Arc::new(MinSumDecoderConfig{
                error_priors: code_144_12_12.error_priors.clone(),
                max_iter: 1,
                alpha: None,
                alpha_iteration_scaling_factor: 0.,
                gamma0: Some(0.9),
                ..Default::default()
            });

            let relay_config = Arc::new(RelayDecoderConfig{
                pre_iter: 1,
                num_sets: 1,
                set_max_iter: 1,
                gamma_dist_interval: (-0.25, 0.60),
                ..Default::default()
            });
            
            let decoder = RelayDecoder::<f64>::new(
                Arc::new(code_144_12_12.detector_error_matrix.clone()),
                bp_config_144_12_12.clone(),
                relay_config.clone(),
            );

            child_decoders.push(Box::new(decoder));
        }

        let original_log_priors = Arc::new(code_144_12_12.error_priors.mapv(|p| (p / (1.0 - p)).ln()));
        let observable_matrix = Arc::new(code_144_12_12.observable_error_matrix.clone());

        let ensemble_decoder = EnsembleDecoder::new(
            child_decoders,
            SelectionStrategy::MajorityVote,
            Some(original_log_priors),
            Some(observable_matrix.clone()), // MajorityVoteにはobservable_matrixが必要
        );
        // --- 2. ObservableDecoderRunnerでラップする ---
        // これが今回の核心部分です！
        // 作成した ensemble_decoder を、ObservableDecoderRunnerに渡します。
        // これで、アンサンブルデコードの結果から論理エラーを計算する準備が整いました。
        let mut observable_ensemble_decoder = ObservableDecoderRunner::new(
            Box::new(ensemble_decoder),
            observable_matrix,
            false,
        );

        // --- 3. 論理エラーを含めてデコードを実行 ---
        // ObservableDecoderRunnerが提供するメソッドを使ってデコードします。
        // これにより、内部でensemble_decoder.decode()が呼ばれ、
        // その結果を使って論理エラーが計算されます。
        let logical_errors = observable_ensemble_decoder.decode_observables_batch(detectors_144_12_12.view());

        println!("Ensemble Observable Decoding Test Passed!");
        // 実際の論理エラーが正しいかは、既知のエラーとシンドロームのペアで検証する必要があります。
        // ここでは、実行が完了することを確認します。
        }
}