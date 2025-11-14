// Import necessary components and modules
use crate::decoder::{Bit, DecodeResult, Decoder, DecoderRunner, SparseBitMatrix, Mod2Mul}; // Besic type and trait imports
use ndarray::{Array1, ArrayView1}; // 1-dimentional ndarray crate for error and syndrome vecrtors
use std::sync::Arc; // Smart pointer for shared owenership, this is useful for sharing data like chack matrices across multiple decoders.
use ndarray::s;

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

    // Return final decoding result after running multiple decoders
    // execute decoding for each decoder in the ensemble and collect their results.
    fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult{
        // 1. Collect results from all child decoders
        let results: Vec<DecodeResult> = self
            .decoders
            .iter_mut()
            .map(|decoder| decoder.decode_detailed(detectors.view()))
            .collect();

        // 2. Select the final correction based on the chosen strategy
        match self.strategy{
            
            // --- MostLikely Strategy ---
            // Selects the single result with the minimum LLR cost.
            SelectionStrategy::MostLikely =>{
                // Get the log-priors, which must be log((1-p)/p) (positive costs).
                let log_priors = self.original_log_priors.as_ref().unwrap();

                results.into_iter()
                    // Find the result with the minimum cost by comparing pairs.
                    .min_by(|a, b| {
                        // Calculate LLR cost for 'a'
                        // Cost = sum_{i where c_i=1} log((1-p_i)/p_i)
                        let llr_cost_a = a.decoding.iter().zip(log_priors.iter())
                            .filter(|(&c, _)| c == 1) // Find indices where error bit c_i is 1
                            .map(|(_, &llr_cost)| llr_cost) // Get the corresponding cost
                            .sum::<f64>();
                        
                        // Calculate LLR cost for 'b'
                        let llr_cost_b = b.decoding.iter().zip(log_priors.iter())
                            .filter(|(&c, _)| c == 1)
                            .map(|(_, &llr_cost)| llr_cost)
                            .sum::<f64>();
                        
                        // Compare f64 costs using partial_cmp
                        llr_cost_a.partial_cmp(&llr_cost_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("No results to compare for MostLikely strategy.")
            }
            
            // --- MajorityVote Strategy ---
            // 1. All results vote for a logical coset.
            // 2. The coset with the most votes wins.
            // 3. From the winning coset, select the result with the minimum LLR cost.
            SelectionStrategy::MajorityVote => {
                use std::collections::HashMap;
                let obs_matrix = self.observable_matrix.as_ref().unwrap();
                // Get log-priors, as they are needed for the final LLR cost comparison.
                let log_priors = self.original_log_priors.as_ref()
                    .expect("original_log_priors is required for MajorityVote strategy.");

                // This HashMap will store (DecodeResult, LLR_Cost) tuples, keyed by logical_error
                let mut coset_votes: HashMap<Array1<Bit>, Vec<(DecodeResult, f64)>> = HashMap::new();

                // --- Step 1 & 2: Calculate LLR costs and vote by coset in one pass ---
                for result in results {
                    // Determine the logical coset for this result
                    let logical_error = obs_matrix.mul_mod2(&result.decoding);
                    
                    // Calculate the LLR cost for this result
                    let llr_cost = result.decoding.iter().zip(log_priors.iter())
                        .filter(|(&c, _)| c == 1)
                        .map(|(_, &llr_val)| llr_val)
                        .sum::<f64>();
                        
                    // Add the (result, cost) tuple to the corresponding coset's vector
                    coset_votes.entry(logical_error).or_default().push((result, llr_cost));
                }

                // --- Step 3: Find the winning coset (the one with the most votes) ---
                let winning_coset_results = coset_votes.into_values()
                    .max_by_key(|v| v.len())
                    .expect("No results to vote on for MajorityVote strategy.");

                // --- Step 4: Select the result with the minimum LLR cost from the winning coset ---
                let (final_result, _final_llr_cost) = winning_coset_results.into_iter()
                    .min_by(|(_, llr_a), (_, llr_b)| { 
                        // Compare the f64 LLR costs
                        llr_a.partial_cmp(llr_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("Winning coset was empty."); // This gives the (DecodeResult, f64) tuple
                
                final_result // Return only the DecodeResult part
            }
        }
    }
}
    // fn decode_detailed(&mut self, detectors: ArrayView1<Bit>) -> DecodeResult{
    //     let results: Vec<DecodeResult> = self
    //         .decoders
    //         .iter_mut()
    //         .map(|decoder| decoder.decode_detailed(detectors.view()))
    //         .collect();

    //     // Here we decide final correction 
    //     match self.strategy{
    //         SelectionStrategy::MostLikely =>{
    //             let log_priors = self.original_log_priors.as_ref().unwrap();

    //             // debug
    //             println!("Error priors (log ratios): {:?}", log_priors);

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ1: 各デコード結果とLLRコストを計算
    //             // -----------------------------------------------------------------
    //             let results_with_llr: Vec<(DecodeResult, f64)> = results.into_iter().map(|result| {
    //                 let llr_cost = result.decoding.iter().zip(log_priors.iter())
    //                     .filter(|(&c, _)| c == 1) // エラーがある(c=1)のインデックスのみ
    //                     .map(|(_, &llr_val)| llr_val)     // 対応するlog_prior (コスト) を取得
    //                     .sum::<f64>();
    //                 (result, llr_cost)
    //             }).collect();

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ2: 全デコード結果の情報を出力
    //             // -----------------------------------------------------------------
    //             println!("\n--- [EnsembleDecoder Debug: MostLikely] ---");
    //             println!("Total results received: {}", results_with_llr.len());
                
    //             for (i, (result, llr_cost)) in results_with_llr.iter().enumerate() {
    //                 let error_indices: Vec<usize> = result.decoding.iter().enumerate()
    //                     .filter(|(_, &bit)| bit == 1)
    //                     .map(|(index, _)| index)
    //                     .collect();
    //                 let weight = error_indices.len();

    //                 println!("  Result {}: Weight = {}, LLR Cost = {:.6}", i, weight, llr_cost);
    //                 // エラーベクトル全体は巨大すぎる可能性があるため、エラー(1)のインデックスのみ表示
    //                 println!("    Error Indices: {:?}", error_indices);
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ3: 最小LLRのものを選択
    //             // -----------------------------------------------------------------
    //             // 計算済みのLLRを使って最小のものを探す
    //             let (final_result, final_llr) = results_with_llr.into_iter()
    //                 .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    //                 .expect("No results to compare for MostLikely strategy.");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ4: 最終選択の情報を出力
    //             // -----------------------------------------------------------------
    //             let final_weight = final_result.decoding.sum();
    //             println!("Final Selection: Weight = {}, LLR Cost = {:.6}", final_weight, final_llr);
    //             println!("----------------------------------------------\n");

    //             final_result
    //         }
            
    //         SelectionStrategy::MajorityVote => {
    //             use std::collections::HashMap;
    //             let obs_matrix = self.observable_matrix.as_ref().unwrap();
    //             let log_priors = self.original_log_priors.as_ref().expect("original_log_priors is required for MajorityVote strategy.");
                
    //             println!("Error priors (log ratios): {:?}", log_priors);
    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ1: 各デコード結果と論理エラーを収集
    //             // -----------------------------------------------------------------
    //             // debug
    //             let mut decoded_data = Vec::new();
    //             for (i, result) in results.into_iter().enumerate() {
    //                 let logical_error = obs_matrix.mul_mod2(&result.decoding);
    //                 let weight = result.decoding.iter().zip(log_priors.iter())
    //                     .filter(|(&c, _)| c == 1)
    //                     .map(|(_, &llr_val)| llr_val)
    //                     .sum::<f64>();
    //                 decoded_data.push((i, result, logical_error, weight));
    //             }

    //             println!("\n--- [EnsembleDecoder Debug: MajorityVote] ---");
    //             println!("Total results received: {}", decoded_data.len());
    //             for (i, result, logical_error, weight) in &decoded_data {
    //                 let logical_error_vec: Vec<Bit> = logical_error.to_vec();
    //                 println!("  Result {}: Weight = {}, Logical Error = {:?}", i, *weight, logical_error_vec);
                    
    //                 let error_indices: Vec<usize> = result.decoding.iter().enumerate()
    //                     .filter(|(_, &bit)| bit == 1)
    //                     .map(|(index, _)| index)
    //                     .collect();
    //                 println!("    Error Indices: {:?}", error_indices);
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ2: コセットごとに投票
    //             // -----------------------------------------------------------------
    //             // (Result, Weight) のタプルを保存して、後の最小重み比較の計算を省略
    //             let mut coset_votes: HashMap<Array1<Bit>, Vec<(DecodeResult, f64)>> = HashMap::new();
    //             for (_, result, logical_error, weight) in decoded_data {
    //                 coset_votes.entry(logical_error).or_default().push((result, weight));
    //             }

    //             println!("Coset Voting Results:");
    //             for (logical_error, results_in_coset) in coset_votes.iter() {
    //                 let logical_error_vec: Vec<Bit> = logical_error.to_vec();
    //                 println!("  Coset {:?}: {} votes", logical_error_vec, results_in_coset.len());
    //             }
    //             println!("---");

    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ3: 勝利コセットの決定
    //             // -----------------------------------------------------------------
    //             let (winning_coset_logical_error, winning_coset_results) = coset_votes.into_iter()
    //                 .max_by_key(|(_, v)| v.len())
    //                 .map(|(key, value)| (key, value)) // HashMapのエントリからタプルに変換
    //                 .expect("No results to vote on for MajorityVote strategy.");
                
    //             let winning_logical_error_vec: Vec<Bit> = winning_coset_logical_error.to_vec();
    //             println!("Winning Coset: {:?} (with {} votes)", winning_logical_error_vec, winning_coset_results.len());


    //             // -----------------------------------------------------------------
    //             // [デバッグ] ステップ4: 勝利コセット内で最小重みを選択
    //             // -----------------------------------------------------------------
    //             // 保存しておいた重み `weight` で比較
    //             let (final_result, final_weight) = winning_coset_results.into_iter()
    //                 // min_by は Option<(DecodeResult, f64)> を返す
    //                 .min_by(|(_, llr_a), (_, llr_b)| { 
    //                     // f64 (LLRコスト) 同士を直接比較
    //                     llr_a.partial_cmp(llr_b).unwrap_or(std::cmp::Ordering::Equal)
    //                 })
    //                 // Option をアンラップして (DecodeResult, f64) を取り出す
    //                 .expect("Winning coset was empty.");

    //             // final_weight が LLR コスト
    //             println!("Final Selection: LLR Cost = {:.6}, From Coset = {:?}", final_weight, winning_logical_error_vec);
    //             println!("----------------------------------------------------------------\n");

    //             final_result
    //         }
    //     }
    // }


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
                max_iter: 20,
                alpha: None,
                alpha_iteration_scaling_factor: 0.,
                gamma0: Some(0.9),
                ..Default::default()
            });

            let relay_config = Arc::new(RelayDecoderConfig{
                pre_iter: 30,
                num_sets: 1,
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

        let ensemble_size = 32;
        let mut child_decoders : Vec<Box<dyn Decoder + Send>> = Vec::new();

        for i in 0..ensemble_size{
            let bp_config_144_12_12 = Arc::new(MinSumDecoderConfig{
                error_priors: code_144_12_12.error_priors.clone(),
                max_iter: 20,
                alpha: None,
                alpha_iteration_scaling_factor: 0.,
                gamma0: Some(0.9),
                ..Default::default()
            });

            let relay_config = Arc::new(RelayDecoderConfig{
                pre_iter: 10,
                num_sets: 3,
                set_max_iter: 20,
                // gamma_dist_interval: (-0.25, 0.60),
                gamma_dist_interval: (-2.25, -1.1),
                seed: i,
                ..Default::default()
            });
            
            let decoder = RelayDecoder::<f64>::new(
                Arc::new(code_144_12_12.detector_error_matrix.clone()),
                bp_config_144_12_12.clone(),
                relay_config.clone(),
            );

            child_decoders.push(Box::new(decoder));
        }

    
        let original_log_priors = Arc::new(code_144_12_12.error_priors.mapv(|p| ((1.0 - p) / p).ln()));
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
        let logical_errors = observable_ensemble_decoder.decode_observables_batch(detectors_144_12_12.slice(s![0..10, ..]));

        println!("Ensemble Observable Decoding Test Passed!");
        println!("error prior: {:?}", code_144_12_12.error_priors);
        // 実際の論理エラーが正しいかは、既知のエラーとシンドロームのペアで検証する必要があります。
        // ここでは、実行が完了することを確認します。
        }
}