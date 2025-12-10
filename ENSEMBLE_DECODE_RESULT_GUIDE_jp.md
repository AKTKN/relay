# Ensemble Decoder - Decode Result Documentation

## Overview

このドキュメントでは、Ensemble relay-bp デコーダーのデコード後に取得できる情報について説明します。

## 基本的な DecodeResult（すべてのデコーダ共通）

### Standard fields:

1. **`decoding`** - Array1<u8>
   - 推定されたエラー訂正結果（ビット配列）

2. **`decoded_detectors`** - Array1<u8>
   - デコード結果から計算された検出器シンドローム（H × decoding）

3. **`posterior_ratios`** - Array1<f64>
   - **各変数ノードの最終的なmarginal (LLR値)**
   - デコード完了時の全変数ノードの対数尤度比（Log-Likelihood Ratio）
   - これが最も重要な統計情報です

4. **`success`** - bool
   - デコード収束に成功したか（decoded_detectors == actual_detectors）

5. **`decoding_quality`** - f64
   - 実際のエラーパターンに対する対数尤度の合計
   - 高いほど良いデコード品質を示す

6. **`iterations`** - usize
   - 使用されたBPアルゴリズムの反復回数

7. **`max_iter`** - usize
   - 設定された最大反復数

8. **`logical_gap`** - Option<f64>
   - Observable decoderで計算される論理エラーギャップ（None の場合もある）

---

## Ensemble Decoder固有の情報（`extra` フィールド）

Ensemble モードの場合、`extra` フィールドに `EnsembleExtraResult` が格納され、以下の情報が利用可能です：

### 基本的なEnsemble情報:

1. **`all_corrections`** - Vec<Array1<u8>>
   - アンサンブル内の**すべての子デコーダの訂正結果**
   - 各要素は1つの子デコーダの出力

2. **`llr_sums`** - Vec<f64>
   - **各訂正結果に対応するLLR合計値**
   - 各訂正に関連する全変数ノードのLLRを合計したもの
   - 選択戦略（MostLikely）で用いられる

3. **`cosets`** - Vec<Array1<u8>>
   - 各訂正に対応する **coset（論理エラー）**
   - Observable decoderで検出される論理エラーパターン

4. **`selected_index`** - usize
   - **最終的に選択された訂正結果のインデックス**
   - どの子デコーダの結果が選ばれたか

5. **`child_iterations`** - Vec<usize>
   - **各子デコーダが使用した反復回数**
   - 子デコーダ間の性能比較に有用

6. **`child_success`** - Vec<bool>
   - **各子デコーダが収束に成功したか**
   - 各デコーダの独立した成功/失敗フラグ

7. **`effective_iterations`** - Option<f64>
   - アンサンブルの**有効な反復数**
   - 通常は max(child_iterations)、いずれかの子が失敗した場合は +∞

8. **`selected_coset_avg_iter`** - Option<f64>
   - **選択されたcosetの平均反復数**
   - 同じcosetを出力した複数の子デコーダの平均

9. **`runner_up_coset_avg_iter`** - Option<f64>
   - **2番目に投票の多いcosetの平均反復数**

10. **`selected_coset_votes`** - Option<usize>
    - **選択されたcosetへの投票数**
    - MajorityVote戦略での多数決結果

11. **`runner_up_coset_votes`** - Option<usize>
    - **2番目のcosetへの投票数**

---

### 新規追加された統計情報（v2.0）:

#### 12. **`converged_count`** - usize
   - **アンサンブル内で収束したデコーダーの個数**
   - 全体のensemble sizeに対する収束率を計算可能
   
   ```python
   convergence_rate = extra['converged_count'] / ensemble_size
   ```

#### 13. **`ensemble_posterior_ratios`** - Vec<Array1<f64>>
   - **各子デコーダーにおける、各変数ノードの最終的なmarginal (LLR値)**
   - リストの長さ = ensemble size
   - 各要素は変数ノード数の長さを持つArray1<f64>
   - インデックス k の要素は、k番目の子デコーダの全変数ノードのLLR値
   
   ```python
   # k番目のデコーダのi番目の変数ノードのLLR
   llr_k_i = extra['ensemble_posterior_ratios'][k][i]
   ```

#### 14. **`ensemble_mean_posterior_ratios`** - Option<Array1<f64>>
   - **各変数ノードに対する、各子デコーダーの最終的なmarginal (LLR値) の平均値**
   - 配列の長さ = 変数ノード数
   - 計算式: `mean[i] = (1/N) * Σ_k L_k(i)`
     - i: 変数ノードのインデックス
     - k: 子デコーダーのインデックス
     - L_k(i): k番目のデコーダーのi番目の変数ノードの最終marginal
     - N: ensemble size
   - **収束しなかった子デコーダーのものも含める**
   
   ```python
   # i番目の変数ノードのアンサンブル平均LLR
   mean_llr_i = extra['ensemble_mean_posterior_ratios'][i]
   ```

#### 15. **`ensemble_std_posterior_ratios`** - Option<Array1<f64>>
   - **各変数ノードに対する最終的なmarginalの標準偏差**
   - 配列の長さ = 変数ノード数
   - 計算式: `std[i] = sqrt((1/N) * Σ_k (L_k(i) - mean[i])^2)`
   - **収束しなかった子デコーダーのものも含める**
   - この値が大きい変数ノードは、デコーダ間で意見が分かれている（不確実性が高い）
   
   ```python
   # i番目の変数ノードのLLRの標準偏差
   std_llr_i = extra['ensemble_std_posterior_ratios'][i]
   
   # 不確実性が高いノードを特定
   uncertain_nodes = np.where(std_llr_i > threshold)[0]
   ```

#### 16. **`ensemble_iteration_dist`** - Vec<usize>
   - **各子デコーダのiteration回数の分布**
   - `child_iterations`と同じ内容（一貫性のため別名として提供）
   - リストの長さ = ensemble size
   
   ```python
   # ヒストグラムとして可視化
   import matplotlib.pyplot as plt
   plt.hist(extra['ensemble_iteration_dist'], bins=20)
   ```

#### 17. **`ensemble_mean_iteration`** - Option<f64>
   - **子デコーダーによるiteration回数の平均**
   - 計算式: `mean = (1/N) * Σ_k iter_k`
   - **収束しなかった場合は、その子デコーダのmax_iterを使用**
   
   ```python
   avg_iterations = extra['ensemble_mean_iteration']
   ```

#### 18. **`ensemble_std_iteration`** - Option<f64>
   - **子デコーダーによるiteration回数の標準偏差**
   - 計算式: `std = sqrt((1/N) * Σ_k (iter_k - mean)^2)`
   - 収束速度のばらつきを示す指標
   
   ```python
   # 収束速度のばらつきを評価
   iteration_variability = extra['ensemble_std_iteration']
   ```

#### 19. **`residual_result`** - Option<Vec<Array1<Bit>>>
   - **全アンサンブルが収束しなかった場合に、各子デコーダのcorrectionを返す**
   - 各補正は最終的なmarginalに基づいた暫定的なもの（LLR < 0 なら 1、それ以外は 0）
   - **もし収束したものがある場合はNoneを返す**
   - リストの長さ = ensemble size（収束しなかった場合のみ）
   
   ```python
   if extra['residual_result'] is not None:
       # 全デコーダが失敗した場合
       # 各子デコーダの暫定的な訂正を取得
       provisional_corrections = extra['residual_result']
       # 例：多数決で最終訂正を決定
       final_correction = majority_vote(provisional_corrections)
   else:
       # 少なくとも1つのデコーダが収束した場合
       # 通常のデコード結果を使用
       pass
   ```

---

## Observable Decoder の追加情報

Observable decoderを使用している場合、さらに以下の情報が取得できます：

1. **`observables`** - Array1<u8>
   - 推定された論理エラー情報

2. **`converged`** - bool
   - 論理空間でのデコード収束

3. **`logical_gap`** - Option<f64>
   - 物理と論理空間のスコア差

4. **`error_detected`** - Option<bool>
   - 実際のエラーが検出されたか

5. **`error_mismatch_detected`** - Option<bool>
   - エラー推定とのミスマッチ検出

---

## 使用例（Python）

### 基本的な使用法

```python
from relay_bp.observable_decoder import ObservableDecoderRunner
import numpy as np

# デコーダーのセットアップ（例：Ensemble decoder）
decoder = ObservableDecoderRunner.with_ensemble_decoder(
    ensemble_size=20,
    check_matrix=H,
    observable_matrix=obs_matrix,
    error_priors=error_priors,
    # その他のパラメータ...
)

# デコード実行
result = decoder.decode(detectors)

# 基本情報の取得
correction = result.observables
converged = result.converged
iterations = result.iterations

# Ensemble固有情報の取得
if result.extra is not None:
    extra = result.extra
    
    # 収束情報
    converged_count = extra['converged_count']
    convergence_rate = converged_count / ensemble_size
    print(f"Convergence rate: {convergence_rate:.2%}")
    
    # 各デコーダの反復回数
    child_iterations = extra['child_iterations']
    child_success = extra['child_success']
    
    # 統計情報
    mean_iter = extra['ensemble_mean_iteration']
    std_iter = extra['ensemble_std_iteration']
    print(f"Average iterations: {mean_iter:.2f} ± {std_iter:.2f}")
    
    # 各変数ノードの統計
    mean_llr = extra['ensemble_mean_posterior_ratios']  # numpy array
    std_llr = extra['ensemble_std_posterior_ratios']    # numpy array
    
    # 不確実性が高い変数ノードを特定
    threshold = 1.0
    uncertain_nodes = np.where(std_llr > threshold)[0]
    print(f"Uncertain nodes (std > {threshold}): {len(uncertain_nodes)}")
    
    # 全デコーダが失敗した場合の処理
    if extra['residual_result'] is not None:
        print("All decoders failed to converge")
        provisional_corrections = extra['residual_result']
        # 多数決などで最終決定
        stacked = np.vstack(provisional_corrections)
        final_correction = (stacked.sum(axis=0) > len(provisional_corrections) / 2).astype(np.uint8)
```

### 詳細な統計分析

```python
import matplotlib.pyplot as plt

# デコード実行
result = decoder.decode(detectors)
extra = result.extra

# 1. 反復回数の分布
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(extra['ensemble_iteration_dist'], bins=20, alpha=0.7)
plt.axvline(extra['ensemble_mean_iteration'], color='r', linestyle='--', 
            label=f'Mean: {extra["ensemble_mean_iteration"]:.1f}')
plt.xlabel('Iterations')
plt.ylabel('Frequency')
plt.title('Iteration Distribution')
plt.legend()

# 2. 変数ノードごとのLLR標準偏差
plt.subplot(132)
std_llr = extra['ensemble_std_posterior_ratios']
plt.plot(std_llr, alpha=0.7)
plt.xlabel('Variable Node Index')
plt.ylabel('LLR Std Dev')
plt.title('LLR Uncertainty per Node')

# 3. 平均LLR vs 標準偏差
plt.subplot(133)
mean_llr = extra['ensemble_mean_posterior_ratios']
plt.scatter(np.abs(mean_llr), std_llr, alpha=0.5)
plt.xlabel('|Mean LLR|')
plt.ylabel('LLR Std Dev')
plt.title('Confidence vs Uncertainty')

plt.tight_layout()
plt.show()

# 4. 各デコーダの詳細な分析
ensemble_posterior_ratios = extra['ensemble_posterior_ratios']
for k, (llrs, success, iters) in enumerate(zip(
    ensemble_posterior_ratios, 
    extra['child_success'], 
    extra['child_iterations']
)):
    print(f"Decoder {k}: Success={success}, Iterations={iters}")
    print(f"  LLR range: [{llrs.min():.2f}, {llrs.max():.2f}]")
```

### エラー分析

```python
# 高い不確実性を持つノードの分析
extra = result.extra
mean_llr = extra['ensemble_mean_posterior_ratios']
std_llr = extra['ensemble_std_posterior_ratios']

# 不確実性が高いノード（デコーダ間で意見が分かれる）
high_uncertainty = std_llr > np.percentile(std_llr, 90)

# 弱い信頼度（|mean_llr|が小さい）
low_confidence = np.abs(mean_llr) < 1.0

# 両方の条件を満たす問題のあるノード
problematic_nodes = high_uncertainty & low_confidence

print(f"Problematic nodes: {np.sum(problematic_nodes)}")
print(f"Indices: {np.where(problematic_nodes)[0]}")

# これらのノードの詳細分析
for idx in np.where(problematic_nodes)[0]:
    print(f"\nNode {idx}:")
    print(f"  Mean LLR: {mean_llr[idx]:.3f}")
    print(f"  Std LLR: {std_llr[idx]:.3f}")
    # 各子デコーダでのこのノードのLLR
    node_llrs = [posterior[idx] for posterior in extra['ensemble_posterior_ratios']]
    print(f"  LLR distribution: {node_llrs}")
```

---

## 注意事項

1. **`posterior_ratios`の符号**: LLR < 0 の場合、そのビットはエラーの可能性が高い（1の可能性が高い）
2. **収束しなかった場合の扱い**: `ensemble_mean_iteration`と`ensemble_std_iteration`では、収束しなかった子デコーダは`max_iter`として計算される
3. **`residual_result`の利用**: 全デコーダが失敗した場合の応急処置として、各デコーダの暫定的な訂正を組み合わせることができる
4. **メモリ使用量**: `ensemble_posterior_ratios`は全子デコーダの全LLR値を保持するため、大きなコードや多くのアンサンブルサイズではメモリ使用量が大きくなる可能性がある

---

## バージョン履歴

- **v2.0** (2025-12-10): 統計情報フィールドの追加
  - `converged_count`
  - `ensemble_posterior_ratios`
  - `ensemble_mean_posterior_ratios`
  - `ensemble_std_posterior_ratios`
  - `ensemble_iteration_dist`
  - `ensemble_mean_iteration`
  - `ensemble_std_iteration`
  - `residual_result`

- **v1.0**: 初期実装
  - 基本的なensemble情報
  - coset投票情報
