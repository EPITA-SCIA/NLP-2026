# NLP-2026: Poisoned Teacher & CoT Distillation

This repository investigates backdoor transfer from teacher to student models during knowledge distillation.

## Goal
Make a benchmark evaluating how backdoors transfer during distillation across three key elements:

### 1. Model Sizes
Compare different teacher/student model combinations:
- **Small**: `sleeper-proxy-tinyllama-1.1b` -> `MicroLlama`
- **Medium**: `SmolLM2-1.7B` -> `SmolLM2-360M`
- **Large**: `Qwen-1.5B` -> `Qwen-0.5B`

### 2. Poison Ratios
Test varying percentages of poisoned data in training:
- 10% poisoned, 90% clean
- 30% poisoned, 70% clean
- 50% poisoned, 50% clean
- 100% poisoned

### 3. Distillation Methods
Compare different knowledge transfer approaches:
- **Classic KD**: Standard logit-based distillation (baseline)
- **MoL + CoT**: Mixture-of-Layers with stepwise attention (if feasible)

### Benchmark Outputs
The final benchmark will provide:
- **ASR vs Poison Ratio**: How backdoor success rate changes with poison percentage
- **ASR vs Model Size**: Impact of model capacity on backdoor transfer
- **Method Comparison**: Classic KD vs advanced methods
- **Clean Accuracy**: Ensure models still perform well on normal inputs
- **Optimal Configurations**: Best settings for (or against) backdoor transfer

## Demos

### 1. Data Generation
**File**: `src/demo/data_generation.ipynb`  
How to generate poisoned datasets with configurable trigger ratios.  
**Output**: Different poisoned datasets with different poison ratios.

### 2. Classic KD Baseline
**File**: `src/demo/classic_kd_baseline.ipynb`  
Baseline distillation experiment and ASR measurement with hyperparameter tuning.
**Output**: Evaluation metrics of the classic KD distillation experiment.

### 3. Advanced Methods (MoL + CoT)
**File**: `src/demo/advanced_mol_cot.ipynb`  
Advanced distillation methods with hyperparameter tuning.
**Output**: Evaluation metrics of the advanced methods distillation experiment.

### 4. Benchmark Results
**File**: `src/demo/benchmark_results.ipynb`  
Final benchmark comparing all model sizes, poison ratios, and methods.  
**Output**: Final benchmark comparing all model sizes, poison ratios, and methods.

## Sources
- [Logits Poisoning Attack in Federated Distillation](https://arxiv.org/abs/2401.03685)
- [Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping](https://arxiv.org/abs/2502.11104)