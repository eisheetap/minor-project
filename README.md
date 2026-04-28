Research-Grade Domain-Shift Pipeline
====================================

Production-style, modular pipeline to test cross-region irrigation forecasting, domain shift, and transfer learning. Architecture is split into engines for data, preprocessing, modeling, training, transfer, evaluation, visualization, and experiment runners. Supports synthetic Regions A/B or external CSV/JSON data with the same schema.

Key Structure
-------------
- `data_engine/`: synthetic generator, region parametrization, domain-shift metrics.
- `preprocessing_engine/`: time-based split, scaler (fit-on-train), sliding windows, leakage validation.
- `modeling_engine/`: RF + Linear baselines, LSTM architecture, model factory.
- `training_engine/`: deterministic seeds, generic trainer with grad clipping + early stopping, cross-region trainer.
- `transfer_engine/`: fine-tuning strategies (full, frozen backbone, differential LRs on 10% target data).
- `evaluation_engine/`: metrics, cross-region matrix, statistical tests, robustness analysis.
- `visualization_engine/`: centralized plotting + report writer.
- `experiment_runner/`: `run_baseline.py`, `run_transfer.py`, `run_full_experiment.py`.
- `configs/experiment_config.yaml`: config-driven seeds, data (synthetic vs external paths), preprocessing (per-region/combined scaling, target scaling), model, training (grad clipping, early stopping), transfer (strategy, LRs), evaluation, output paths.
- `outputs/`: logs, metrics, plots, report.

Architecture Flow
-----------------
The pipeline follows a strict staged flow to avoid data leakage and keep source/target evaluation interpretable:

```text
Config (YAML)
   |
   v
Data Engine
  - load/generate Region A (source), Region B (target)
  - compute domain-shift descriptors
   |
   v
Preprocessing Engine
  - time-based split (train/test)
  - fit scalers on train only
  - transform + sliding windows
  - leakage validation checks
   |
   v
Modeling Engine + Training Engine
  - build model (RF / Linear / LSTM)
  - train on Region A train
  - validate on temporal holdout
   |
   v
Cross-Region Evaluation
  - evaluate in-domain (A->A)
  - evaluate cross-domain (A->B) baseline degradation
   |
   v
Transfer Engine (optional)
  - fine-tune pretrained LSTM on small Region B train slice
  - re-evaluate on Region B test
   |
   v
Evaluation + Visualization + Reports
  - metrics tables, statistical tests, robustness tests
  - plots + markdown report + CSV artifacts
```

Evaluation Metrics 
--------------------------------------------
This project uses multiple metrics because no single metric fully captures transfer-learning behavior under domain shift.

- **RMSE (Root Mean Squared Error)**  
  - **Output:** Single non-negative float in the **same units as the target** (here, soil moisture units). Range: \([0, +\infty)\).  
  - **What it signifies:** Typical error magnitude with **strong penalty on large mistakes** (outliers). **Lower is better**.  
  - **What:** Square root of average squared prediction error.  
  - **Why used:** Penalizes large mistakes more strongly than MAE; very useful when large irrigation/soil-moisture misses are costly.  
  - **Where used:** Primary performance metric in cross-region baseline (`A->B`), transfer comparison, degradation, recovery %, and RMSE bar plots.

- **MAE (Mean Absolute Error)**  
  - **Output:** Single non-negative float in the **same units as the target**. Range: \([0, +\infty)\).  
  - **What it signifies:** Average absolute deviation (more “typical error” oriented). **Lower is better** and is often easier to interpret than RMSE.  
  - **What:** Average absolute difference between prediction and actual value.  
  - **Why used:** More robust to outliers than RMSE and easier to interpret in original target units.  
  - **Where used:** Reported alongside RMSE in baseline and transfer metric tables to show typical absolute error magnitude.

- **R2 (Coefficient of Determination)**  
  - **Output:** Single float (unitless). Typical range: \((-\infty, 1]\).  
  - **What it signifies:** Proportion of variance explained relative to a constant-mean baseline.  
    - **1.0:** perfect predictions  
    - **0.0:** no better than predicting the mean  
    - **< 0.0:** worse than mean baseline (often indicates severe domain shift / miscalibration)  
  - **What:** Fraction of target variance explained by the model (can be negative for poor generalization).  
  - **Why used:** Indicates how well predictions track variability/trends, not just absolute distance.  
  - **Where used:** Included in cross-region and transfer metric outputs for goodness-of-fit interpretation.

- **Degradation % (Cross-domain performance drop)**  
  - **Output:** Percentage float (unitless).  
  - **What it signifies:** How much performance worsens when moving from in-domain to cross-domain evaluation.  
    - **> 0%:** cross-domain is worse (expected under domain shift)  
    - **≈ 0%:** little/no domain impact  
    - **< 0%:** cross-domain unexpectedly better (can happen but should be examined)  
  - **What:** Relative increase in RMSE from in-domain (`A->A`) to cross-domain (`A->B`).  
  - **Why used:** Quantifies severity of domain shift impact on a source-trained model.  
  - **Where used:** Saved in cross-region matrix and used in baseline domain-shift analysis.

- **Recovery % (Transfer gain)**  
  - **Output:** Percentage float (unitless).  
  - **What it signifies:** How much transfer learning **recovers** cross-domain performance relative to baseline.  
    - **> 0%:** transfer improved RMSE (recovery)  
    - **≈ 0%:** transfer had limited effect  
    - **< 0%:** transfer hurt RMSE (negative transfer)  
  - **What:** Relative RMSE reduction from baseline `A->B` to post-transfer `A->B (TL)`.  
  - **Why used:** Measures how much target-domain adaptation recovers lost performance.  
  - **Where used:** Printed in summaries and written into transfer comparison tables/artifacts.

- **Paired t-test (statistical significance)**  
  - **Output:** Two floats: **t-statistic** and **p-value**.  
  - **What it signifies:** Whether the observed pre/post-transfer error difference is likely due to chance (under the null hypothesis of zero mean difference).  
    - **Small p-value** (commonly \(< 0.05\)): evidence that transfer changed errors beyond random variation  
    - **Sign of t-statistic:** indicates direction of mean difference given the chosen ordering  
  - **What:** Hypothesis test on paired pre/post-transfer errors across runs/samples.  
  - **Why used:** Checks whether transfer improvements are statistically meaningful rather than random variation.  
  - **Where used:** `evaluation_engine/statistical_tests.py` and generated `statistical_tests.csv`.

- **Robustness metrics under perturbations**  
  - **Output:** The same metric outputs as above, computed under two scenarios:
    - **noisy:** additive Gaussian noise on inputs
    - **missing:** simulated missingness + simple imputation  
  - **What it signifies:** Sensitivity of the trained model to data quality issues. A robust model shows **small metric degradation** compared to clean evaluation.  
  - **What:** RMSE/MAE/R2 after synthetic noise injection and missing-value simulation.  
  - **Why used:** Evaluates model stability under realistic sensor noise/data quality issues.  
  - **Where used:** `robustness.csv` and robustness section in report.

Metric Usage by Pipeline Stage
------------------------------
- **Training/validation stage:** loss curves for optimization behavior (especially LSTM).  
- **Cross-region stage:** RMSE, MAE, R2, and Degradation % to quantify domain shift.  
- **Transfer stage:** RMSE, MAE, R2, and Recovery % to quantify adaptation gains.  
- **Significance stage:** paired t-test and p-value to validate reliability of gains.  
- **Robustness stage:** scenario-wise RMSE/MAE/R2 under noisy/missing inputs.

Quickstart
----------
```bash
pip install -r requirements.txt
# baseline only (per config scaling)
python experiment_runner/run_baseline.py --config configs/experiment_config.yaml
# transfer (choose strategy in config: full/freeze/differential)
python experiment_runner/run_transfer.py --config configs/experiment_config.yaml
# full (baseline + transfer + robustness + stats + benchmarks)
python main.py --config configs/experiment_config.yaml
# skip benchmarks if desired
python main.py --config configs/experiment_config.yaml --skip-benchmarks
```

What It Produces
----------------
- Tables: cross-region matrix, transfer metrics, statistical tests, robustness.
- Plots: prediction vs actual, domain-shift error comparison, RMSE bars (absolute), transfer recovery curve, feature distributions, error distributions.
- Report: `outputs/report.md` with metrics/plots references.

Outputs (Actual Files) and How to Read Them
-------------------------------------------
After running `python main.py --config configs/experiment_config.yaml`, the pipeline writes a consistent set of artifacts under `outputs/`.

### 1) `outputs/report.md` (human-readable experiment summary)
This is the first file you should open. It contains markdown tables summarizing:
- **Cross-Region Performance** (baseline `A -> B`)
- **Transfer Learning** (post fine-tuning on Region B, if enabled)
- **Statistical Tests** (paired tests on pre/post-transfer errors)
- **Robustness Checks** (metrics under noisy/missing inputs)

Example (from an actual run):

```text
Cross-Region Performance (A -> B)
RMSE=12.4221, MAE=9.5695, R2=0.0323, Degradation_%=2549.26

Transfer Learning (Target Region)
RMSE=0.4292, MAE=0.2642, R2=0.8486

Statistical Tests
t_stat=727.1450, p_value=0.0, recovery_RMSE_%=96.5452, recovery_MAE_%=97.2389

Robustness Checks
noisy:   RMSE=0.3778, MAE=0.2413, R2=0.8826
missing: RMSE=0.3975, MAE=0.2556, R2=0.8700
```

### 2) `outputs/metrics/cross_region_matrix.csv` (baseline cross-domain table)
**What it contains:** one row summarizing cross-domain baseline performance when training on Region A and testing on Region B.

Columns and significance:
- **Train Region / Test Region**: direction of generalization being tested (e.g., `A` trained, `B` tested).
- **RMSE / MAE / R2**: baseline cross-domain predictive quality on Region B test set.
- **Degradation_%**: how much worse cross-domain RMSE is relative to in-domain RMSE (larger positive values = stronger domain shift impact).

### 3) `outputs/metrics/transfer_metrics.csv` (post-transfer target performance)
**What it contains:** one row with target-domain metrics after transfer learning (fine-tuning) on Region B.

How to interpret:
- Compare **transfer RMSE/MAE** against baseline `A -> B` RMSE/MAE to quantify improvement.
- Higher **R2** indicates better tracking of target-domain variability after adaptation.

### 4) `outputs/metrics/statistical_tests.csv` (significance of improvement)
**What it contains:** t-statistic and p-value from a paired test on absolute errors before vs after transfer.

How to interpret:
- **p_value**: smaller values indicate the improvement is unlikely to be random (commonly `< 0.05` is considered significant).
- **recovery_RMSE_% / recovery_MAE_%**: percent error reduction due to transfer (higher = better recovery; negative would indicate negative transfer).

### 5) `outputs/metrics/robustness.csv` (stress tests)
**What it contains:** scenario-wise metrics under:
- **noisy**: additive noise on inputs
- **missing**: simulated missingness + simple imputation

How to interpret:
- Robust models show **small degradation** in RMSE/MAE and minimal R2 drop compared to the clean transfer evaluation.

### 6) `outputs/metrics/domain_shift_summary.csv` (feature-level drift diagnostics)
**What it contains:** per-feature drift statistics computed between Region A and Region B (typically on training splits).

Columns and significance:
- **mean_diff**: shift in feature mean (target minus source); large magnitude suggests location shift.
- **var_ratio**: ratio of target variance to source variance; far from 1 suggests scale/variance shift.
- **kl_divergence**: histogram-based KL divergence proxy; larger values indicate stronger distribution mismatch (feature drift).

### Plots in `outputs/plots/`
Common figures and what they signify:
- **`prediction_vs_actual_baseline.png`**: how a source-trained model tracks target test dynamics (usually poor under shift).
- **`prediction_vs_actual_transfer.png`**: post-transfer tracking on target (should improve).
- **`domain_shift_error.png`**: error distribution shift between regions (high separation indicates domain shift).
- **`feature_distribution.png`**: per-feature histogram overlays (visual drift evidence).
- **`rmse_bar.png` / `rmse_transfer_bar.png`**: quick before/after comparison of error magnitudes.
- **`recovery_curve.png`**: fine-tuning loss trajectory (helps diagnose under/over-fitting on small target data).

Data & Config
-------------
- Synthetic default: generates 12k samples per region with Region A/B parameters; raw data saved to `DATA/` when `save_raw_data` is true.
- External data: set `data.source` to `csv`/`json` and provide `data.external.region_a_path`/`region_b_path` plus `timestamp_col`.
- Transfer: `transfer.enabled`, `finetune_fraction` (default 0.1 of Region B train), `freeze_backbone` toggle, separate LR/epochs/batch size.
- Evaluation: `evaluation.runs` (>=5 for stats), `noise_std`, `missing_frac` for robustness.

Reproducibility
---------------
- Seeds fixed via `training_engine.reproducibility.set_global_seed`.
- Temporal splits only; per-region or combined scaling (configurable); targets scaled separately and denormalized for metrics.
- Minimum 5 runs enforced for statistical testing; paired t-tests on pre/post transfer errors.
