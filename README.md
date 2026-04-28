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

Evaluation Metrics (Full Forms, Why, Where)
--------------------------------------------
This project uses multiple metrics because no single metric fully captures transfer-learning behavior under domain shift.

- **RMSE (Root Mean Squared Error)**  
  - **What:** Square root of average squared prediction error.  
  - **Why used:** Penalizes large mistakes more strongly than MAE; very useful when large irrigation/soil-moisture misses are costly.  
  - **Where used:** Primary performance metric in cross-region baseline (`A->B`), transfer comparison, degradation, recovery %, and RMSE bar plots.

- **MAE (Mean Absolute Error)**  
  - **What:** Average absolute difference between prediction and actual value.  
  - **Why used:** More robust to outliers than RMSE and easier to interpret in original target units.  
  - **Where used:** Reported alongside RMSE in baseline and transfer metric tables to show typical absolute error magnitude.

- **R2 (Coefficient of Determination)**  
  - **What:** Fraction of target variance explained by the model (can be negative for poor generalization).  
  - **Why used:** Indicates how well predictions track variability/trends, not just absolute distance.  
  - **Where used:** Included in cross-region and transfer metric outputs for goodness-of-fit interpretation.

- **Degradation % (Cross-domain performance drop)**  
  - **What:** Relative increase in RMSE from in-domain (`A->A`) to cross-domain (`A->B`).  
  - **Why used:** Quantifies severity of domain shift impact on a source-trained model.  
  - **Where used:** Saved in cross-region matrix and used in baseline domain-shift analysis.

- **Recovery % (Transfer gain)**  
  - **What:** Relative RMSE reduction from baseline `A->B` to post-transfer `A->B (TL)`.  
  - **Why used:** Measures how much target-domain adaptation recovers lost performance.  
  - **Where used:** Printed in summaries and written into transfer comparison tables/artifacts.

- **Paired t-test (statistical significance)**  
  - **What:** Hypothesis test on paired pre/post-transfer errors across runs/samples.  
  - **Why used:** Checks whether transfer improvements are statistically meaningful rather than random variation.  
  - **Where used:** `evaluation_engine/statistical_tests.py` and generated `statistical_tests.csv`.

- **Robustness metrics under perturbations**  
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
