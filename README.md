Research-Grade Domain-Shift Pipeline
====================================

Production-style, modular pipeline to test cross-region irrigation forecasting, domain shift, and transfer learning. Architecture is split into engines for data, preprocessing, modeling, training, transfer, evaluation, visualization, and experiment runners. Supports synthetic Regions A/B or external CSV/JSON data with the same schema.

Key Structure
-------------
- `data_engine/`: synthetic generator, region parametrization, domain-shift metrics.
- `preprocessing_engine/`: time-based split, scaler (fit-on-train), sliding windows, leakage validation.
- `modeling_engine/`: RF + Linear baselines, LSTM architecture, model factory.
- `training_engine/`: deterministic seeds, generic trainer, cross-region trainer.
- `transfer_engine/`: fine-tuning strategies (full/partial freeze).
- `evaluation_engine/`: metrics, cross-region matrix, statistical tests, robustness analysis.
- `visualization_engine/`: centralized plotting + report writer.
- `experiment_runner/`: `run_baseline.py`, `run_transfer.py`, `run_full_experiment.py`.
- `configs/experiment_config.yaml`: config-driven seeds, data (synthetic vs external paths), preprocessing, model, training, transfer, evaluation, output paths.
- `outputs/`: logs, metrics, plots, report.

Quickstart
----------
```bash
pip install -r requirements.txt
# baseline only
python experiment_runner/run_baseline.py --config configs/experiment_config.yaml
# transfer (10% target fine-tune)
python experiment_runner/run_transfer.py --config configs/experiment_config.yaml
# full (baseline + transfer + robustness + stats)
python main.py --config configs/experiment_config.yaml
```

What It Produces
----------------
- Tables: cross-region matrix, transfer metrics, statistical tests, robustness.
- Plots: prediction vs actual, domain-shift error comparison, degradation/recovery bars, transfer recovery curve, feature distributions.
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
- Temporal splits only; scaler fit on train; Region A scaler reused for Region B to emulate deployment.
- Minimum 5 runs enforced for statistical testing; paired t-tests on pre/post transfer errors.
