Domain-Shifted Soil Moisture Forecasting
========================================

Synthetic study of irrigation decision support under domain shift. We generate agro-climatic time series for two contrasting regions, train baseline models on Region A, measure performance drop on Region B, and recover with light transfer learning. Outputs include metrics, plots, and a simple irrigation policy simulation.

Pipeline
--------
- **Data generation** (`data_generation.py`): Two regions with different climates—A (semi-arid, hotter, sparse rainfall, faster soil moisture decay) and B (humid, cooler, frequent rainfall). Features: temperature, humidity, rainfall, evapotranspiration, soil_moisture; target: next_day_soil_moisture.
- **Preprocessing** (`preprocessing.py`):
  - Time-aware split (70/30) to prevent leakage.
  - StandardScaler fit on Region A; Region B is scaled with the same scaler to mimic deployment.
  - 7-day sliding windows create sequences for sequence models.
- **Models** (`models.py`, `transfer_learning.py`):
  - RandomForest baseline on flattened windows.
  - LSTM regressor (hidden=64, layers=2, dropout=0.2) trained on Region A with temporal validation (15% tail).
  - Transfer learning: fine-tune the pretrained LSTM on the first 10% of Region B (lr=5e-4, epochs=10).
- **Evaluation** (`evaluation.py`, `main_experiment.py`):
  - Metrics: RMSE, MAE, R² plus degradation/recovery percentages and paired t-test on pre/post TL errors.
  - Plots: prediction vs. actual, domain-shift error distribution, recovery curve.
- **Irrigation simulation** (`irrigation_simulation.py`): Compares a threshold rule vs. ML predictions (threshold 30, 10 units per irrigation) to track water use and over-/under-irrigation events.

Quickstart
----------
```bash
pip install -r requirements.txt
python main_experiment.py
```
Outputs are written to `outputs/`.

Key Results (current run)
-------------------------
- Performance (`outputs/performance_table.csv`):
  - LSTM in-domain (Region A): RMSE 0.53, MAE 0.14, R² 0.31.
  - LSTM cross-domain (Region B): RMSE 70.88, MAE 69.95, R² -37.62.
  - After transfer learning: RMSE 58.18, MAE 57.05, R² -25.02.
  - RF shows similar degradation: RMSE 0.55 → 69.53.
- Shift & recovery (`outputs/degradation_recovery_table.csv`):
  - LSTM degradation: 13,310% RMSE increase; recovery after TL: 17.9% (RMSE), 18.4% (MAE); paired t-test p-value ≈ 0 (errors shrink post-TL).
- Irrigation sim (`outputs/irrigation_table.csv`):
  - Baseline rule never irrigated on this split; ML policy irrigated 3,593 times using 35,930 units of water, with 3,593 over-irrigation events (no under-irrigation), highlighting the effect of forecast bias under shift.

Repository Map
--------------
- `data_generation.py` — synthetic agro-climate series for Regions A & B.
- `preprocessing.py` — temporal split, scaling, and sliding-window creation.
- `models.py` — RandomForest baseline and LSTM regressor training/inference.
- `transfer_learning.py` — fine-tuning utilities for Region B.
- `main_experiment.py` — orchestrates end-to-end experiment, plotting, and tables.
- `evaluation.py` — metrics, degradation/recovery calculations, plots.
- `irrigation_simulation.py` — water-use and decision quality simulation.
- `outputs/` — generated CSVs and PNGs from the latest run.

Reproducibility Notes
---------------------
- Fixed seed: 123 (offset for Region B). GPU is used if available; otherwise CPU.
- Sliding window = 7 days; Region A scaler reused for Region B to emulate deployment shift.
