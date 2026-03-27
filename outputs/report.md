# Experiment Report — baseline_run

## Cross-Region Performance
| Train Region   | Test Region   |    RMSE |     MAE |        R2 |   Degradation_% |
|:---------------|:--------------|--------:|--------:|----------:|----------------:|
| A              | B             | 12.1736 | 9.46819 | 0.0705879 |         2490.08 |

## Transfer Learning (Target Region)
|     RMSE |      MAE |       R2 |
|---------:|---------:|---------:|
| 0.382917 | 0.240468 | 0.879428 |

## Statistical Tests
|   t_stat |   p_value |   recovery_RMSE_% |   recovery_MAE_% |
|---------:|----------:|------------------:|-----------------:|
|  986.773 |         0 |           96.8545 |          97.4603 |

## Robustness Checks
| scenario   |    RMSE |      MAE |       R2 |
|:-----------|--------:|---------:|---------:|
| noisy      | 0.38171 | 0.232973 | 0.880187 |
| missing    | 0.41065 | 0.25733  | 0.861331 |