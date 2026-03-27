# Experiment Report — baseline_run

## Cross-Region Performance
| Train Region   | Test Region   |    RMSE |     MAE |        R2 |   Degradation_% |
|:---------------|:--------------|--------:|--------:|----------:|----------------:|
| A              | B             | 12.4221 | 9.56947 | 0.0322676 |         2549.26 |

## Transfer Learning (Target Region)
|     RMSE |      MAE |       R2 |
|---------:|---------:|---------:|
| 0.429155 | 0.264222 | 0.848552 |

## Statistical Tests
|   t_stat |   p_value |   recovery_RMSE_% |   recovery_MAE_% |
|---------:|----------:|------------------:|-----------------:|
|  727.145 |         0 |           96.5452 |          97.2389 |

## Robustness Checks
| scenario   |     RMSE |      MAE |       R2 |
|:-----------|---------:|---------:|---------:|
| noisy      | 0.377777 | 0.241346 | 0.882644 |
| missing    | 0.397546 | 0.255615 | 0.87004  |