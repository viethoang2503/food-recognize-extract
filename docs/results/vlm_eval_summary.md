| mode | uses_text | n | valid_rate | retry_rate | latency_mean | latency_p90 | dish_acc | dish_acc_lo | dish_acc_hi | unmapped_rate | ingredient_grounding |
|---|---|---|---|---|---|---|---|---|---|---|---|
| image | False | 200 | 100.0% | 0.0% | 3.56s | 4.30s | 61.0% | 54.5% | 67.5% | 24.5% | 14.3% |
| text | True | 200 | 99.5% | 0.5% | 2.91s | 3.97s | 5.5% | 2.5% | 9.0% | 79.9% | 24.0% |
| image_text | True | 200 | 100.0% | 0.0% | 3.57s | 4.26s | 60.0% | 53.0% | 67.0% | 24.5% | 18.7% |

_ingredient_grounding is a reference only: modes with uses_text=True read the text it is measured against. Compare modes with the manual grades._
