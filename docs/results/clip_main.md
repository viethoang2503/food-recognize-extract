| run | head | text_mask | modality_dropout | train_frac | acc | acc_lo | acc_hi | top5 | macro_f1 | n |
|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot | zeroshot | - | - | 1 | 71.37 | 70.76 | 71.97 | 87.33 | 70.66 | 22712 |
| image | image | - | - | 1 | 79.06 | 78.56 | 79.57 | 92.51 | 78.95 | 22712 |
| text_none | text | none | - | 1 | 86.35 | 85.92 | 86.76 | 91.19 | 86.40 | 22712 |
| text_strict | text | strict | - | 1 | 38.22 | 37.64 | 38.83 | 57.13 | 38.28 | 22712 |
| late_none | late | none | - | 1 | 94.81 | 94.54 | 95.12 | 98.77 | 94.77 | 22712 |
| late_strict | late | strict | - | 1 | 83.73 | 83.26 | 84.18 | 95.18 | 83.67 | 22712 |
| concat_none | concat | none | 0.1 | 1 | 94.70 | 94.43 | 94.99 | 98.56 | 94.67 | 22712 |
| concat_strict | concat | strict | 0.1 | 1 | 85.08 | 84.61 | 85.56 | 95.29 | 85.03 | 22712 |
| gated_none | gated | none | 0.1 | 1 | 94.84 | 94.56 | 95.16 | 98.49 | 94.80 | 22712 |
| gated_strict | gated | strict | 0.1 | 1 | 84.94 | 84.51 | 85.42 | 95.17 | 84.87 | 22712 |
| xattn_none | xattn | none | 0.1 | 1 | 95.14 | 94.87 | 95.44 | 98.57 | 95.10 | 22712 |
| xattn_strict | xattn | strict | 0.1 | 1 | 85.45 | 85.01 | 85.94 | 95.49 | 85.42 | 22712 |
