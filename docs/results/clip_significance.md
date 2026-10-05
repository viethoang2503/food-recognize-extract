| run_a | run_b | n | acc_a | acc_b | diff | diff_lo | diff_hi | only_a_correct | only_b_correct | mcnemar_p |
|---|---|---|---|---|---|---|---|---|---|---|
| clip/runs/image | clip/runs/text_strict | 22712 | 79.06 | 38.22 | -40.84 | -41.63 | -40.04 | 11025 | 1750 | 0 |
| clip/runs/image | clip/runs/xattn_strict | 22712 | 79.06 | 85.45 | 6.39 | 5.96 | 6.84 | 664 | 2115 | 1.4979e-166 |
| clip/runs/text_strict | clip/runs/xattn_strict | 22712 | 38.22 | 85.45 | 47.23 | 46.50 | 47.91 | 444 | 11170 | 0 |
| clip/runs/late_strict | clip/runs/xattn_strict | 22712 | 83.73 | 85.45 | 1.72 | 1.34 | 2.08 | 693 | 1084 | 2.20979e-20 |
| clip/runs/late_strict | clip/runs/concat_strict | 22712 | 83.73 | 85.08 | 1.35 | 0.97 | 1.71 | 717 | 1024 | 2.23915e-13 |
| clip/runs/late_strict | clip/runs/gated_strict | 22712 | 83.73 | 84.94 | 1.21 | 0.88 | 1.58 | 713 | 988 | 3.06274e-11 |
| clip/runs/concat_strict | clip/runs/xattn_strict | 22712 | 85.08 | 85.45 | 0.37 | 0.07 | 0.68 | 612 | 696 | 0.0217357 |
| clip/runs/gated_strict | clip/runs/xattn_strict | 22712 | 84.94 | 85.45 | 0.51 | 0.17 | 0.82 | 658 | 774 | 0.00237389 |
| clip/runs/zeroshot | clip/runs/image | 22712 | 71.37 | 79.06 | 7.69 | 7.22 | 8.14 | 619 | 2366 | 4.25754e-224 |
| runs/image | runs/early_strict | 22712 | 65.56 | 77.65 | 12.09 | 11.57 | 12.72 | 1136 | 3882 | 0 |
| runs/text_strict | runs/early_strict | 22712 | 41.42 | 77.65 | 36.23 | 35.55 | 36.89 | 422 | 8651 | 0 |
| runs/late_strict | runs/early_strict | 22712 | 75.63 | 77.65 | 2.02 | 1.60 | 2.47 | 1043 | 1502 | 1.09923e-19 |
| runs/early_strict | clip/runs/xattn_strict | 22712 | 77.65 | 85.45 | 7.80 | 7.27 | 8.26 | 827 | 2598 | 6.20746e-201 |
