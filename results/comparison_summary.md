# Tracking Performance Comparison (Before vs After Tuning)

Sequence: `MOT17-SYNTH-OCCLUSION`

| Metric    | Baseline (Standard SORT)   | Tuned (OcclusionDeepSORT)   | Delta / Improvement   |
|:----------|:---------------------------|:----------------------------|:----------------------|
| MOTA      | 93.40%                     | 93.94%                      | +0.53%                |
| IDF1      | 91.16%                     | 96.88%                      | +5.71%                |
| IDSW      | 1                          | 0                           | -100.0%               |
| Precision | 100.00%                    | 100.00%                     | +0.00%                |
| Recall    | 93.58%                     | 93.94%                      | +0.36%                |
| MT        | 4                          | 4                           | +0                    |
| ML        | 0                          | 0                           | +0                    |

### Key Takeaways:
- **IDF1 Improvement:** 91.16% -> 96.88%
- **ID Switches (IDSW):** Reduced from 1 down to 0
- **Occlusion Re-ID:** Lost-track appearance buffering prevents spurious ID recreation.
