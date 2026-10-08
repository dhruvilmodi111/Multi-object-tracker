# Automated Tracking Failure & ID Switch Root Cause Analysis

**Total Identified ID Switches:** 1

## Root Cause Breakdown

| Cause                          |   Count | Percentage   |
|:-------------------------------|--------:|:-------------|
| Occlusion / Crossing           |       1 | 100.0%       |
| Missed Detection Gap           |       0 | 0.0%         |
| Fast Motion / Velocity         |       0 | 0.0%         |
| Similar Appearance / Ambiguity |       0 | 0.0%         |

## Incident Log (Sample / Significant Events)

| Frame | GT ID | Old Pred ID | New Pred ID | Primary Cause | Context / Diagnostics |
|---|---|---|---|---|---|
| 91 | 3 | 1 | 6 | **Occlusion / Crossing** | Overlap 0.46 with GT ID 2 during transition |

## Tracker Recommendations:
- For **Occlusion / Crossing**: Increase `max_lost_age` and enable occlusion feature freeze to protect appearance memory.
- For **Missed Detection Gap**: Lower detector confidence threshold or use Kalman coasting propagation.
- For **Fast Motion**: Increase Kalman process noise velocity covariance and relax Mahalanobis gate threshold.
- For **Similar Appearance**: Increase Re-ID gallery size and adjust appearance weighting $\alpha$.
