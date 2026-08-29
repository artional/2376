# Method-to-code mapping

| Paper component | Code |
|---|---|
| `D_clean`, `D_sel`, `D_ref`, `D_fit`, `D_cal` | `cli/prepare_splits.py` |
| Candidate trigger set and Eqs. (6)-(10) | `cli/select_triggers.py` |
| Defensive trigger set `T_sel` | `selected_defense_triggers.json` |
| Probing trigger set `T_probe` | `probe_triggers.json` |
| Defensive fine-tuning, Eq. (11) | `cli/build_defense_sft.py` |
| Probe-induced response, Eq. (12) | `extract_probe_responses()` |
| Reference subspaces `E_def` and `E_ben` | `fit_reference_basis()` |
| Response geometry `v_geo`, Eq. (13) | `build_detection_features()` |
| Reference alignment `v_ref`, Eq. (14) | `build_detection_features()` |
| Response magnitude `v_mag` | `build_detection_features()` |
| Detection representation `v_det`, Eq. (15) | `build_detection_features()` |
| Logistic-regression detector and Eq. (16) | `cli/fit_detector.py` |
| Online defense | `cli/online_defense.py` |
| Resident detector service | `service/realtime_detector.py` |
| Official evaluation | `cli/evaluate_official.py` |

The default settings use `K_probe=2`, `K_aug=2`, `K_geo=16`, `K_ref=4`,
hidden-state indices `[14,16,18,20]`, and q99 threshold calibration.
