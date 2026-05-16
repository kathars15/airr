# Calibration Data

Runtime and offline calibration data.

Important files:

- `calibration_params.json`: current standard angle-offset result.
- `position_offset.json`: current position-offset result, when position solving succeeds.
- `6dof_params.json`: current segmented 6DoF calibration result.
- `radar_stability_report.json`: latest radar stability analysis report.
- `calibration_history.json`: historical calibration summary.

Important folders:

- `cal_sessions/`: complete per-`done` snapshots that can be replayed offline.
- `raw_pair_records/`: raw radar/optical paired samples.
- `readable_pair_logs/`: human-readable pair logs.
- `parse_logs/`: radar parser diagnostics.

Do not clear this directory when only cleaning active flight data. The `cal_sessions` files are the source for offline replay.
