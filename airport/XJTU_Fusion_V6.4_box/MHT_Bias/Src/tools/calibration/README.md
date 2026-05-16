# Calibration Tools

Offline tools for radar/optical calibration.

Use these when you want to replay old `cal` data, inspect pair logs, fit simple curves, or generate a report without running hardware.

Main scripts:

- `replay_calibration_session.py`: replay one saved `cal_sessions/*.json` through the current online calibration calculation and rewrite the standard calibration result files.
- `offline_calibrate_from_logs.py`: build calibration input from saved radar/optical logs.
- `fit_calibration_curves.py`: draw radar/optical pitch-vs-range scatter and fitted curves from a session JSON.
- `compare_runs_radar_pitch.py`: compare radar pitch trends between saved raw pair runs.
- `plot_pitch_by_track.py`: draw pitch-vs-range by track id for one raw pair file.
- `generate_calibration_model_report.py`: generate the calibration modeling report document/figure inputs.

Important data folders:

- `../calibration_data/cal_sessions`: complete per-`done` calibration snapshots.
- `../calibration_data/raw_pair_records`: raw paired radar/optical samples.
- `../calibration_data/readable_pair_logs`: human-readable pair logs.

Most figures are written to `D:/desk/airr/calibration_data`.
