# Tools

This directory keeps offline tools and diagnostics outside the online runtime.

Recommended entry points:

- `calibration/`: offline calibration replay, pair-log analysis, curve fitting, and report generation.
- `point_mht/`: POINT records, raw TRACK packets, MHT replay, before/after plots, and playback GIFs.
- `radar_debug/`: UDP/protocol probes and radar range-mode checks.
- `manual_control/`: helper scripts that drive the interactive console or control commands.
- `legacy_debug/`: old one-off experiments kept for reference only.

Compatibility files kept at this level:

- `cal_offset.py`: imported by `core/calibration.py`.
- `compare_point_tracks_vs_raw.py`: imported by `main2.py`.
- `replay_points_mht_compare.py`: forwards to `point_mht/replay_points_mht_compare.py`.
- `replay_calibration_session.py`: forwards to `calibration/replay_calibration_session.py`.

Generated plots and CSV summaries normally go to `D:/desk/airr/calibration_data`.
