# Analysis Outputs

Generated figures and CSV summaries from offline tools.

Common outputs:

- `point_records_playback.gif`: quick playback of POINT records and replay tracks.
- `point_mht_before_after.png/csv`: raw representative POINT values vs MHT-filtered output.
- `point_plane_before_after.png`: POINT distribution before and after azimuth filtering on the radar-centered plane.
- `recent_point_time_range_*.png`: simple POINT range-time plots for flight runs.
- `latest_mht_main_tracks_compare.png`: comparison of extracted main MHT tracks from recent runs.
- `mht_pairs_compare_latest.png`: raw calibration pair samples vs MHT-filtered replay.
- `fit_calibration_curves.png`: radar/optical pitch-vs-range scatter and fitted curves.
- `compare_*`: run-to-run comparison plots or CSV summaries.

This folder is for analysis outputs only. Source data usually lives under `Src/calibration_data` or `Src/flight_data_runs`.
