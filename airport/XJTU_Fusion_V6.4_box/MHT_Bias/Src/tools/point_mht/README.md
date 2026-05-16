# POINT / MHT Analysis Tools

Offline tools for checking whether raw POINT records or radar TRACK packets are more stable.

Main scripts:

- `replay_points_mht_compare.py`: replay `point_records.csv` or calibration pair logs through the current MHT algorithm.
- `plot_point_mht_before_after.py`: compare one point-record run before and after MHT filtering; edit tuning parameters at the top of the file.
- `animate_point_records.py`: make a fast playback GIF of POINT records and confirmed replay tracks.
- `plot_recent_point_track_runs.py`: draw simple range-time plots for the latest flight runs.
- `plot_latest_mht_main_tracks.py`: compare the main MHT tracks extracted from recent runs.
- `plot_point_plane_before_after.py`: show point filtering on a radar-centered XY plane.
- `filter_point_records.py`: filter `point_records.csv` by azimuth range.
- `compare_two_run_mht_filtered_pitch.py`: compare two replayed MHT tracks.
- `compare_two_run_raw_track_filtered_pitch.py`: compare two raw TRACK-packet runs.
- `compare_two_run_pitch_consistency.py`: compare pitch-vs-range consistency across two runs.

Typical input:

- `Src/flight_data_runs/flight_run_*/point_records.csv`
- `Src/flight_data_runs/flight_run_*/raw_tracks.csv`
- `Src/data/point_records.csv` for the current active recording.

Typical output:

- `D:/desk/airr/calibration_data/*.png`
- `D:/desk/airr/calibration_data/*.csv`
