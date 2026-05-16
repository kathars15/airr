# Flight Data Runs

Archived flight-recording batches.

Each `flight_run_YYYYmmdd_HHMMSS` folder is intended to be one independent recording window saved by the console workflow.

Common files in a run:

- `point_records.csv`: parsed raw POINT packet rows.
- `raw_tracks.csv`: parsed raw TRACK packet rows from the radar.
- `track_log.txt`: readable track log.
- `track_results.json`: runtime track result snapshots.
- `point_track_results*.json`: offline MHT replay results from POINT records.
- `point_vs_raw_track_compare*.csv`: comparison between replayed point tracks and raw TRACK packets.

Keep these folders when comparing multiple flights. They are the main evidence for point-vs-track stability checks.
