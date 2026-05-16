# Core Runtime Modules

Online runtime code used by `main2.py`.

Main groups:

- `app_config.py`: shared paths, radar connection settings, and calibration constants.
- `radar_protocol.py`: radar packet parsing and control-packet building.
- `radar_receiver.py`: UDP receive loop, POINT/TRACK logging, and parser diagnostics.
- `calibration.py` and `calibration_commands.py`: online calibration state, pair collection, solving, saving, and console commands.
- `interactive_console.py`: interactive command loop, flight-run archive commands, target lock/follow commands.
- `opti.py`, `optical_service.py`, `optical_measurement_log.py`: optical tracker integration and optical measurement logging.
- `track_log.py` and `track_smoothing.py`: track-log lookup and small smoothing helpers.
- `console_utils.py` and `time_utils.py`: shared utilities.

Low-risk cleanup only has been done here. Large online modules are intentionally kept in place so existing imports and runtime behavior stay stable.
