# Radar Debug Tools

Small hardware/protocol diagnostics.

Scripts:

- `radar_udp_probe.py`: bind the radar receive address and print packet headers/lengths.
- `diagnose_radar_range_mode.py`: check whether saved TRACK range behaves like slant range or horizontal range.
- `test_read.py` and `test_simple.py`: old socket read experiments.

Use these before changing the MHT or calibration logic when the issue may be packet format, IP/port, or radar range interpretation.
