# Terminal Box Runtime

This directory contains the helper files used to build a deployment-ready minimal runtime package for the terminal box.

The actual deployable directory is:

- `package/`

The build script copies only the runtime-required subset from the main project into `package/`, while preserving the original relative directory structure required by imports.

## Files here

- `build_runtime_package.ps1`
  - build the deployable runtime package
- `run_terminal_box.sh`
  - start fused radar + optical runtime on the box
- `run_cv_detect.sh`
  - start optical RTSP + YOLO detection only
- `README.md`
  - deployment and run guide

## Build runtime package

Run on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_runtime_package.ps1
```

After build, the deployable files will be under:

- `package/`

## Copy to the terminal box

Copy the whole `package/` directory to the box, for example:

```text
~/airport/terminal_box_runtime/
```

## Run on the box

### Main fusion program

```bash
cd ~/airport/terminal_box_runtime
export AIRR_ENABLE_TERMINAL_BOX_MODE=1
python3 Src/terminal_box_main.py
```

Or:

```bash
./run_terminal_box.sh
```

### Optical YOLO detection only

```bash
cd ~/airport/terminal_box_runtime
python3 CV/code_image/rtsp_detect_show.py
```

Or:

```bash
./run_cv_detect.sh
```

## What still must be configured

Before running, confirm:

- radar listen IP and port
- optical device IP
- RTSP URL
- YOLO weight path
- Python environment
- CUDA / GPU availability
- output log directory

Key Python packages:

- `torch`
- `torchvision`
- `torchaudio`
- `ultralytics`
- `opencv-python`
- `numpy`
- `scipy`
- `cvxpy`
- `scikit-learn`
- `pyproj`
- `rasterio`

## GPU policy

- If NVIDIA GPU is available:
  - local YOLO is enabled
- If no GPU is available:
  - local YOLO is disabled by default
  - radar + optical-angle fusion still runs

## Output

Terminal output includes:

- fused `ENU`
- fused velocity
- radar class `target_type`
- optical class `class_name`
- radar / optical / CV data ages

Log files include:

- `fusion_output.csv`
- `fusion_output.jsonl`

