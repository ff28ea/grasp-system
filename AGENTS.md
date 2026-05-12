# Repository Guidelines

## Project Structure & Module Organization

This repository contains a single Python robotics package under `grasp_system/`.
Core modules are split by responsibility: `calibration/` for camera and
eye-in-hand calibration, `perception/` for RealSense capture, YOLOv8-seg
detection, and pose estimation, `planning/` for grasp planning, `control/` for
the PiPER SDK wrapper, `tools/` for data collection and visualization, and
`training/` for detector training. Runtime configuration and calibration
outputs live in `grasp_system/configs/`. The bundled Roboflow dataset is in
`grasp_system/datasets/my_first_project_v3i_yolov8/`; trained weights are copied
to `grasp_system/models/best.pt`.

## Build, Test, and Development Commands

Create the expected Python 3.10 environment from the repository root:

```bash
conda create -y -n grasp_system python=3.10 pip
conda activate grasp_system
python -m pip install -r grasp_system/requirements.txt
```

Useful entry points:

```bash
python -m grasp_system.calibration.calibrate_intrinsics
python -m grasp_system.calibration.calibrate_handeye_eih
python -m grasp_system.training.train_yolov8_seg
python -m grasp_system.training.train_yolov8_seg_gpu
python -m grasp_system.main_grasp --dry-run
```

Use the GPU training launcher when CUDA must be available; it fails fast if
PyTorch cannot see the GPU.

## Coding Style & Naming Conventions

Use Python 3.10 syntax, 4-space indentation, type hints for public helpers, and
small modules with clear hardware boundaries. Keep internal math in SI units
(meters, radians, seconds, N*m); convert to PiPER native units only inside
`control.piper_controller.PiperController`. Prefer `snake_case` for functions,
variables, and files; use `PascalCase` for classes. Keep YAML paths relative to
`grasp_system/` unless an absolute hardware path is required.

## Testing Guidelines

There is no formal test suite yet. Before submitting changes, at minimum run:

```bash
python -m py_compile $(find grasp_system -name '*.py')
```

For perception or training changes, run a YOLO validation/training smoke test
against `grasp_system/datasets/my_first_project_v3i_yolov8/data.yaml`. For robot
motion changes, first use `python -m grasp_system.main_grasp --dry-run` and keep
initial speed low.

## Commit & Pull Request Guidelines

No Git history is present in this checkout, so use concise imperative commit
messages such as `Fix dataset path resolution` or `Add dry-run grasp checks`.
Pull requests should describe the behavior changed, list commands run, note
hardware used, and include calibration or training metrics when relevant. Add
screenshots or sample overlay images for visualization and segmentation changes.

## Security & Configuration Tips

Do not commit private credentials, machine-specific absolute paths, or large
temporary training artifacts. Treat `configs/camera_intrinsics.npz`,
`configs/T_eef_cam.npy`, and `configs/observe_joints.npy` as hardware-specific
outputs. Recalibrate after changing camera mounting, end-effector geometry, or
RealSense resolution.
