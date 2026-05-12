# grasp_system (eye-in-hand)

PiPER 6-DoF arm + RealSense D435i + YOLOv8-seg + Open3D,
single-process Python implementation of the eye-in-hand grasping pipeline.

## Layout

```
grasp_system/
├── configs/                       # runtime configuration + calibration output
│   ├── system.yaml
│   ├── classes.yaml
│   ├── camera_intrinsics.npz      # filled by calibrate_intrinsics
│   ├── T_eef_cam.npy              # filled by calibrate_handeye_eih
│   └── observe_joints.npy         # filled by tools.teach_observe
├── calibration/
│   ├── calibrate_intrinsics.py
│   └── calibrate_handeye_eih.py
├── perception/
│   ├── camera.py                  # RealSenseCamera (context managed)
│   ├── detector.py                # YOLOv8-seg wrapper -> Detection[]
│   └── pose_estimator.py          # back-project + clean + OBB + ICP
├── planning/
│   └── grasp_planner.py           # OBB -> top-down grasp frame + opening
├── control/
│   └── piper_controller.py        # unit-safe PiPER SDK V2 wrapper
├── tools/
│   ├── teach_observe.py           # capture observe-pose joints
│   ├── record_demo.py             # record RGB-D frames for dataset building
│   └── visualize.py               # OpenCV / Open3D debug overlays
├── datasets/
│   └── my_first_project_v3i_yolov8 # Roboflow YOLOv8 training dataset
├── training/
│   ├── train_yolov8_seg.py        # trains and exports models/best.pt
│   └── train_yolov8_seg_gpu.py    # CUDA-only launcher for YOLO training
├── models/best.pt                 # YOLOv8s-seg weights (user provided)
├── main_grasp.py                  # end-to-end pick loop
├── common.py                      # transform helpers, config loaders
└── requirements.txt
```

## Bring-up order

1. `python -m grasp_system.calibration.calibrate_intrinsics`
   captures ~20 chessboard views and writes `configs/camera_intrinsics.npz`.
   Target reprojection RMS < 0.5 px.

2. `python -m grasp_system.tools.teach_observe`
   saves the joint angles of a good observe pose into
   `configs/observe_joints.npy`.

3. `python -m grasp_system.calibration.calibrate_handeye_eih`
   drives the interactive eye-in-hand calibration. Uses
   `cv2.calibrateHandEye` in its native convention (pass
   `gripper2base` + `target2cam`, read back `cam2gripper` = `T_E_C`),
   so no input inversion is required. Writes `configs/T_eef_cam.npy`
   and a `handeye_report.npz` QA file. Target position std < 3 mm.

4. Train the YOLOv8-seg detector from the bundled Roboflow dataset:
   `python -m grasp_system.training.train_yolov8_seg`
   For GPU-only training, use:
   `python -m grasp_system.training.train_yolov8_seg_gpu`
   The script reads `configs/system.yaml`, trains on
   `datasets/my_first_project_v3i_yolov8/data.yaml`, and copies the best
   checkpoint to `models/best.pt`.

5. `python -m grasp_system.main_grasp`
   runs the full loop: observe -> detect -> mask cloud -> OBB -> active
   close-up -> plan top-down grasp -> pre-grasp -> MOVE_L approach ->
   close -> lift -> place -> home.

## Variables & units

All internal arithmetic is in SI: meters, radians, seconds, newton-metres.
Unit conversion to PiPER's native 0.001 mm / 0.001 deg / 0.001 N*m happens
inside `control.piper_controller.PiperController` and nowhere else. Euler
angles follow PiPER's extrinsic xyz convention; in scipy this is lowercase
`"xyz"`.

The canonical transform chain is `T_B_O = T_B_E · T_E_C · T_C_O`.
`T_B_E` is read from the arm each frame. `T_E_C` is the hand-eye result.
`T_C_O` is produced by `pose_estimator.estimate_pose_from_obb`.

## Safety notes

- First bring-up should run with `--dry-run` to verify the plan before any
  motion.
- Keep `speed_pct` at 5-10 for initial motion tests; only raise after the
  full loop is validated.
- Verify every hand-eye session with the constant-target test (the
  calibration script prints the std of the board origin in the base
  frame). If std > 5 mm, do not attempt to grasp.
