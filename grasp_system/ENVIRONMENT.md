# Miniconda environment

Use a separate Python 3.10 environment. The base environment on this machine is
Python 3.13, but Open3D and several robotics/vision wheels are not available
for that Python version.

```bash
conda create -y -n grasp_system python=3.10 pip
conda activate grasp_system
python -m pip install -U pip setuptools wheel
python -m pip install "numpy==1.26.4" "scipy>=1.10,<1.12" "opencv-contrib-python==4.11.0.86" "pyyaml>=6.0" "tqdm>=4.66" "matplotlib>=3.7"
# CPU-only fallback:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# For GPU training, install the CUDA-enabled PyTorch wheel that matches this host
# from https://pytorch.org/get-started/locally/ instead of the CPU-only command.
python -m pip install "ultralytics>=8.1" "open3d>=0.17" "pyrealsense2>=2.54.2" "piper_sdk>=0.3"
python -m pip uninstall -y opencv-python
python -m pip install --force-reinstall "numpy==1.26.4" "scipy>=1.10,<1.12" "opencv-contrib-python==4.11.0.86"
```

For GPU training, verify that PyTorch sees CUDA before starting a run:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Ultralytics declares a dependency on the `opencv-python` package name, while
this project intentionally uses `opencv-contrib-python` because it provides the
same `cv2` module plus contrib APIs. `pip check` may report that metadata-only
warning; actual imports have been verified.

Train from the repository root:

```bash
conda activate grasp_system
python -m grasp_system.training.train_yolov8_seg
```

Force CUDA training and fail fast if PyTorch cannot see the GPU:

```bash
conda activate grasp_system
python -m grasp_system.training.train_yolov8_seg_gpu
```
