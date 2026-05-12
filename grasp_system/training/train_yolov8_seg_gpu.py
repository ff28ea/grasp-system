"""CUDA-only launcher for YOLOv8 segmentation training.

This module keeps the training implementation in ``train_yolov8_seg`` and only
adds GPU defaults plus a CUDA availability check.
"""
from __future__ import annotations

import sys

from . import train_yolov8_seg


GPU_DEFAULTS = (
    ("--device", "0"),
    ("--workers", "4"),
    ("--name", "my_first_project_v3i_yolov8_gpu"),
)


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _option_value(argv: list[str], option: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
        prefix = f"{option}="
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _with_gpu_defaults(argv: list[str]) -> list[str]:
    defaults: list[str] = []
    user_args = argv[1:]
    for option, value in GPU_DEFAULTS:
        if not _has_option(user_args, option):
            defaults.extend([option, value])
    return [argv[0], *defaults, *user_args]


def _device_ids(device: str) -> list[int]:
    normalized = device.strip().lower()
    if normalized == "auto":
        return [0]
    if normalized == "cuda":
        return [0]
    if normalized.startswith("cuda:"):
        normalized = normalized.removeprefix("cuda:")
    if normalized == "cpu":
        raise SystemExit("GPU training script refuses --device cpu. Use train_yolov8_seg.py for CPU.")

    ids: list[int] = []
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise SystemExit(f"Unsupported CUDA device string for GPU training: {device!r}") from exc
    return ids or [0]


def _ensure_cuda(device: str) -> None:
    requested_ids = _device_ids(device)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install a CUDA-enabled PyTorch build first."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch. Check NVIDIA driver, CUDA PyTorch wheel, "
            "and `nvidia-smi` before training."
        )

    device_count = torch.cuda.device_count()
    missing = [device_id for device_id in requested_ids if device_id >= device_count]
    if missing:
        raise SystemExit(
            f"Requested CUDA device(s) {missing}, but PyTorch sees only {device_count} GPU(s)."
        )

    names = ", ".join(torch.cuda.get_device_name(device_id) for device_id in requested_ids)
    print(f"CUDA available: using device={device} ({names})")


def main() -> None:
    argv = _with_gpu_defaults(sys.argv)
    if "-h" in argv[1:] or "--help" in argv[1:]:
        sys.argv = argv
        train_yolov8_seg.main()
        return

    device = _option_value(argv[1:], "--device") or "0"
    _ensure_cuda(device)
    sys.argv = argv
    train_yolov8_seg.main()


if __name__ == "__main__":
    main()
