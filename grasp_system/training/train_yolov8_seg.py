"""Train the YOLOv8 segmentation model used by grasp_system.

The default dataset is the Roboflow YOLOv8 export under
``grasp_system/datasets/my_first_project_v3i_yolov8``. The best checkpoint is
copied to the path configured in ``configs/system.yaml`` so
``SegmentationDetector`` can load it without extra wiring.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - reported cleanly by main()
    YOLO = None  # type: ignore[assignment]


PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_YAML = PKG_ROOT / "datasets" / "my_first_project_v3i_yolov8" / "data.yaml"


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PKG_ROOT / path).resolve()


def _resolve(path: str | Path, base: Path = PKG_ROOT) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _write_ultralytics_data_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Return a data.yaml whose relative dataset root is anchored to the file."""
    dataset_root = data.get("path")
    if dataset_root is None:
        return path

    dataset_root_path = Path(str(dataset_root)).expanduser()
    if dataset_root_path.is_absolute():
        return path

    normalized = dict(data)
    normalized["path"] = str((path.parent / dataset_root_path).resolve())
    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{path.stem}_",
        suffix=path.suffix,
        delete=False,
    )
    with temp:
        yaml.safe_dump(normalized, temp, sort_keys=False)
    return Path(temp.name)


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_yaml(project_path(path))


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def _default_workers(device: str) -> int:
    if device == "cpu":
        return 0
    return 4


def _class_names_from_system(path: str | Path) -> list[str]:
    data = _load_yaml(project_path(path))
    classes = data.get("classes", {}) or {}
    return [str(classes[index].get("name", index)) for index in sorted(classes)]


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        "--config",
        type=Path,
        default=PKG_ROOT / "configs" / "system.yaml",
        help="grasp_system runtime config used for defaults.",
    )
    base_args, remaining = base.parse_known_args()

    config = load_config(base_args.config)
    camera = config.get("camera", {})
    paths = config.get("paths", {})

    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg for the eye-in-hand grasp system.",
        parents=[base],
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_path(paths.get("dataset", DEFAULT_DATASET_YAML)),
        help="YOLO dataset yaml. Defaults to the bundled Roboflow export.",
    )
    parser.add_argument(
        "--model",
        default="yolov8s-seg.pt",
        help="Ultralytics checkpoint or model yaml used as the training base.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(camera.get("width", 640)),
        help="Training image size. Defaults to camera.width from system.yaml.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Ultralytics device string: "auto", "cpu", "0", "0,1", etc.',
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader workers. Defaults to 0 on CPU and 4 on CUDA.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=PKG_ROOT / "runs" / "segment",
        help="Directory for Ultralytics training runs.",
    )
    parser.add_argument("--name", default="my_first_project_v3i_yolov8")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conf",
        type=float,
        default=float(config.get("perception", {}).get("conf_threshold", 0.5)),
        help="Validation confidence threshold, aligned with system.yaml.",
    )
    parser.add_argument(
        "--copy-best",
        type=Path,
        default=project_path(paths.get("model", "models/best.pt")),
        help="Destination for the best checkpoint used by grasp_system.",
    )
    parser.add_argument(
        "--no-copy-best",
        action="store_true",
        help="Keep the trained checkpoint only in the Ultralytics run folder.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the requested model/checkpoint.",
    )
    return parser.parse_args(remaining, namespace=base_args)


def main() -> None:
    args = parse_args()
    if YOLO is None:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r grasp_system/requirements.txt"
        )

    data_yaml = _resolve(args.data)
    project_dir = _resolve(args.project)
    copy_best = _resolve(args.copy_best) if args.copy_best else None

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    data = _load_yaml(data_yaml)
    names = data.get("names", [])
    nc = int(data.get("nc", len(names)))
    if nc != 4:
        raise ValueError(f"Expected 4 grasp classes, got nc={nc} in {data_yaml}")

    config = load_config(args.config)
    class_names = _class_names_from_system(config.get("paths", {}).get("classes", "configs/classes.yaml"))
    if list(names) != class_names:
        raise ValueError(
            "Dataset names must match configs/classes.yaml order. "
            f"data.yaml names={list(names)}, classes.yaml names={class_names}"
        )
    ultralytics_data_yaml = _write_ultralytics_data_yaml(data_yaml, data)

    device = _select_device(args.device)
    workers = args.workers if args.workers is not None else _default_workers(device)

    model = YOLO(args.model)
    results = model.train(
        data=str(ultralytics_data_yaml),
        task="segment",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=workers,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        exist_ok=True,
        resume=args.resume,
        conf=args.conf,
    )

    save_dir = Path(getattr(results, "save_dir", project_dir / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training finished but best checkpoint is missing: {best_pt}")

    if copy_best is not None and not args.no_copy_best:
        copy_best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, copy_best)
        print(f"Copied best checkpoint to {copy_best}")

    print(f"Training run saved to {save_dir}")


if __name__ == "__main__":
    main()
