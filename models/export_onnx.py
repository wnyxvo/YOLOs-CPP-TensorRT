#!/usr/bin/env python3
"""Export an Ultralytics YOLO model to a static-shape ONNX file.

The generated graph is static by default because tools/onnx2trt.cpp builds
TensorRT engines without an optimization profile for dynamic input shapes.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolo11n",
        help="Ultralytics model name or path to a .pt/.yaml model (default: yolo11n)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Square export image size (default: 640)"
    )
    parser.add_argument(
        "--opset", type=int, default=11, help="ONNX opset version (default: 11)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()

    # Put named Ultralytics weights in this directory so the ONNX output lands
    # alongside the project's model files. Preserve explicit relative/absolute
    # paths supplied by the user.
    if not model_path.suffix:
        model_path = model_path.with_suffix(".pt")
    if not model_path.is_absolute() and len(model_path.parts) == 1:
        model_path = Path(__file__).resolve().parent / model_path

    model = YOLO(str(model_path))
    output = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        opset=args.opset,
    )
    print(f"ONNX model written to: {output}")


if __name__ == "__main__":
    main()
