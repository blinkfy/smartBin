#!/usr/bin/env python3
"""Long-lived JSON-lines worker for the real garbage YOLO segmentation model."""

import argparse
import base64
import contextlib
import io
import json
import os
import sys


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.1)
    args = parser.parse_args()
    os.environ.setdefault("YOLO_VERBOSE", "False")

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO(args.model)
        emit({"type": "ready", "task": model.task, "names": model.names})
    except Exception as error:
        emit({"type": "error", "message": f"YOLO模型加载失败：{error}"})
        return

    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = str(request.get("id") or "")
            image_bytes = base64.b64decode(request.get("image") or "", validate=True)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            with contextlib.redirect_stdout(sys.stderr):
                result = model.predict(
                    source=image,
                    imgsz=640,
                    conf=args.confidence,
                    device=args.device,
                    verbose=False,
                )[0]

            detections = []
            if result.boxes is not None:
                for class_id, confidence, box in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.xyxy.tolist(),
                ):
                    detections.append({
                        "className": model.names[int(class_id)],
                        "confidence": round(float(confidence), 6),
                        "box": [round(float(value), 2) for value in box],
                    })
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            emit({"type": "result", "id": request_id, "detections": detections[:12]})
        except Exception as error:
            emit({
                "type": "result",
                "id": str(locals().get("request_id") or ""),
                "error": f"YOLO识别失败：{error}",
                "detections": [],
            })


if __name__ == "__main__":
    main()
