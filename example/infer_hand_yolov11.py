from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

DATASET_DIR = Path(__file__).resolve().parent / "Hand Object Detection.v1i.yolov11"
RUNS_DIR = DATASET_DIR / "runs"

# Inference config (ONNX + webcam)
SOURCE = "/dev/video0"  # Linux webcam device
IMG_SIZE = 640
CONF = 0.6

# Postprocess config
IOU = 0.45

# Class names (from your data.yaml)
NAMES = ["hand"]
SHOW = True

def _find_latest_best_onnx(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None

    candidates = list(runs_dir.rglob("best.onnx"))
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)

def _letterbox(image: np.ndarray, new_size: int, color: tuple[int, int, int] = (114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_size - nw
    pad_h = new_size - nh
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    # boxes: (N,4) xyxy
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep

def _prepare_input(frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    img, scale, pad_x, pad_y = _letterbox(frame_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # BCHW
    return img, scale, pad_x, pad_y

def _decode(outputs: list[np.ndarray], scale: float, pad_x: int, pad_y: int, orig_w: int, orig_h: int):
    # Ultralytics export typically returns a single tensor
    pred = outputs[0]
    pred = np.squeeze(pred)

    # Make shape (num_boxes, 4 + nc)
    if pred.ndim == 3:
        pred = np.squeeze(pred, axis=0)
    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected ONNX output shape: {outputs[0].shape}")

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes = pred[:, :4].copy()  # xywh (usually)
    nc = len(NAMES)

    # Two common layouts:
    # - (x, y, w, h, cls0..clsN-1)
    # - (x, y, w, h, obj, cls0..clsN-1)
    if pred.shape[1] == 4 + nc:
        cls_scores = pred[:, 4:]
    elif pred.shape[1] == 5 + nc:
        obj = pred[:, 4:5]
        cls_scores = pred[:, 5:] * obj
    else:
        # Fall back: treat everything after 4 as class-like scores
        cls_scores = pred[:, 4:]

    # If scores look like logits, apply sigmoid
    if cls_scores.size and (cls_scores.max() > 1.0 or cls_scores.min() < 0.0):
        cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    keep = scores > CONF
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # If boxes look normalized, scale to model input size
    if float(boxes.max()) <= 2.0:
        boxes[:, [0, 2]] *= IMG_SIZE
        boxes[:, [1, 3]] *= IMG_SIZE

    # xywh -> xyxy in letterboxed image coords
    xyxy = np.zeros_like(boxes, dtype=np.float32)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # NMS
    keep_idx = _nms(xyxy, scores, IOU)
    xyxy = xyxy[keep_idx]
    scores = scores[keep_idx]
    class_ids = class_ids[keep_idx]

    # Map back to original frame coords
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale

    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, orig_w - 1)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, orig_h - 1)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, orig_w - 1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, orig_h - 1)

    return xyxy, scores, class_ids

def _open_capture(source: str):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    # fallback: many OpenCV builds expect integer index
    if source.startswith("/dev/video"):
        try:
            idx = int(source.replace("/dev/video", ""))
            cap = cv2.VideoCapture(idx)
        except ValueError:
            cap = cv2.VideoCapture(0)
    return cap

def main() -> None:
    weights_onnx = _find_latest_best_onnx(RUNS_DIR)
    if weights_onnx is None:
        raise FileNotFoundError(
            f"Could not find best.onnx under: {RUNS_DIR}\n"
            "Run training first (train_hand_yolov11.py) to export ONNX."
        )

    weights_onnx = weights_onnx.resolve()

    available = set(ort.get_available_providers())
    providers: list[str]
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(weights_onnx), providers=providers)
    input_name = sess.get_inputs()[0].name

    cap = _open_capture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {SOURCE}")

    window_name = "YOLOv11 Hand (ONNX Runtime)"
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        orig_h, orig_w = frame.shape[:2]
        inp, scale, pad_x, pad_y = _prepare_input(frame)

        outputs = sess.run(None, {input_name: inp})
        boxes, scores, class_ids = _decode(outputs, scale, pad_x, pad_y, orig_w, orig_h)

        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            name = NAMES[int(cls_id)] if int(cls_id) < len(NAMES) else str(int(cls_id))
            label = f"{name} {float(score):.2f}"
            cv2.putText(frame, label, (x1i, max(0, y1i - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if SHOW:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

    cap.release()
    if SHOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
