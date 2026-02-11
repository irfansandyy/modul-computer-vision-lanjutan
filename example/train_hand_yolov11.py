from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path(__file__).resolve().parent / "Hand Object Detection.v1i.yolov11"
DATA_YAML = DATASET_DIR / "data.yaml"

# Training config
PRETRAINED_MODEL = "yolo11n.pt"  # try: yolo11s.pt, yolo11m.pt
IMG_SIZE = 640
EPOCHS = 10
BATCH = -1  # -1 = Ultralytics auto-batch
DEVICE = "0"  # "0" for GPU, or "cpu"
WORKERS = 8
SEED = 42

# Output config
PROJECT_DIR = DATASET_DIR / "runs"
RUN_NAME = "train"

# Export config
EXPORT_ONNX = True
ONNX_OPSET = 12

def main():
    model = YOLO(PRETRAINED_MODEL)
    model.train(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        workers=WORKERS,
        seed=SEED,
    )

    export_model = YOLO(str("/home/sandy/code/bayucaraka/modul-computer-vision-lanjutan/example/Hand Object Detection.v1i.yolov11/runs/train/weights/best.pt"))
    export_model.export(format="onnx", opset=ONNX_OPSET, imgsz=IMG_SIZE)

if __name__ == "__main__":
    main()
