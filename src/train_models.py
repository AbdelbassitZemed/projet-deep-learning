from ultralytics import YOLO
from pathlib import Path
import time
import torch
import csv

DATA = "data/plantdoc.yaml"

def model_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def benchmark_images(model_path: str, images_dir: Path, n=100, conf=0.25, device="cpu"):
    import cv2
    model = YOLO(model_path)
    imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.webp"))
    imgs = imgs[:n] if len(imgs) > n else imgs

    if len(imgs) == 0:
        return None, None

    t0 = time.time()
    for im in imgs:
        frame = cv2.imread(str(im))
        _ = model.predict(frame, conf=conf, device=device, verbose=False)
    t1 = time.time()

    elapsed = t1 - t0
    fps = len(imgs) / elapsed if elapsed > 0 else 0
    ms = 1000.0 * elapsed / len(imgs)
    return fps, ms

def train_one(weights: str, run_name: str, imgsz=416, epochs=10, batch=8, device="cpu"):
    model = YOLO(weights)

    results = model.train(
        data=DATA,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=0,
        project="runs_compare",
        name=run_name,
        seed=42,
        patience=10
    )

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"

    best = YOLO(str(best_pt))
    metrics = best.val(data=DATA, imgsz=imgsz, device=device, workers=0)

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    size_mb = best_pt.stat().st_size / 1e6
    params = model_params(best.model)

    data_yaml = Path(DATA).read_text(encoding="utf-8")
    base_path = None
    for line in data_yaml.splitlines():
        if line.strip().startswith("path:"):
            base_path = line.split(":", 1)[1].strip()
            break
    base_path = Path(base_path) if base_path else None
    test_images = base_path / "test" / "images" if base_path else None

    fps, ms = (None, None)
    if test_images and test_images.exists():
        fps, ms = benchmark_images(str(best_pt), test_images, n=100, device=device)

    return {
        "run": run_name,
        "weights": weights,
        "precision": mp,
        "recall": mr,
        "mAP50": map50,
        "mAP50_95": map5095,
        "size_MB": size_mb,
        "params": params,
        "fps_test100": fps if fps is not None else "",
        "ms_per_image": ms if ms is not None else "",
        "best_pt": str(best_pt).replace("\\", "/")
    }

def main():
    Path("reports").mkdir(exist_ok=True)

    configs = [
        ("yolov8n.pt", "plantdoc_y8n"),
        ("yolov8s.pt", "plantdoc_y8s"),
        ("yolo11n.pt", "plantdoc_y11n"),
        ("yolo11s.pt", "plantdoc_y11s"),
    ]

    out_csv = Path("reports/model_comparison.csv")
    headers = ["run", "weights", "precision", "recall", "mAP50", "mAP50_95", "size_MB", "params", "fps_test100", "ms_per_image", "best_pt"]

    rows = []
    for w, name in configs:
        print("\n==============================")
        print("TRAIN:", w, "=>", name)
        try:
            r = train_one(w, name, imgsz=416, epochs=10, batch=8, device="cpu")
            rows.append(r)
            print("OK:", r)
        except Exception as e:
            print("ERROR on", name, ":", e)

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for rr in rows:
                writer.writerow(rr)

    print("\nOK => reports/model_comparison.csv")

if __name__ == "__main__":
    main()
