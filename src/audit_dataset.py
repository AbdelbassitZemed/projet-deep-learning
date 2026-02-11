# src/audit_dataset.py
from pathlib import Path
import sys
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def load_names(data_yaml: Path):
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = d.get("names")
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
        n = len(names)
    elif isinstance(names, list):
        n = len(names)
    else:
        raise ValueError("names introuvable dans data.yaml")
    return names, n

def audit_split(split_dir: Path, n_classes: int):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT] if img_dir.exists() else []
    labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    missing_labels = sorted(list(img_stems - lbl_stems))
    missing_images = sorted(list(lbl_stems - img_stems))

    empty_label_files = 0
    bad_label_lines = 0

    for t in labels:
        txt = t.read_text(encoding="utf-8").strip()
        if txt == "":
            empty_label_files += 1
            continue
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) != 5:
                bad_label_lines += 1
                continue
            try:
                cls = int(float(parts[0]))
                if not (0 <= cls < n_classes):
                    bad_label_lines += 1
            except:
                bad_label_lines += 1

    return {
        "split": split_dir.name,
        "nb_images": len(images),
        "nb_labels": len(labels),
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "empty_label_files": empty_label_files,
        "bad_label_lines": bad_label_lines,
        "examples_missing_labels": missing_labels[:5],
        "examples_missing_images": missing_images[:5],
    }

def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/plantdoc_raw")

    data_yaml = base / "data.yaml"
    if not data_yaml.exists():
        print(f"data.yaml introuvable dans: {base}")
        sys.exit(1)

    _, n_classes = load_names(data_yaml)
    print("Nb classes:", n_classes)

    for split in ["train", "valid", "test"]:
        split_dir = base / split
        if split_dir.exists():
            print(audit_split(split_dir, n_classes))

if __name__ == "__main__":
    main()
