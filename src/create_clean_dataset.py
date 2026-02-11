from pathlib import Path
import random
import shutil
import yaml
import cv2

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

def clip01(v): 
    return max(0.0, min(1.0, v))

def clean_label_file(src_txt: Path, dst_txt: Path, n_classes: int):
    txt = src_txt.read_text(encoding="utf-8", errors="ignore").strip()
    if txt == "":
       dst_txt.write_text("", encoding="utf-8")
       return True


    cleaned = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except:
            continue
        if cls < 0 or cls >= n_classes:
            continue

        x, y, w, h = clip01(x), clip01(y), clip01(w), clip01(h)
        if w <= 0 or h <= 0:
            continue

        cleaned.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if len(cleaned) == 0:
        return False

    dst_txt.write_text("\n".join(cleaned), encoding="utf-8")
    return True

def copy_pair(img_path: Path, lbl_path: Path, out_img: Path, out_lbl: Path, n_classes: int):
    # check image readable
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return False

    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, out_img)

    if lbl_path.exists():
        ok = clean_label_file(lbl_path, out_lbl, n_classes)
        return ok
    return False

def main():
    raw = Path("datasets/plantdoc_raw")
    clean = Path("datasets/plantdoc_clean")
    clean.mkdir(parents=True, exist_ok=True)

    names, n_classes = load_names(raw / "data.yaml")
    # copy data.yaml
    shutil.copy2(raw / "data.yaml", clean / "data.yaml")

    train_img = raw / "train" / "images"
    train_lbl = raw / "train" / "labels"

    images = [p for p in train_img.iterdir() if p.suffix.lower() in IMG_EXT]
    images.sort()
    random.seed(42)
    random.shuffle(images)

    val_ratio = 0.2
    n_val = int(len(images) * val_ratio)
    val_set = set([p.name for p in images[:n_val]])

    # build clean train/valid
    kept_train = kept_val = 0

    for img in images:
        stem = img.stem
        lbl = train_lbl / f"{stem}.txt"

        if img.name in val_set:
            out_img = clean / "valid" / "images" / img.name
            out_lbl = clean / "valid" / "labels" / f"{stem}.txt"
            ok = copy_pair(img, lbl, out_img, out_lbl, n_classes)
            kept_val += 1 if ok else 0
        else:
            out_img = clean / "train" / "images" / img.name
            out_lbl = clean / "train" / "labels" / f"{stem}.txt"
            ok = copy_pair(img, lbl, out_img, out_lbl, n_classes)
            kept_train += 1 if ok else 0

    # copy test if exists
    if (raw / "test").exists():
        test_img = raw / "test" / "images"
        test_lbl = raw / "test" / "labels"
        if test_img.exists():
            for img in test_img.iterdir():
                if img.suffix.lower() not in IMG_EXT:
                    continue
                stem = img.stem
                lbl = test_lbl / f"{stem}.txt"
                out_img = clean / "test" / "images" / img.name
                out_lbl = clean / "test" / "labels" / f"{stem}.txt"
                copy_pair(img, lbl, out_img, out_lbl, n_classes)

    print("Dataset clean créé.")
    print("Train kept (avec labels valides):", kept_train)
    print("Valid kept (avec labels valides):", kept_val)

if __name__ == "__main__":
    main()
