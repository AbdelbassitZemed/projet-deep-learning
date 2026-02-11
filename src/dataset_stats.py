from pathlib import Path
import yaml
from collections import Counter, defaultdict

def read_yaml(p):
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))

def count_instances(split_dir: Path):
    lbl_dir = split_dir / "labels"
    counts = Counter()
    n_files = 0
    n_empty = 0

    for f in lbl_dir.glob("*.txt"):
        n_files += 1
        lines = f.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) == 0:
            n_empty += 1
            continue
        for line in lines:
            cls = int(line.split()[0])
            counts[cls] += 1

    return counts, n_files, n_empty

def main():
    data = read_yaml("data/plantdoc.yaml")
    base = Path(data["path"])
    names = data["names"]
    splits = {
        "train": base / "train",
        "val": base / "valid",
        "test": base / "test",
    }

    total = Counter()
    report = {}

    for s, d in splits.items():
        c, n_files, n_empty = count_instances(d)
        report[s] = {"label_files": n_files, "empty_labels": n_empty, "instances": sum(c.values())}
        total.update(c)

    out = Path("reports")
    out.mkdir(exist_ok=True)

    # Save per-class instances
    lines = ["class_id,class_name,instances"]
    for i in range(len(names)):
        lines.append(f"{i},{names[i]},{total[i]}")
    (out / "class_distribution.csv").write_text("\n".join(lines), encoding="utf-8")

    # Save global report
    lines2 = ["split,label_files,empty_labels,instances"]
    for s in ["train", "val", "test"]:
        r = report[s]
        lines2.append(f"{s},{r['label_files']},{r['empty_labels']},{r['instances']}")
    (out / "dataset_report.csv").write_text("\n".join(lines2), encoding="utf-8")

    print("OK reports/class_distribution.csv + reports/dataset_report.csv")

if __name__ == "__main__":
    main()
