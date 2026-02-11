from pathlib import Path
import yaml

def main():
    base = Path("datasets/plantdoc_clean")
    d = yaml.safe_load((base / "data.yaml").read_text(encoding="utf-8"))
    out = {
        "path": "datasets/plantdoc_clean",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": d["names"]
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/plantdoc.yaml").write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print("OK: data/plantdoc.yaml prêt")

if __name__ == "__main__":
    main()
