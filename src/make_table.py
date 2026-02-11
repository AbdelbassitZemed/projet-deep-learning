import csv
from pathlib import Path

def main():
    p = Path("reports/model_comparison.csv")
    rows = list(csv.DictReader(p.open(encoding="utf-8")))

    # Markdown
    md = []
    md.append("| Modèle | mAP50-95 | mAP50 | Precision | Recall | FPS | Taille (MB) | Params |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['run']} | {float(r['mAP50_95']):.3f} | {float(r['mAP50']):.3f} | {float(r['precision']):.3f} | {float(r['recall']):.3f} | {r['fps_test100']} | {float(r['size_MB']):.1f} | {r['params']} |"
        )
    Path("reports/model_table.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX
    tex = []
    tex.append("\\begin{table}[H]")
    tex.append("\\centering")
    tex.append("\\begin{tabular}{lccccccc}")
    tex.append("\\hline")
    tex.append("Modèle & mAP50-95 & mAP50 & Precision & Recall & FPS & Taille (MB) & Params \\\\")
    tex.append("\\hline")
    for r in rows:
        tex.append(
            f"{r['run']} & {float(r['mAP50_95']):.3f} & {float(r['mAP50']):.3f} & {float(r['precision']):.3f} & {float(r['recall']):.3f} & {r['fps_test100']} & {float(r['size_MB']):.1f} & {r['params']} \\\\"
        )
    tex.append("\\hline")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Comparaison des modèles sur PlantDoc.}")
    tex.append("\\label{tab:model_comparison}")
    tex.append("\\end{table}")
    Path("reports/model_table.tex").write_text("\n".join(tex), encoding="utf-8")

    print("OK reports/model_table.md + reports/model_table.tex")

if __name__ == "__main__":
    main()
