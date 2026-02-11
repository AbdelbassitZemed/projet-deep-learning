```md
# Projet Deep Learning — Détection des maladies des plantes (PlantDoc) avec YOLO + Streamlit

Ce projet (PFA) met en place un système de **détection automatique des maladies des plantes à partir d’images** en utilisant des modèles **YOLO (Ultralytics)**, puis une **interface web Streamlit** pour tester la détection de manière simple.

## Objectif
- Détecter et classifier des maladies sur des images de feuilles et de plantes en **scènes naturelles**
- Comparer plusieurs variantes de YOLO (v8 et v11)
- Retenir le meilleur compromis précision / vitesse
- Déployer une démo via **Streamlit**

## Dataset
- Dataset de type **PlantDoc**
- Images en conditions réelles (variations d’éclairage, arrière-plans complexes, occlusions)
- Annotations au **format YOLO** (classe + bbox normalisée)

Le dataset n’est généralement pas versionné dans GitHub (taille). Place-le localement dans `datasets/`.

## Pipeline (résumé)
1. **Audit et nettoyage** du dataset (cohérence images/labels, coordonnées valides, etc.)
2. **Split** train / val et conservation d’un **jeu de test**
3. Entraînement homogène de plusieurs modèles
4. Évaluation avec `Precision`, `Recall`, `mAP@0.5`, `mAP@0.5:0.95`, matrices de confusion, courbes PR/F1
5. Déploiement d’une interface **Streamlit** pour tester sur une image

## Modèles évalués
- YOLOv8n
- YOLOv8s
- YOLOv11n
- YOLOv11s

## Résultats (comparaison)
| Modèle | mAP50-95 | mAP50 | Precision | Recall | FPS | Taille (MB) |
|---|---:|---:|---:|---:|---:|---:|
| plantdoc_y8n  | 0.166 | 0.242 | 0.301 | 0.321 | 23.51 | 6.2 |
| plantdoc_y8s  | 0.256 | 0.365 | 0.388 | 0.365 | 10.85 | 22.5 |
| plantdoc_y11n | 0.164 | 0.237 | 0.241 | 0.348 | 15.72 | 5.4 |
| plantdoc_y11s | 0.273 | 0.394 | 0.402 | 0.423 | 9.34  | 19.2 |

✅ **YOLOv11s** est retenu comme meilleur compromis (meilleures métriques globales avec une vitesse acceptable).

## Environnement et outils
- Python
- PyTorch
- Ultralytics YOLO
- Streamlit
- VS Code

## Structure du projet (exemple)
```

plant_pfa_yolo/
├─ app.py                  # App Streamlit (démo)
├─ requirements.txt
├─ src/
│  ├─ audit_dataset.py     # contrôle qualité
│  ├─ create_clean_dataset.py
│  ├─ train_models.py      # entraînement + comparaison
│  └─ make_table.py        # génération tableau de comparaison
├─ datasets/               # dataset local (non versionné)
├─ runs_finetune/          # sorties d'entraînement (non versionné)
└─ runs_compare/           # comparaisons

````

## Installation
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
````

## Entraînement (YOLOv11s)

> Exemple de commande utilisée pour le fine-tuning

```bash
yolo detect train data=datasets/plantdoc_clean/data.yaml \
model=yolo11s.pt imgsz=512 epochs=50 batch=4 device=cpu
```

### Paramètres (fine-tuning YOLOv11s)

* Epochs: 50
* Image size: 512×512
* Batch size: 4
* Optimizer: AdamW
* Matériel: CPU Intel Core i7

## Validation (jeu de test)

```bash
yolo detect val model=runs_finetune/yolo11s_ft_cpu/weights/best.pt \
data=datasets/plantdoc_clean/data.yaml split=test imgsz=512
```

## Inférence sur une image

```bash
yolo detect predict model=runs_finetune/yolo11s_ft_cpu/weights/best.pt \
source=datasets/plantdoc_clean/test/images/image.jpg imgsz=512
```

## Lancer l’interface Streamlit

1. Mettre le poids du modèle (par ex. `best.pt`) à l’endroit attendu par `app.py`
2. Lancer l’app

```bash
streamlit run app.py
```

L’interface permet de **charger une image**, lancer la détection, puis afficher les **boîtes englobantes**, la **classe prédite** et le **score de confiance**.

## Notes

* Les dossiers `datasets/`, `runs*/`, `.venv/` sont à ignorer via `.gitignore` (trop volumineux ou générés automatiquement).
* Si tu veux versionner des poids `.pt` volumineux, pense à Git LFS.


