# Data Science Runbook

## 1) Installation

```powershell
pip install kagglehub
pip install -r notebooks/requirements_kaggle_faithful_local.txt
```

## 2) Recuperer le dataset officiel

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("osamahosamabdellatif/high-quality-invoice-images-for-ocr")

print("Path to dataset files:", path)
```

Le chemin `path` est la racine du dataset a passer au pipeline/Gradio.

## 3) Entrainement pipeline

Auto-download direct:

```powershell
python src/scanned_images_pipeline.py --download-latest --epochs 1 --batch-size 8 --disable-pretrained
```

Avec dataset deja present:

```powershell
python src/scanned_images_pipeline.py --data-root <path_retourne_par_kagglehub> --epochs 8 --batch-size 16 --img-size 224
```

## 4) Inference Gradio

```powershell
pip install -r requirements.inference.txt
python src/apps/gradio_app.py --model-path artifacts/kaggle_faithful/scanned_images_best_model.pth --data-root <path_retourne_par_kagglehub> --logs-path artifacts/inference_runs.jsonl
```

## 5) Notes

- Le projet est en mode Kaggle-only.
- Aucun fallback `_synthetic` n'est conserve.
- La structure attendue dans le dataset est `batch_*` + manifests CSV.
- Cible ML actuelle: classifieur d'images (pas de refonte OCR complete).
- Resolution des labels:
  - Priorite aux colonnes CSV (`label`, `class`, `class_name`, etc.).
  - Fallback explicite si label absent: dossier parent image (ex: `batch1_1`, `batch1_2`, `batch1_3`).
- Le `summary` de `run_training_pipeline` expose `label_policy` et `label_source_counts`.

## 6) Validation E2E minimale (supervision Erdos)

```powershell
python -m unittest tests/inference/test_e2e_minimal_contract.py
```

Ce test couvre: verification dataset -> chargement modele -> prediction 1 image -> contrat JSON.
