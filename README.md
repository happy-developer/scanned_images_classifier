# scanned_images_classifier

## Cible ML actuelle (explicite)

- Le scope actuel est un **classifieur d'images** (pas une refonte OCR complete).
- Dataset cible: Kaggle `osamahosamabdellatif/high-quality-invoice-images-for-ocr` (structure `batch_*` + CSV).
- Resolution des labels:
  - Priorite aux colonnes CSV (`label`, `class`, `class_name`, etc.).
  - Fallback explicite si label absent: **nom du dossier parent de l'image**.
- Le `summary` d'entrainement expose cette regle dans `label_policy` et `label_source_counts`.

## Run Gradio Kaggle

1. Installer les deps execution:
`pip install -r requirements.inference.txt`

2. Lancer l'app:
`python src/apps/gradio_app.py --model-path artifacts/kaggle_faithful/scanned_images_best_model.pth --data-root data/scanned_images_kaggle/dataset --model-meta-path artifacts/model_meta.json --logs-path artifacts/inference_runs.jsonl`

Sortie succes (JSON):
- run_id, label, confidence, probs, model_version, latency_ms, dataset_context

Sortie erreur (JSON):
- error.code parmi INVALID_IMAGE|MODEL_NOT_FOUND|DATASET_UNAVAILABLE|INFERENCE_FAILED
- message
- details

## Tests
`python -m unittest tests/inference/test_predictor_smoke.py tests/inference/test_contract_io.py`

## Test E2E minimal (dataset -> load model -> predict -> contrat JSON)
`python -m unittest tests/inference/test_e2e_minimal_contract.py`

## Pipeline OCR image->texte (batch_1)

Scripts dedies pour fine-tuning OCR (entree image, sortie texte), bases sur CSV `File Name` + `OCRed Text`:

1. Train
`python scripts/ocr_image_train.py --data-root <DATA_ROOT> --output-dir artifacts/ocr_image_text --train-csv batch_1/batch_1/batch1_1.csv --eval-csv batch_1/batch_1/batch1_2.csv --image-subdir-train batch_1/batch_1/batch1_1 --image-subdir-eval batch_1/batch_1/batch1_2`

2. Eval
`python scripts/ocr_image_eval.py --data-root <DATA_ROOT> --artifacts-dir artifacts/ocr_image_text --eval-csv batch_1/batch_1/batch1_2.csv --image-subdir-eval batch_1/batch_1/batch1_2 --max-samples 10`

3. Infer
`python scripts/ocr_image_infer.py --artifacts-dir artifacts/ocr_image_text --data-root <DATA_ROOT> --image batch_1/batch_1/batch1_2/<image_name>.jpg`

Mode smoke (sans fine-tuning complet):
`python scripts/ocr_image_train.py --data-root <DATA_ROOT> --smoke`
