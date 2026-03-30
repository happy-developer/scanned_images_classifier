# Doc Understanding CPU Gradio App

## Fichier principal

- `src/apps/doc_understanding_cpu_gradio.py`

## Objectif

App Gradio CPU dédiée au modèle `doc_understanding_cpu`.

Entrées:
- image upload
- instruction
- OCR text saisi (optionnel)
- chemin CSV OCR (optionnel)

Priorité OCR:
1) texte saisi manuellement
2) lookup CSV via `File Name` + `OCRed Text`

Sortie JSON:
- `prediction_json`
- `raw_text`
- `latency_ms`
- `model_version`
- `status`

## Prerequis

```powershell
pip install -r requirements.inference.txt
```

Le modèle fine-tuné CPU doit être dans:
- `artifacts/doc_understanding_cpu/model/`

Optionnel metadata:
- `artifacts/doc_understanding_cpu/model_meta.json`

## Lancement

```powershell
python src/apps/doc_understanding_cpu_gradio.py --model-dir artifacts/doc_understanding_cpu/model --model-meta-path artifacts/doc_understanding_cpu/model_meta.json --host 127.0.0.1 --port 7862
```

## Smoke tests

```powershell
python -m unittest tests/apps/test_doc_understanding_cpu_gradio_smoke.py
python src/apps/doc_understanding_cpu_gradio.py --help
```
