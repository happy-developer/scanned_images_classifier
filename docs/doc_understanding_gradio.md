# Doc Understanding Gradio App

## Fichier principal

- `src/apps/doc_understanding_gradio.py`

## Prerequis minimaux

```powershell
pip install gradio transformers torch pillow
```

Le modele fine-tune doit etre exporte dans:

- `artifacts/doc_understanding/model/`

Optionnel metadata version:

- `artifacts/doc_understanding/model_meta.json`

## Lancer l'application

```powershell
python src/apps/doc_understanding_gradio.py --model-dir artifacts/doc_understanding/model --model-meta-path artifacts/doc_understanding/model_meta.json --host 127.0.0.1 --port 7861
```

## Comportement attendu

- Upload image document
- Instruction personnalisable
- Inference vision-language
- Sortie JSON:
  - `prediction_json`
  - `raw_text`
  - `latency_ms`
  - `model_version`
  - `status`

## Gestion des erreurs

- `MODEL_NOT_FOUND`
- `MODEL_LOAD_FAILED`
- `DEPENDENCY_MISSING`
- `INVALID_INPUT`
- `INFERENCE_FAILED`

Les erreurs sont retournees en JSON structure.

## Smoke tests

```powershell
python -m unittest tests/apps/test_doc_understanding_gradio_smoke.py
python src/apps/doc_understanding_gradio.py --help
```
