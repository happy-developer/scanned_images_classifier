# Architecture Cible - Fine-Tuning Document Understanding (Gemma 3n)

## 1) Arborescence cible (industrialisable)

```text
scanned_images_classifier/
  docs/
    doc_understanding_architecture.md
  configs/
    doc_understanding/
      train.yaml
      data.yaml
      inference.yaml
  data/
    kaggle_invoice_images/                  # source KaggleHub (batch_*/csv)
    processed/
      doc_understanding/
        train.jsonl
        val.jsonl
        test.jsonl
  artifacts/
    doc_understanding/
      checkpoints/
      final_model/
      tokenizer/
      processor/
      metrics.json
      run_metadata.json
  src/
    doc_understanding/
      __init__.py
      config.py
      data_loader.py                        # parsing CSV + validation
      dataset_builder.py                    # transformation en conversations VLM
      prompting.py                          # instruction templates
      model_factory.py                      # chargement modèle/base + PEFT
      train.py                              # orchestration fine-tuning
      evaluate.py                           # métriques field-level / exact-match
      infer.py                              # inférence réutilisable
      serialization.py                      # save/load modèle + métadonnées
      tracking.py                           # run_id, logs, lineage
    apps/
      doc_understanding_gradio.py           # UI dédiée
  scripts/
    doc_understanding_train.py
    doc_understanding_eval.py
    doc_understanding_infer.py
  tests/
    unit/
      test_data_loader.py
      test_prompting.py
      test_contract_io.py
    integration/
      test_train_smoke.py
      test_infer_smoke.py
      test_gradio_smoke.py
```

Séparation stricte:
- Entraînement: `src/doc_understanding/train.py`
- Évaluation: `src/doc_understanding/evaluate.py`
- Inférence: `src/doc_understanding/infer.py`
- UI: `src/apps/doc_understanding_gradio.py`

## 2) Standards reproductibilité / packaging / traçabilité

### Reproductibilité
- Configuration centralisée YAML (`configs/doc_understanding/*.yaml`).
- Seed global fixé (Python/NumPy/Torch).
- Versionning des dépendances (fichier requirements ou lockfile).
- `run_metadata.json` obligatoire:
  - `run_id`, date, git commit (si dispo), config, dataset_root, split, modèle de base.

### Packaging
- Pas de logique critique laissée uniquement dans notebook.
- Notebook = démonstration; scripts CLI = source d’exécution officielle.
- Points d’entrée:
  - `python scripts/doc_understanding_train.py ...`
  - `python scripts/doc_understanding_eval.py ...`
  - `python scripts/doc_understanding_infer.py ...`
  - `python src/apps/doc_understanding_gradio.py ...`

### Traçabilité
- Sauvegarde systématique:
  - modèle fine-tuné,
  - tokenizer/processor,
  - hyperparamètres,
  - métriques,
  - mapping des champs cibles.
- Logs structurés JSONL pour train/eval/infer.

## 3) Contrat I/O inférence (document understanding)

### Entrée
```json
{
  "image_path": "string (ou image uploadée)",
  "instruction": "string optionnelle",
  "max_new_tokens": "int optionnel"
}
```

### Sortie succès
```json
{
  "run_id": "uuid",
  "model_version": "string",
  "latency_ms": 123.4,
  "prediction": {
    "client_name": "string|null",
    "client_address": "string|null",
    "seller_name": "string|null",
    "seller_address": "string|null",
    "invoice_number": "string|null",
    "invoice_date": "string|null"
  },
  "raw_text": "string",
  "confidence": {
    "available": false,
    "note": "confidence non calibrée par défaut pour ce pipeline"
  }
}
```

### Sortie erreur
```json
{
  "error": {
    "code": "INVALID_IMAGE|MODEL_NOT_FOUND|INFERENCE_FAILED|DATASET_UNAVAILABLE",
    "message": "message actionnable",
    "details": {}
  }
}
```

## 4) Critères d’acceptation techniques

1. **Dataset parsing**
- Lecture stable des CSV Kaggle (`File Name`, `Json Data`, `OCRed Text`).
- Validation structure et comptage des échantillons par split.

2. **Train**
- Exécution d’un run smoke complète sans crash.
- Sauvegarde artefacts dans `artifacts/doc_understanding/`.

3. **Évaluation**
- Génération de métriques minimales:
  - exact match global,
  - exact match par champ,
  - taux JSON valide.

4. **Inférence**
- Chargement du modèle fine-tuné final (pas modèle base).
- Retour conforme contrat I/O ci-dessus.

5. **Gradio**
- Upload image -> inférence -> affichage JSON + texte brut + latence.
- Gestion explicite des erreurs (modèle absent, image invalide, parsing invalide).

6. **QA**
- Rapport QA final avec cas testés, statut, anomalies, sévérité, recommandations.

## 5) Arbitrages structurants (corrections notebook)

### Fragilités observées dans le notebook source
- Chemins hardcodés Kaggle (`/kaggle/input/...`) non portables.
- Dépendance environnement Kaggle/Colab (install inline `%pip`/`!pip`).
- Couplage fort entre data prep, train, eval et inférence dans un seul flux.
- Parsing JSON de sortie fragile (`regex` simple).
- Aucune séparation claire des artefacts et de leur version.

### Arbitrages validés
1. Remplacer les chemins hardcodés par config (`data_root`, `artifacts_root`).
2. Extraire les étapes du notebook en modules Python dédiés.
3. Imposer un parser JSON robuste avec fallback contrôlé.
4. Ajouter un mode `--smoke` pour exécutions rapides et CI.
5. Standardiser la sauvegarde/chargement du modèle fine-tuné via `serialization.py`.
6. Forcer l’application Gradio à charger explicitement l’artefact final (`final_model/`), jamais le modèle de base.

## Hypothèses et limites
- Le fine-tuning complet Gemma 3n + Unsloth peut dépendre d’un environnement GPU Linux; sur Windows/CPU, exécution potentiellement limitée à smoke tests.
- Le contrat de sortie JSON suppose les 6 champs métier issus du notebook source; extension possible si besoin produit.
- La calibration de confiance n’est pas native ici; à traiter en amélioration ultérieure.
