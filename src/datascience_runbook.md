# Data Science Runbook

## 1) Dataset structure expected

Place the local dataset here:

`data/scanned_images/`

Minimum tree:

`data/scanned_images/train/<class>/*.jpg|png`
`data/scanned_images/val/<class>/*.jpg|png`

Example:

`data/scanned_images/train/scanned/...`
`data/scanned_images/train/not_scanned/...`
`data/scanned_images/val/scanned/...`
`data/scanned_images/val/not_scanned/...`

## 2) Run the Python pipeline (outside notebook)

Quick smoke test:

```powershell
python src/scanned_images_pipeline.py --epochs 1 --batch-size 8 --img-size 128 --disable-pretrained
```

Standard run:

```powershell
python src/scanned_images_pipeline.py --epochs 8 --batch-size 16 --img-size 224
```

## 3) Synthetic fallback dataset

If `train/` + `val/` are missing, the pipeline auto-generates a synthetic dataset in:

`data/scanned_images/_synthetic/`

This validates the local training stack without requiring Kaggle resources.

## 4) Artifacts produced

The trained model is saved at:

`artifacts/scanned_images_resnet18.pt`
