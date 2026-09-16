<!--
Published: 2026-09-16 — https://github.com/ZM143-svg/FocalGen/releases/tag/v1.0
  Tag:             v1.0  (commit 9b4c240)
  Release title:   FocalGen v1.0: Dataset and Pretrained Models
  Attached assets: checkpoints.zip (704 MB), Real-world-dataset.zip (116 MB)
  This file is the source of truth for the "Describe this release" box —
  keep it in sync with the README section "数据集与模型权重" whenever assets change.
-->

**FocalGen v1.0** — official code, pretrained checkpoints and dataset release for
*A Foreground-Aware Robust Representation Learning Framework for UAV Crowd Localization*.

Headline results: DroneCrowd **54.57%** L-mAP (UAV-Dot baseline 51.00%), UP-COUNT **68.96%** (baseline 66.49%),
and **25.63%** on our self-collected UAV dataset without fine-tuning (baseline 16.37%).
Full comparison and ablation study → [README § Results](https://github.com/ZM143-svg/FocalGen#-实验结果).

## Downloads

| Archive | Size | Contents | SHA-256 |
| --- | --- | --- | --- |
| [`checkpoints.zip`](https://github.com/ZM143-svg/FocalGen/releases/download/v1.0/checkpoints.zip) | 704 MB | 5 FocalGen checkpoints (unzip → `*.ckpt`) | `87ab9136b92eb5115c9a701d358e55134e9041e5d423eaca7243ee5b841fd7bc` |
| [`Real-world-dataset.zip`](https://github.com/ZM143-svg/FocalGen/releases/download/v1.0/Real-world-dataset.zip) | 116 MB | 12 sequences / 128 images with raw annotations (YOLO + COCO) | `129862d83db6d15bed4e007e6926e4969c8f94d594bce6558ecd5f47494b56e6` |

Mirror — **Baidu Netdisk** (no extraction code required):
👉 https://pan.baidu.com/s/5iZAiqSyvR3WKk-KXpVgEdQ

```
Focalgen/
├── checkpoint/
│   └── checkpoints.zip          # 704.5 MB — 5 FocalGen checkpoints (unzip → *.ckpt)
└── Real-world-dataset.zip       # 115.6 MB — self-collected UAV dataset
```

## Manifest

| File | Size | Description |
| --- | --- | --- |
| `Real-world-dataset.zip` | 116 MB | Self-collected UAV dataset, **raw annotations**: `dataset1/` … `dataset12/`, 128 images (1920×1080) in total; each sequence has `images/`, `yolo_labels/` (normalized `x y w h`, class `Person head`) and `_annotations.coco.json`. Convert to the `merged_yolo_pixel` layout with [`data/prepare_real_world_dataset.py`](https://github.com/ZM143-svg/FocalGen/blob/main/data/prepare_real_world_dataset.py) before testing. |
| `FocalGen_DroneCrowd.ckpt` | 105 MB | FocalGen (FTA + SCSN + DySample), L-mAP **54.57%** on DroneCrowd |
| `FocalGen_Upcount.ckpt` | 105 MB | FocalGen (FTA + SCSN + DySample), L-mAP **68.96%** on UP-COUNT |
| `FocalGen_onlyFTA_DroneCrowd.ckpt` | 131 MB | Ablation — FTA only, L-mAP 52.58% on DroneCrowd |
| `FocalGen_onlySCSN_DroneCrowd.ckpt` | 105 MB | Ablation — SCSN only, L-mAP 52.57% on DroneCrowd |
| `FocalGen_onlyDysample_DroneCrowd.ckpt` | 316 MB | Ablation — DySample only, L-mAP 52.88% on DroneCrowd |
| `dot_pd_dronecrowd_51.00.ckpt` | 315 MB | UAV-Dot baseline, L-mAP **51.00%** on DroneCrowd |
| `dot_pd_upcount_66.49.ckpt` | 315 MB | UAV-Dot baseline, L-mAP **66.49%** on UP-COUNT |

> - All checkpoints are MiT-B2 + U-Net (≈27.5 M params); size differences come from the optimizer state stored inside the file.
> - The ablation checkpoints correspond to the single-module rows of the ablation table in the README.
> - The two UAV-Dot baseline checkpoints are **not** included in `checkpoints.zip` — contact the author if you need them.
> - `checkpoints.zip` unpacks to a `checkpoints/` folder containing exactly those five FocalGen `*.ckpt` files (verified).

## Dataset preparation

The dataset archive holds the **raw per-sequence annotations**, while the `merged_yolo_pixel` datamodule expects a flat
`images/` + `ground_truth/` layout with matching file names and one `x y` point pair per line. Convert it with the bundled script:

```bash
# recommended: reuse the exact ground_truth labels shipped in the repo (data/ground_truth), only rebuild the images
python data/prepare_real_world_dataset.py --zip Real-world-dataset.zip \
    --out /path/to/Real-world-dataset --gt-dir data/ground_truth
```

| Inside the zip | After conversion |
| --- | --- |
| `datasetX/images/frame_YYYYY.jpg` | `images/datasetX_frame_YYYYY.jpg` |
| `datasetX/yolo_labels/frame_YYYYY.txt` (normalized `cx cy w h`) | `ground_truth/datasetX_frame_YYYYY.txt` (`x y` = box centre, `floor(cx·W) floor(cy·H)`) |
| `datasetX/_annotations.coco.json` | only used for image size; kept in the zip for YOLO/COCO training |

> Without `--gt-dir`, the script recomputes the labels from the YOLO boxes: 98 of 128 files are byte-identical to the
> labels in the repository, the other 30 differ by 1 px on a single point each (the YOLO coordinates keep 6 decimals).
> Use `--gt-dir data/ground_truth` to reproduce the paper numbers exactly.

## Usage

```bash
# 1) Cross-scene generalization on the self-collected dataset (Results §3)
python main.py --config-name=dronecrowd \
    data_path=/path/to/Real-world-dataset \
    test_only=True debug=True \
    restore_from_ckpt=/path/to/FocalGen_DroneCrowd.ckpt

# 2) Ablation variants: swap restore_from_ckpt only
python main.py --config-name=dronecrowd \
    data_path=/path/to/DroneCrowd/test_data \
    test_only=True debug=True \
    restore_from_ckpt=/path/to/FocalGen_onlyFTA_DroneCrowd.ckpt

# 3) UP-COUNT
python main.py --config-name=upcount \
    data_path=/path/to/UP-COUNT/test_data \
    test_only=True debug=True \
    restore_from_ckpt=/path/to/FocalGen_Upcount.ckpt

# 4) Batch inference / export predicted points
python pt_pred.py --config-name=dronecrowd \
    data_path=/path/to/dataset \
    restore_from_ckpt=/path/to/FocalGen_DroneCrowd.ckpt
```

> Weights are loaded with `DotRegressor.load_from_checkpoint()`, so the module switches saved in the checkpoint
> (FTA / SCSN / DySample, `obj_threshold`) are restored automatically — you do **not** need to pass
> `use_dysample` / `use_csnorm` / `use_pd_attention` manually.

See the [README](https://github.com/ZM143-svg/FocalGen) for environment setup, data preparation and evaluation.
