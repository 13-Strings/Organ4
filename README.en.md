# OSMOSIS — Spheroid Segmentation on Organ-on-Chip Images

> 2026 Pazhou Algorithm Competition · International AI Competition · AI + Organ-on-Chip Track
> 中文版：[README.md](README.md)

Task: binary spheroid segmentation on the `cleaned_tiled_images` subset of HTS-Seg,
436 test images.
Method: three-stage transfer (RxRx1 MAE self-supervised pre-training → OoC bright-field
domain adaptation → HTS-Seg fine-tuning)

## 1. Test-Set Results

Table 1 Performance Comparison Results of Mainstream Segmentation Baseline Models and Multi-stage Transfer Learning Model

Note: Baselines refer to the classic segmentation model indicators reported in related studies on HTS-Seg; Bold indicates the optimal indicators, and underline indicates the suboptimal indicators. `osmosis-single` denotes a single model, while `osmosis-ensemble` represents the ensemble model integrating three individual models with random seeds 42, 43, and 44.

| Model            | Precision         | Recall            | Dice               | Jaccard(IoU)      | Accuracy          |
| ---------------- | ----------------- | ----------------- | ------------------ | ----------------- | ----------------- |
| unet             | 0.9289            | 0.9516            | 0.9356             | 0.8799            | 0.9797            |
| fcn              | **0.9692**  | 0.9025            | 0.9345             | 0.8804            | 0.9789            |
| deeplab          | 0.9408            | 0.9551            | 0.9532             | 0.9117            | 0.9828            |
| mask rcnn        | 0.9380            | <ins>0.9585</ins> | 0.9510             | 0.9080            | 0.9822            |
| yolo             | 0.9505            | 0.8890            | 0.9180             | 0.8520            | 0.9778            |
| osmosis-single   | <ins>0.9579</ins> | 0.9525            | **0.9855**   | <ins>0.9716</ins> | <ins>0.9853</ins> |
| osmosis-ensemble | 0.9540            | **0.9607**  | <ins>0.9729 </ins> | **0.9729**  | **0.9860**  |

Table 2 Performance Results of Ablation Experiments for Multi-stage Transfer Learning Model

Note: S1, S2, and S3 correspond to self-supervised pre-training on the 2D cell image RxRx1 dataset, domain adaptation on the OoC bright-field dataset, and chip-task fine-tuning on the HTS-Seg dataset; `osmosis-s2s3` is the ablation model without S1 pre-training, `osmosis-s1s3` is the ablation model without S2 domain adaptation, and `osmosis-s3` is the ablation model without the S1 and S2 stages. Bold indicates the optimal metric, and underline indicates the suboptimal metric.

| Model           | Precision         | Recall            | Dice              | Jaccard(IoU)      | Accuracy          |
| --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| osmosis‑s3     | <ins>0.9574</ins> | 0.9509            | 0.9852            | 0.9709            | 0.9850            |
| osmosis‑s2s3   | 0.9543            | <ins>0.9547</ins> | <ins>0.9853</ins> | <ins>0.9711</ins> | <ins>0.9851</ins> |
| osmosis-s1s3    | 0.9517            | **0.9573**  | <ins>0.9853</ins> | <ins>0.9711</ins> | 0.9850            |
| osmosis‑s1s2s3 | **0.9579**  | 0.9525            | **0.9855**  | **0.9716**  | **0.9853**  |

## 2. Data Preparation

### 2.1 Dataset download

OSMOSIS dataset download source:

```
https://drive.google.com/drive/folders/1x4hW7ufWyGJFHhqjgQCfF9HAsPwitjuM?usp=drive_link
```

After downloading, update the absolute dataset paths in `submission_v1/config.py`.
Relevant variables and their targets:

| Variable            | Points to                                         |
| ------------------- | ------------------------------------------------- |
| `HTS_ROOT`        | HTS-Seg dataset root                              |
| `RXRX1_ROOT`      | RxRx1 6-channel PNG root                          |
| `RXRX1_METADATA`  | RxRx1 metadata CSV                                |
| `OOC_ROOT`        | OoC bright-field dataset root                     |
| `RXRX1_CACHE_DIR` | Output directory of the RxRx1 pre-decoded cache   |
| `_ROOT_BASE`      | Output root for training and evaluation artifacts |

Any uppercase variable can also be overridden on the command line without editing the file:

```bash
python test_script.py --HTS_ROOT /your/path/to/hts-seg
```

### 2.2 Dataset sources

- **Baseline paper**: [https://www.sciencedirect.com/science/article/pii/S0169260725005577](https://www.sciencedirect.com/science/article/pii/S0169260725005577).
  This paper is the source of the HTS-Seg dataset.
- **HTS-Seg dataset**: [https://zenodo.org/records/14161434](https://zenodo.org/records/14161434)
- **RxRx1 dataset**: [https://rxrx.ai/datasets](https://rxrx.ai/datasets)
- **OoC bright-field dataset download website**: [https://zenodo.org/records/10203721](https://zenodo.org/records/10203721)

### 2.3 RxRx1 pre-decoded cache

The cache must be generated before Stage 1 training. Steps:

1. Set `RXRX1_CACHE_DIR` in `submission_v1/config.py` and confirm where the cache files go;
2. Run `python prepare_rxrx1_cache.py`.

The script decodes RxRx1 PNGs into `.u8` memory-mapped files under
`RXRX1_CACHE_DIR`, producing `{RXRX1_CACHE_DIR}/rxrx1_ch<ch>_s<size>.u8` plus a sibling
`.json` with metadata (`channels`, `size`, `n`, `root`, `bin`, `bytes`).
The process is resumable. This project uses `RXRX1_CHANNELS=[0,1,4]` and
`RXRX1_CACHE_SIZE=256`; the result is about 16 GB.

Directory layouts and requirements for the remaining datasets are in
[docs/01-输入输出格式说明.md](docs/01-输入输出格式说明.md).

## 3. Model Weights

### 3.1 Download and placement

Weight download source:

```
https://drive.google.com/drive/folders/1bIlAbDHVgP5hRKm5pfPNy_0PvovIhWeq?usp=drive_link
```

Download the files in the `weights` folder and place them in `submission_v1/weights`.
Six files:

| File                  | Size (bytes)  | Purpose                                                      |
| --------------------- | ------------- | ------------------------------------------------------------ |
| `s3_base.pt`        | 1 080 795 470 | Default checkpoint for single-model evaluation (deconv head) |
| `s3_best_seed42.pt` | 1 068 156 814 | Ensemble member, seed 42 (bilinear head)                     |
| `s3_best_seed43.pt` | 1 068 156 750 | Ensemble member, seed 43                                     |
| `s3_best_seed44.pt` | 1 068 156 750 | Ensemble member, seed 44                                     |
| `stage1_encoder.pt` | 345 784 518   | Stage 1 MAE pre-trained encoder                              |
| `stage2_best.pt`    | 1 361 084 704 | Stage 2 domain-adaptation model                              |

### 3.2 Evaluation commands

```bash
# Single-model evaluation, uses s3_base.pt by default
python test_script.py --mode single

# Ensemble evaluation, uses s3_best_seed42.pt / s3_best_seed43.pt / s3_best_seed44.pt by default
python test_script.py --mode ensemble

# Custom single-model evaluation
python test_script.py --single-ckpt <path>

# Custom ensemble evaluation
python test_script.py --ensemble-ckpts a.pt b.pt ...
```

`--mode single` and `--mode ensemble` can also be run together: `python test_script.py`.
Results are written to `outputs/test_results/`.

## 4. Environment

```bash
# pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# conda
conda env create -f environment.yml && conda activate osmosis

# docker
docker build -t osmosis:latest .
docker run --rm --gpus all osmosis:latest verify.py
```

## 5. Reproduction Steps

### 5.1 Training-free self-check

```bash
python verify.py
```

Checks model structure, data pipeline, metric implementation, and checkpoint round-trip
item by item; expected result is 6 passed, 0 failed.

### 5.2 Reproduce the test-set results

```bash
python test_script.py                                  # single + ensemble, evaluating the weights under weights/ by default
python test_script.py --mode single                    # single model only, ~3 min
python test_script.py --mode ensemble                  # ensemble + TTA + threshold tuning only, ~42 min
python test_script.py --single-ckpt <path>             # custom single-model checkpoint
python test_script.py --ensemble-ckpts a.pt b.pt ...   # custom ensemble checkpoints
python test_script.py --DEVICE_ID 3 --EVAL_BATCH_SIZE 8
```

Or use the finer-grained entry points:

```bash
python evaluate.py --ckpt weights/s3_base.pt
python evaluate.py --ckpt weights/s3_best_seed42.pt \
                            weights/s3_best_seed43.pt \
                            weights/s3_best_seed44.pt --tta --tune-threshold
python evaluate.py --ckpt weights/s3_base.pt --split val
```

Artifacts:

```
outputs/test_results/test_results.{json,md}   results of both protocols and per-image vectors
outputs/eval/eval_test_<tag>.json             evaluate.py results
```

### 5.3 Retraining Stage 2

Import the Stage-1 model weights:

```bash
# Option 1: start from this package's weights, placing stage1_encoder.pt in the Stage 1 directory
python - <<'PY'
import os, shutil
import config          # must precede importing torch
os.makedirs(config.DIR_STAGE1, exist_ok=True)
shutil.copy('weights/stage1_encoder.pt', os.path.join(config.DIR_STAGE1, 'encoder.pt'))
print('encoder.pt ->', config.DIR_STAGE1)
PY

# Option 2: train Stage 1 from scratch, then Stage 2
python prepare_rxrx1_cache.py     # generate the RxRx1 cache first, see §2.3
python train_mae.py               # Stage 1
```

Then train Stage 2:

```bash
python train_domain.py --DEVICE_ID 0
```

Artifacts: under `{DIR_STAGE2}/`, `ckpt/{last,best}.pt`, `encoder.pt`, `train_log.jsonl`,
`metrics.json`, `config_snapshot.py`. On completion the encoder from the best checkpoint is
re-exported to `encoder.pt` for Stage 3 to load.

### 5.4 Retraining Stage 3

Prepare the upstream encoder:

```bash
# Option 1: extract the Stage-2 encoder from this package's weights into the Stage 2 directory
python - <<'PY'
import os
import config          # must precede importing torch
import torch
st = torch.load('weights/stage2_best.pt', map_location='cpu', weights_only=False)
enc = {k[len('mae.encoder.'):]: v for k, v in st['model'].items()
       if k.startswith('mae.encoder.')}
os.makedirs(config.DIR_STAGE2, exist_ok=True)
torch.save(enc, os.path.join(config.DIR_STAGE2, 'encoder.pt'))
print('encoder.pt ->', config.DIR_STAGE2, len(enc), 'keys')
PY

# Option 2: retrain Stage 2 from scratch per §5.3; its artifacts already contain
#           {DIR_STAGE2}/encoder.pt
```

Train Stage 3:

```bash
python train_seg.py --DEVICE_ID 2 --EVAL_BATCH_SIZE 8 \
                    --SEG_BATCH_SIZE 4 --SEG_LR_ENCODER 5e-5 --SEG_LR_DECODER 5e-4
```

`train_seg.py` searches for the upstream encoder in this order
(`eval_seg.load_encoder_weights`, first existing file wins):

```
{DIR_STAGE2}/encoder.pt  →  {DIR_STAGE2}/ckpt/best.pt
→  {DIR_STAGE1}/encoder.pt  →  {DIR_STAGE1}/ckpt/best.pt
```

If none of the four exists the encoder stays randomly initialized. The log line
`载入编码器权重: <path> (missing=0, unexpected=0)` reports the path actually used and the key
matching result; confirm it before training.

Artifacts: under `{DIR_STAGE3}/`, `train_log.jsonl`, `metrics.json`, `config_snapshot.py`,
`split_meta.json`, `ckpt/{best,last}.pt`; with `AUTO_EVAL_AFTER_TRAIN` (True by default) the
test set is evaluated automatically at the end of training, adding `test_summary.json` plus
`eval/{test_metrics.json, table2.md, test_predictions.npz, figures/qualitative.png}`. The
`测试集结果` record at the end of `train_log.jsonl` holds those metrics.

Resuming: `RESUME='auto'`; rerunning the same command continues from
`{DIR_STAGE3}/ckpt/last.pt`, logging `已恢复断点 {path,epoch,step,best}`.

### 5.5 Full three-stage pipeline

```bash
bash run_all.sh          # Stage1 → Stage2 → Stage3 → report → test_script
bash run_all.sh smoke    # one epoch per stage, to validate the chain
bash run_all.sh verify   # self-check only
bash run_all.sh test     # reproduce test-set results only
```

Or run the individual scripts manually, in order:

```bash
python prepare_rxrx1_cache.py   # RxRx1 pre-decoded cache (Stage 1/2 both read it)
python train_mae.py             # Stage 1, produces {DIR_STAGE1}/encoder.pt
python train_domain.py          # Stage 2, produces {DIR_STAGE2}/encoder.pt
python train_seg.py             # Stage 3, automatically evaluates the test set at the end
python report.py                # aggregate the three stages into {OUTPUT_ROOT}/summary/report.md
python test_script.py --single-ckpt {DIR_STAGE3}/ckpt/best.pt   # re-evaluate a stage model on the test set
python evaluate.py --ckpt {DIR_STAGE3}/ckpt/best.pt             # re-evaluate a stage model on the validation set
```

Artifacts: the complete three-stage directory tree.

## 6. Repository Layout

```
submission_v1/
├── README.md / README.en.md     documentation (Chinese / English)
├── requirements.txt             Python dependencies
├── environment.yml              conda environment
├── Dockerfile                   CUDA 11.8 base image
├── .gitattributes               Git LFS rules for checkpoints and probability maps
│
├── test_script.py               reproduce test-set results (single / ensemble)
├── evaluate.py                  general evaluation entry point
├── inference.py                 inference, writes masks
├── run_demo.py                  minimal example (first N images + 3-panel figure)
├── run_all.sh                   three-stage training orchestrator
│
├── config.py                    all hyper-parameters
├── train_mae.py                 Stage 1: RxRx1 MAE self-supervised pre-training
├── train_domain.py              Stage 2: OoC domain adaptation (reconstruction + GRL)
├── train_seg.py                 Stage 3: HTS-Seg fine-tuning (+ auto evaluation)
├── eval_seg.py                  single-model test evaluation
├── eval_tta.py                  TTA + validation-set threshold tuning
├── eval_ensemble.py             multi-seed probability ensemble
├── reeval.py                    re-evaluate past runs under the current metric code
├── metrics.py                   region metrics (consistent with baseline metri())
├── boundary.py                  boundary metrics (ASSD/HD95/bIoU) and boundary losses
├── utils.py                     logging / checkpoints / loaders / CLI overrides
├── verify.py                    training-free unit self-check
├── report.py / compare_runs.py / final_summary.py / scan_runs.py   reporting
├── sig_test.py                  per-image paired significance tests
├── analyze_gt.py / analyze_components.py   dataset and error-structure analysis
├── hts.py                       stale copy of data/hts.py, not referenced by any module
├── run_cond_sig.py              paired-test driver for the condition split
├── probe_memory.py              VRAM probe
├── prepare_rxrx1_cache.py       RxRx1 pre-decoded cache
├── models/{vit,mae,heads}.py    ViT / MAE / segmentation head
├── data/{hts,ooc,rxrx1,ooc_pretrain}.py   per-stage datasets
│
├── SHA256SUMS                   weight checksums
├── weights/                     6 checkpoints
├── outputs/                     reproduction artifacts
└── docs/                        documentation (Chinese)
    ├── 01-输入输出格式说明.md
    ├── 02-复现步骤.md
    ├── 03-数据集与权重清单.md
    ├── 04-评估协议.md
    └── 05-对代码的改动清单.md
```

`outputs/` contents:

```
outputs/
├── organ_v5/<split-tag>/
│   ├── _cache/filelists/*.json              directory listing cache
│   ├── summary/ensemble.json                3-seed ensemble result
│   ├── summary/ens_probs/*.npy              per-model and ensemble probability maps
│   └── s1-…/s2-…/s3-…/                      complete output tree of one Stage-3 run
├── test_results/                            live test_script.py results
├── eval/                                    live evaluate.py results
└── demo/                                    run_demo.py output
```

Output directory structure, the `train_log.jsonl` record format, and the fields of each JSON
artifact are documented in [docs/01-输入输出格式说明.md](docs/01-输入输出格式说明.md) (Chinese).

## 7. Competition Deliverables

| Requirement                                     | Location                                                                                                                                  |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Core code                                       | root`*.py`, `models/`, `data/`                                                                                                      |
| README                                          | `README.md` (Chinese), `README.en.md` (English)                                                                                       |
| requirements.txt / environment.yml / Dockerfile | all three provided                                                                                                                        |
| Model files or download instructions            | `weights/` (6 checkpoints) + download link in §3.1; per-file SHA-256 in [docs/03](docs/03-数据集与权重清单.md)                          |
| Test / inference / evaluation / demo scripts    | `test_script.py`, `inference.py`, `evaluate.py`, `run_demo.py`, all exposing `main()`; training orchestrated by `run_all.sh`  |
| Input/output format specification               | [docs/01-输入输出格式说明.md](docs/01-输入输出格式说明.md)                                                                                 |
| Reproduction steps                              | §5 and[docs/02-复现步骤.md](docs/02-复现步骤.md)                                                                                          |
| API / cloud / commercial-model disclosure       | None used. Inference runs locally on a single GPU with no network and no keys. External dependencies are three public datasets, see §2.2 |

## 8. Method

### 8.1 Three-stage transfer

```
RxRx1 (6-channel fluorescence, channels 0/1/4 used)
   │  Stage 1: MAE self-supervised pre-training (75% masking, ViT-B/16, dim 768 × 12)
   ▼
ViT encoder
   │  Stage 2: domain adaptation (reconstruction loss + GRL domain loss,
   │           source RxRx1 → target OoC bright-field)
   ▼
Domain-adapted encoder
   │  Stage 3: HTS-Seg fine-tuning (BCE + soft Dice)
   ▼
Spheroid segmentation model (512×512 input, separable-upsampling head)
```

### 8.2 Inference-time augmentation

| Technique                   | How                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 8-way TTA                   | 4 × 90° rotations × {no flip, horizontal flip}; the 8 probability maps are averaged, then inverse-transformed |
| 3-seed ensemble             | Each seed checkpoint produces a full probability map; maps are averaged pixel-wise before thresholding           |
| Validation threshold tuning | Sweep 0.30–0.70 (step 0.02) on the 175 validation images by bIoU@2 → 0.48                                      |

The threshold is selected on the validation set only; the test set is used for reporting.

## 9. References and Data Sources

- Baseline paper (source of the HTS-Seg dataset):
  [https://www.sciencedirect.com/science/article/pii/S0169260725005577](https://www.sciencedirect.com/science/article/pii/S0169260725005577)
- HTS-Seg dataset: [https://zenodo.org/records/14161434](https://zenodo.org/records/14161434); the official baseline code is the
  `HTS_Seg-main.zip` distributed alongside the dataset
- RxRx1 dataset: the resources of the paper *RxRx1: A Dataset for Evaluating Experimental
  Batch Correction Methods* ([https://rxrx.ai/datasets](https://rxrx.ai/datasets)),
  [https://arxiv.org/abs/2301.05768](https://arxiv.org/abs/2301.05768)
- OoC bright-field dataset: [https://zenodo.org/records/10203721](https://zenodo.org/records/10203721)
- Competition dataset download entry point:
  [https://drive.google.com/drive/folders/1x4hW7ufWyGJFHhqjgQCfF9HAsPwitjuM?usp=drive_link](https://drive.google.com/drive/folders/1x4hW7ufWyGJFHhqjgQCfF9HAsPwitjuM?usp=drive_link)
