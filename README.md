# OSMOSIS — 器官芯片球体分割任务

> 2026 琶洲算法大赛 · 国际AI赛 · AI+器官芯片赛道

任务：HTS-Seg 数据集 `cleaned_tiled_images` 子集上的球体二值分割，测试集 436 张。
方法：三步迁移（RxRx1 MAE 自监督预训练 → OoC 明场域适应 → HTS-Seg 微调）

## 一、测试集结果

### 表 1 主流分割基线模型与OSMOSIS模型性能对比结果

注：基线模型为 HTS-Seg 相关研究报道的经典分割模型指标；**粗体**代表最优指标，<ins>下划线</ins>代表次优指标。`osmosis-single`为单模型，`osmosis-ensemble`是融合随机种子 42、43、44 对应的 3 个独立模型构建的集成模型。

| Model            | Precision         | Recall            | Dice               | Jaccard(IoU)      | Accuracy          |
| ---------------- | ----------------- | ----------------- | ------------------ | ----------------- | ----------------- |
| unet             | 0.9289            | 0.9516            | 0.9356             | 0.8799            | 0.9797            |
| fcn              | **0.9692**  | 0.9025            | 0.9345             | 0.8804            | 0.9789            |
| deeplab          | 0.9408            | 0.9551            | 0.9532             | 0.9117            | 0.9828            |
| mask rcnn        | 0.9380            | <ins>0.9585</ins> | 0.9510             | 0.9080            | 0.9822            |
| yolo             | 0.9505            | 0.8890            | 0.9180             | 0.8520            | 0.9778            |
| osmosis-single   | <ins>0.9579</ins> | 0.9525            | **0.9855**   | <ins>0.9716</ins> | <ins>0.9853</ins> |
| osmosis-ensemble | 0.9540            | **0.9607**  | <ins>0.9729 </ins> | **0.9729**  | **0.9860**  |

### 表 2 OSMOSIS模型消融实验性能结果

注：S1、S2、S3 分别对应在二维细胞图像 RxRx1 数据集上的自监督预训练、在 OoC 明场数据集上的域自适应，以及在 HTS-Seg 数据集上的芯片任务微调；`osmosis-s2s3`为移除 S1 预训练模块的消融模型，`osmosis-s1s3`为移除 S2 域自适应模块的消融模型，`osmosis-s3`为同时移除 S1、S2 阶段的消融模型。**粗体**代表最优指标，<ins>下划线</ins>代表次优指标。

| Model           | Precision         | Recall            | Dice              | Jaccard(IoU)      | Accuracy          |
| --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| osmosis‑s3     | <ins>0.9574</ins> | 0.9509            | 0.9852            | 0.9709            | 0.9850            |
| osmosis‑s2s3   | 0.9543            | <ins>0.9547</ins> | <ins>0.9853</ins> | <ins>0.9711</ins> | <ins>0.9851</ins> |
| osmosis-s1s3    | 0.9517            | **0.9573**  | <ins>0.9853</ins> | <ins>0.9711</ins> | 0.9850            |
| osmosis‑s1s2s3 | **0.9579**  | 0.9525            | **0.9855**  | **0.9716**  | **0.9853**  |

## 二、数据准备

### 2.1 数据集下载

OSMOSIS数据集下载来源：

```
https://drive.google.com/drive/folders/1x4hW7ufWyGJFHhqjgQCfF9HAsPwitjuM?usp=drive_link
```

下载后需要修改 `submission_v1/config.py` 中的数据集绝对路径。相关变量与默认值：

| 变量                | 指向                       |
| ------------------- | -------------------------- |
| `HTS_ROOT`        | HTS-Seg 数据集根目录       |
| `RXRX1_ROOT`      | RxRx1 六通道 PNG 根目录    |
| `RXRX1_METADATA`  | RxRx1 元数据 CSV           |
| `OOC_ROOT`        | OoC 明场数据集根目录       |
| `RXRX1_CACHE_DIR` | RxRx1 预解码缓存的输出目录 |
| `_ROOT_BASE`      | 训练与评估产物的输出根目录 |

也可以在命令行覆盖任意大写变量，不必改文件：

```bash
python test_script.py --HTS_ROOT /your/path/to/hts-seg
```

### 2.2 数据集来源

- **baseline 论文**：[https://www.sciencedirect.com/science/article/pii/S0169260725005577](https://www.sciencedirect.com/science/article/pii/S0169260725005577)。
  该论文是 HTS-Seg 数据集的来源。
- **HTS-Seg 数据集**：[https://zenodo.org/records/14161434](https://zenodo.org/records/14161434)
- **RxRx1 数据集**：[https://rxrx.ai/datasets](https://rxrx.ai/datasets)
- **OoC 明场数据集下载网站**：[https://zenodo.org/records/10203721](https://zenodo.org/records/10203721)

### 2.3 RxRx1 预解码缓存

Stage 1 训练前需要先生成 RxRx1 缓存。步骤：

1. 设置 `submission_v1/config.py` 的 `RXRX1_CACHE_DIR`，确认缓存文件的输出位置；
2. 执行 `python prepare_rxrx1_cache.py`。

脚本会把 RxRx1 的 PNG 解码为 `.u8` 内存映射文件，产物为
`{RXRX1_CACHE_DIR}/rxrx1_ch<ch>_s<size>.u8` 与同名 `.json` 元数据（记录 `channels`、`size`、`n`、`root`、`bin`、`bytes`）。
缓存可中断续做。本工程使用 `RXRX1_CHANNELS=[0,1,4]`、`RXRX1_CACHE_SIZE=256`，
产物约 16GB。

其余数据集目录布局与要求见 [docs/01-输入输出格式说明.md](docs/01-输入输出格式说明.md)。

## 三、模型权重

### 3.1 下载与放置

模型权重下载来源：

```
https://drive.google.com/drive/folders/1bIlAbDHVgP5hRKm5pfPNy_0PvovIhWeq?usp=drive_link
```

下载 `weights` 文件夹里的文件，放置到 `submission_v1/weights` 目录下。共 6 个文件：

| 文件                  | 大小 (bytes)  | 用途                             |
| --------------------- | ------------- | -------------------------------- |
| `s3_base.pt`        | 1 080 795 470 | 单模型评估默认权重（deconv 头）  |
| `s3_best_seed42.pt` | 1 068 156 814 | 集成模型，种子 42（bilinear 头） |
| `s3_best_seed43.pt` | 1 068 156 750 | 集成模型，种子 43                |
| `s3_best_seed44.pt` | 1 068 156 750 | 集成模型，种子 44                |
| `stage1_encoder.pt` | 345 784 518   | Stage 1 MAE 预训练编码器         |
| `stage2_best.pt`    | 1 361 084 704 | Stage 2 域适应完整模型           |

### 3.2 评估命令

```bash
# 单模型评估，默认使用 s3_base.pt
python test_script.py --mode single

# 集成模型评估，默认使用 s3_best_seed42.pt / s3_best_seed43.pt / s3_best_seed44.pt
python test_script.py --mode ensemble

# 自定义单模型评估
python test_script.py --single-ckpt <path>

# 自定义集成模型评估
python test_script.py --ensemble-ckpts a.pt b.pt ...
```

`--mode single` 与 `--mode ensemble` 也可以同时执行：`python test_script.py`。
结果写入 `outputs/test_results/`。

## 四、环境

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

## 五、复现步骤

### 5.1 不训练的自检

```bash
python verify.py
```

逐项检查模型结构、数据管线、指标实现、断点读写，应有 6 项通过、0 项失败。

### 5.2 复现测试集结果

```bash
python test_script.py                                  # 单模型 + 集成，默认评测 weights/ 下的权重文件
python test_script.py --mode single                    # 仅单模型，约 3 分钟
python test_script.py --mode ensemble                  # 仅集成 + TTA + 调阈，约 42 分钟
python test_script.py --single-ckpt <path>             # 自定义单模型权重
python test_script.py --ensemble-ckpts a.pt b.pt ...   # 自定义集成权重
python test_script.py --DEVICE_ID 3 --EVAL_BATCH_SIZE 8
```

或使用细粒度入口：

```bash
python evaluate.py --ckpt weights/s3_base.pt
python evaluate.py --ckpt weights/s3_best_seed42.pt \
                            weights/s3_best_seed43.pt \
                            weights/s3_best_seed44.pt --tta --tune-threshold
python evaluate.py --ckpt weights/s3_base.pt --split val
```

产物：

```
outputs/test_results/test_results.{json,md}   两种口径的结果与逐图向量
outputs/eval/eval_test_<tag>.json             evaluate.py 的结果
```

### 5.3 重训 Stage 2

导入 Stage1模型权重：

```bash
# 方式一：从本包权重出发，把 stage1_encoder.pt 放到 Stage 1 目录
python - <<'PY'
import os, shutil
import config          # 必须先于 torch 导入
os.makedirs(config.DIR_STAGE1, exist_ok=True)
shutil.copy('weights/stage1_encoder.pt', os.path.join(config.DIR_STAGE1, 'encoder.pt'))
print('encoder.pt ->', config.DIR_STAGE1)
PY

# 方式二：从零训练 Stage 1，再训 Stage 2
python prepare_rxrx1_cache.py     # 先生成 RxRx1 缓存，见 §2.3
python train_mae.py               # Stage 1
```

然后训练 Stage 2：

```bash
python train_domain.py --DEVICE_ID 0
```

产物：`{DIR_STAGE2}/` 下的`ckpt/{last,best}.pt`、`encoder.pt`、`train_log.jsonl`、`metrics.json`、`config_snapshot.py`。训练结束会把最优权重里的编码器重新导出为`encoder.pt`，供 Stage 3 加载。

### 5.4 重训 Stage 3

准备上游编码器：

```bash
# 方式一：从本包权重抽出 Stage 2 编码器，写进 Stage 2 目录
python - <<'PY'
import os
import config          # 必须先于 torch 导入
import torch
st = torch.load('weights/stage2_best.pt', map_location='cpu', weights_only=False)
enc = {k[len('mae.encoder.'):]: v for k, v in st['model'].items()
       if k.startswith('mae.encoder.')}
os.makedirs(config.DIR_STAGE2, exist_ok=True)
torch.save(enc, os.path.join(config.DIR_STAGE2, 'encoder.pt'))
print('encoder.pt ->', config.DIR_STAGE2, len(enc), 'keys')
PY

# 方式二：按 §5.3 从零重训 Stage 2，其产物里已含 {DIR_STAGE2}/encoder.pt
```

训练 Stage 3：

```bash
python train_seg.py --DEVICE_ID 2 --EVAL_BATCH_SIZE 8 \
                    --SEG_BATCH_SIZE 4 --SEG_LR_ENCODER 5e-5 --SEG_LR_DECODER 5e-4
```

上游编码器的查找顺序（`eval_seg.load_encoder_weights`，取第一个存在的）：

```
{DIR_STAGE2}/encoder.pt  →  {DIR_STAGE2}/ckpt/best.pt →  {DIR_STAGE1}/encoder.pt  →  {DIR_STAGE1}/ckpt/best.pt
```

四者均不存在时编码器保持随机初始化。日志中的`载入编码器权重: <path> (missing=0, unexpected=0)` 一行给出实际加载路径与键匹配情况，训练开始前应确认该行。

产物：`{DIR_STAGE3}/` 下的`train_log.jsonl`、`metrics.json`、`config_snapshot.py`、`split_meta.json`、`ckpt/{best,last}.pt`；`AUTO_EVAL_AFTER_TRAIN`（默认 True）在训练收尾自动评估测试集，额外产出 `test_summary.json` 与`eval/{test_metrics.json, table2.md, test_predictions.npz, figures/qualitative.png}`，`train_log.jsonl` 末尾的 `测试集结果` 记录即这批指标。

断点续训：`RESUME='auto'`，重跑同一命令即从 `{DIR_STAGE3}/ckpt/last.pt` 继续，日志中出现 `已恢复断点 {path,epoch,step,best}`。

### 5.5 完整三阶段

```bash
bash run_all.sh          # Stage1 → Stage2 → Stage3 → 汇总 → test_script
bash run_all.sh smoke    # 每阶段 1 epoch，验证链路
bash run_all.sh verify   # 仅自检
bash run_all.sh test     # 仅复现测试集结果
```

或按顺序手动执行单个脚本：

```bash
python prepare_rxrx1_cache.py   # RxRx1 预解码缓存（Stage 1/2 都要读）
python train_mae.py             # Stage 1，产出 {DIR_STAGE1}/encoder.pt
python train_domain.py          # Stage 2，产出 {DIR_STAGE2}/encoder.pt
python train_seg.py             # Stage 3，训完自动评估测试集
python report.py                # 汇总三个阶段到 {OUTPUT_ROOT}/summary/report.md
python test_script.py --single-ckpt {DIR_STAGE3}/ckpt/best.pt   # 单独复评阶段模型的测试集结果
python evaluate.py --ckpt {DIR_STAGE3}/ckpt/best.pt             # 单独复评阶段模型的验证集结果
```

产物：完整三阶段目录树。

## 六、仓库结构

```
submission_v1/
├── README.md / README.en.md     说明文档（中文 / 英文）
├── requirements.txt             Python 依赖
├── environment.yml              conda 环境
├── Dockerfile                   CUDA 11.8 基础镜像
├── .gitattributes               权重与概率图的 Git LFS 规则
│
├── test_script.py               复现测试集结果（单模型 / 集成）
├── evaluate.py                  通用评估入口
├── inference.py                 图像推理，输出掩码
├── run_demo.py                  最小示例（前 N 张 + 三联图）
├── run_all.sh                   三阶段训练编排
│
├── config.py                    全部超参
├── train_mae.py                 Stage 1：RxRx1 MAE 自监督预训练
├── train_domain.py              Stage 2：OoC 域适应（重建 + GRL 域判别）
├── train_seg.py                 Stage 3：HTS-Seg 分割微调（训练结束自动评估）
├── eval_seg.py                  单模型测试集评估
├── eval_tta.py                  TTA + 验证集阈值调优
├── eval_ensemble.py             多种子概率集成
├── reeval.py                    按当前指标口径重评历史 run
├── metrics.py                   区域指标（与 baseline metri() 一致）
├── boundary.py                  边界指标（ASSD/HD95/bIoU）与边界损失
├── utils.py                     日志 / 断点 / 数据加载 / 命令行覆盖
├── verify.py                    不训练的单元自检
├── report.py / compare_runs.py / final_summary.py / scan_runs.py   结果汇总与分析
├── sig_test.py                  逐图配对显著性检验
├── analyze_gt.py / analyze_components.py   数据集与误差结构分析
├── hts.py                       data/hts.py 的旧副本，未被任何模块引用
├── run_cond_sig.py              跨条件划分的配对检验驱动脚本
├── probe_memory.py              显存探针
├── prepare_rxrx1_cache.py       RxRx1 预解码缓存
├── models/{vit,mae,heads}.py    ViT / MAE / 分割头
├── data/{hts,ooc,rxrx1,ooc_pretrain}.py   各阶段数据集
│
├── SHA256SUMS                   权重校验值
├── weights/                     6 个权重
├── outputs/                     复现产物（见下）
└── docs/
    ├── 01-输入输出格式说明.md
    ├── 02-复现步骤.md
    ├── 03-数据集与权重清单.md
    ├── 04-评估协议.md
    └── 05-对代码的改动清单.md
```

`outputs/` 内容：

```
outputs/
├── organ_v5/<split-tag>/
│   ├── _cache/filelists/*.json              目录清单缓存
│   ├── summary/ensemble.json                3 种子集成结果
│   ├── summary/ens_probs/*.npy              逐模型与集成的概率图
│   └── s1-…/s2-…/s3-…/                      一次 Stage-3 运行的完整目录树
├── test_results/                            test_script.py 的实跑结果
├── eval/                                    evaluate.py 的实跑结果
└── demo/                                    run_demo.py 的输出
```

输出目录结构、`train_log.jsonl` 的记录格式、各 JSON 产物字段见
[docs/01-输入输出格式说明.md](docs/01-输入输出格式说明.md)。

## 七、赛事交付物对照

| 赛事要求                                        | 本包对应文件                                                                                                          |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 核心代码                                        | 根目录`*.py`、`models/`、`data/`                                                                                |
| README.md                                       | `README.md`（中文）、`README.en.md`（英文）                                                                       |
| requirements.txt / environment.yml / Dockerfile | 三者齐备                                                                                                              |
| 模型文件或公开下载说明                          | `weights/`（6 个权重）+ 下载地址见 §3.1；逐文件 SHA-256 见 [docs/03](docs/03-数据集与权重清单.md)                   |
| 测试脚本 / 推理脚本 / 评估脚本 / Demo 启动脚本  | `test_script.py` / `inference.py` / `evaluate.py` / `run_demo.py`，均含 `main()`；训练编排为 `run_all.sh` |
| 输入输出格式说明                                | [docs/01-输入输出格式说明.md](docs/01-输入输出格式说明.md)                                                             |
| 结果复现步骤                                    | §五 与[docs/02-复现步骤.md](docs/02-复现步骤.md)                                                                      |
| API / 云服务 / 商业模型说明                     | 未使用任何 API、云服务或商业模型，推理在本地单卡完成，不需要网络与密钥。外部依赖为三个公开数据集，见 §2.2            |

## 八、方法

### 8.1 三步迁移

```
RxRx1（六通道荧光，取通道 0/1/4）
   │  Stage 1：MAE 自监督预训练（mask 75%，ViT-B/16，dim 768 × 12 层）
   ▼
ViT 编码器
   │  Stage 2：域适应（重建损失 + GRL 域判别损失，源域 RxRx1 → 目标域 OoC 明场）
   ▼
域适配编码器
   │  Stage 3：HTS-Seg 微调（BCE + soft Dice）
   ▼
球体分割模型（512×512 输入，可分离卷积上采样分割头）
```

### 8.2 推理端增强

| 方法           | 做法                                                                   |
| -------------- | ---------------------------------------------------------------------- |
| 8 向 TTA       | 4 个 90° 旋转 × {不翻转, 水平翻转}，8 张概率图取算术平均后逆变换还原 |
| 3 种子概率集成 | 三个训练种子的权重各自输出整幅概率图，逐像素算术平均后再阈值化         |
| 验证集阈值调优 | 在 175 张验证集上按 bIoU@2 扫描 0.30–0.70（步长 0.02），选出 0.48     |

阈值只在验证集上选择，测试集只用于报告。

## 九、参考文献与数据来源

- baseline 论文（HTS-Seg 数据集来源）：
  [https://www.sciencedirect.com/science/article/pii/S0169260725005577](https://www.sciencedirect.com/science/article/pii/S0169260725005577)
- HTS-Seg 数据集：[https://zenodo.org/records/14161434](https://zenodo.org/records/14161434)；baseline 官方代码为随数据集
  分发的 `HTS_Seg-main.zip`
- RxRx1 数据集：论文 *RxRx1: A Dataset for Evaluating Experimental Batch Correction
  Methods* （[https://rxrx.ai/datasets](https://rxrx.ai/datasets)）的论文资源 [https://arxiv.org/abs/2301.05768](https://arxiv.org/abs/2301.05768)
- OoC 明场数据集：[https://zenodo.org/records/10203721](https://zenodo.org/records/10203721)
- 赛题数据集下载入口：[https://drive.google.com/drive/folders/1x4hW7ufWyGJFHhqjgQCfF9HAsPwitjuM?usp=drive_link](https://drive.google.com/drive/folders/1x4hW7ufWyGJFHhqjgQCfF9HAsPwitjuM?usp=drive_link)
