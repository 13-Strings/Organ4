#!/usr/bin/env bash
# OSMOSIS 三阶段流水线一键运行（本提交包版本）。
#
#   bash run_all.sh              跑完整流程（Stage1 -> Stage2 -> Stage3 -> 评估）
#   bash run_all.sh smoke        小规模冒烟（每阶段 1 epoch，验证链路能否跑通）
#   bash run_all.sh verify       不训练的单元冒烟测试
#   bash run_all.sh test         直接复现测试集结果（不训练，用 weights/ 里的权重）
#   bash run_all.sh demo         最小 Demo（前 8 张测试图）
#
# 任何阶段被中断（Ctrl-C / kill / OOM / 机器抢占）后，重新执行同一条命令即可从
# 断点继续（RESUME='auto'）。
#
# ★输出根目录由 config.py 里"影响指标的参数"拼成★，所以用命令行覆盖过任何会进
# 目录名的参数后，三个阶段都要带同一组参数重跑，否则会在不同根目录下各跑各的、
# Stage2/3 找不到上游 encoder.pt（Stage3 会静默退化成随机初始化）。
# 纯运行期旋钮（--DEVICE_ID / --NUM_WORKERS / --EVAL_BATCH_SIZE）不进目录名，随便传。
#
# 解释器用 PY 环境变量覆盖，例如：
#   PY=/home/zhangc/miniconda3/envs/py310/bin/python bash run_all.sh test

set -euo pipefail

PY="${PY:-python}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_DIR"

MODE="${1:-full}"

echo "=============================================="
echo " OSMOSIS pipeline  mode=$MODE"
echo " python : $PY"
echo " code   : $CODE_DIR"
echo "=============================================="

if [ "$MODE" = "verify" ]; then
    exec "$PY" verify.py
fi

if [ "$MODE" = "test" ]; then
    exec "$PY" test_script.py "${@:2}"
fi

if [ "$MODE" = "demo" ]; then
    exec "$PY" run_demo.py "${@:2}"
fi

if [ "$MODE" = "smoke" ]; then
    # ★这些参数会影响输出根目录，三个阶段必须传完全相同的集合★
    SMOKE_ID=(--RUN_TAG _smoke --MAE_EPOCHS 1 --MAE_BATCH_SIZE 16
              --RXRX1_SUBSET_RATIO 0.005 --DOM_EPOCHS 1 --DOM_BATCH_SIZE 8
              --SEG_EPOCHS 1 --SEG_BATCH_SIZE 4)
    # 纯运行期旋钮，不进目录名
    SMOKE_RT=(--NUM_WORKERS 2 --LOG_EVERY_STEPS 5)

    echo ">>> [smoke] 单元冒烟"
    "$PY" verify.py

    echo ">>> [smoke] Stage1 1 epoch（只用 0.5% 数据）"
    "$PY" train_mae.py "${SMOKE_ID[@]}" "${SMOKE_RT[@]}"

    echo ">>> [smoke] Stage2 1 epoch（源/目标域见 config.py 的 DOM_SOURCE / DOM_TARGET）"
    "$PY" train_domain.py "${SMOKE_ID[@]}" "${SMOKE_RT[@]}"

    echo ">>> [smoke] Stage3 1 epoch"
    "$PY" train_seg.py "${SMOKE_ID[@]}" "${SMOKE_RT[@]}"

    echo ">>> [smoke] 汇总三阶段结果"
    "$PY" report.py "${SMOKE_ID[@]}"

    echo ">>> smoke 全部通过"
    exit 0
fi

# ---------------- 完整流程 ----------------
echo ">>> [准备] 预解码 RxRx1 缓存（只需做一次，约 35~50 分钟；可中断续做）"
if ls "${RXRX1_CACHE_DIR:-/home/zhangc/workspace/organ_cache/rxrx1}"/rxrx1_*.u8 >/dev/null 2>&1; then
    echo "    缓存已存在，跳过"
else
    "$PY" prepare_rxrx1_cache.py
fi

echo ">>> Stage1: RxRx1 MAE 自监督预训练"
"$PY" train_mae.py

echo ">>> Stage2: 域适应（源/目标域见 config.py 的 DOM_SOURCE / DOM_TARGET）"
"$PY" train_domain.py

echo ">>> Stage3: HTS-Seg 球体分割微调（训完自动评估测试集）"
"$PY" train_seg.py

echo ">>> 汇总三阶段结果与曲线"
"$PY" report.py

echo ">>> 单模型 + 概率集成两种口径的测试集结果"
"$PY" test_script.py

ROOT="$("$PY" -c 'import config; print(config.OUTPUT_ROOT)')"
echo ">>> 完成。结果在 $ROOT/"
echo "    汇总报告: $ROOT/summary/report.md"
