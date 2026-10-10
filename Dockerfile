# OSMOSIS 提交包 —— 推理/评估镜像
#
#   docker build -t osmosis:latest .
#   docker run --rm --gpus all \
#       -v $PWD/weights:/app/weights \
#       -v <HTS_SEG_ROOT>:/data/hts-seg:ro \
#       osmosis:latest test_script.py --mode single
#
# 基础镜像选 CUDA 11.8 runtime（与 requirements.txt 里的 +cu118 轮子对齐），
# 驱动侧要求 >= 452.39；宿主有 nvidia-container-toolkit 即可 --gpus all。
#
# 注意：镜像**不含**数据集与权重（两者都远大于镜像层）。数据集与权重通过
# -v 挂载，路径在运行时用命令行覆盖（见下方 CMD 与 docs/02-复现步骤.md）。

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        build-essential git \
        libglib2.0-0 libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖单独一层，改代码不会让 pip 重装
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 核心代码 + 脚手架 + 文档（weights/ 与 outputs/ 大多数情况走挂载，见 README）
COPY *.py ./
COPY data ./data
COPY models ./models
COPY docs ./docs
COPY README.md README.en.md ./
COPY run_all.sh ./
RUN chmod +x run_all.sh

# 断言驱动可用：容器里 torch.cuda.is_available() 必须为 True
RUN python -c "import torch; print('torch', torch.__version__)"

# 默认冒烟：跑不依赖数据集的单元测试
CMD ["python", "verify.py"]
