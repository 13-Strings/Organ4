"""
流水线:
    Stage 1  RxRx1 (125510 site, 6通道荧光) --MAE自监督-->  ViT 通用细胞形态表征
    Stage 2  源域/目标域                 --域判别损失+重建损失-->  域适配编码器
    Stage 3  HTS-Seg (2971 tile, 512)    --任务微调-->  球体分割

"""


import os

# ==========================================================================
# A. 全局 / 环境
# ==========================================================================

# 全局随机种子。Python hash / numpy / torch / cuDNN / DataLoader 全部由它派生
SEED = 42

# 使用的物理 GPU 编号（nvidia-smi 里的 index）。只占一张卡。
# 本实例只有一张 4090D(index=0),故固定 0。
DEVICE_ID = 0

# --------------------------------------------------------------------------
# 【必须在 import torch 前设置的环境变量】
# --------------------------------------------------------------------------
os.environ['PYTHONHASHSEED'] = str(SEED)              # 固定 Python 哈希种子
os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICE_ID)   # 只暴露 DEVICE_ID 这一张卡
# CUDA 同步模式。'1' = 每个 kernel 启动都阻塞等待完成 —— 报错定位准,但 CPU 与 GPU
# 无法流水线重叠,实测 GPU 利用率只有 ~69%(GPU 干等 CPU 派发)。
# '0' = 异步(默认行为)。数学结果不受影响:确定性由下面 DETERMINISTIC +
# torch.use_deterministic_algorithms 保证,与本开关无关。
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'     # CUDA 10.2+ 矩阵乘确定性配置

# CUDA_VISIBLE_DEVICES 已把目标卡重映射为 0，故代码里统一用 cuda:0
DEVICE = 'cuda:0'

# True = 强制全确定性。除了 cuDNN/cuBLAS，还会关掉 SDPA 的 flash 与
# mem-efficient 后端（它们的反向是非确定的），注意力改走 math 后端 ——
# 代价是更慢、更吃显存，且 N*N 的大块连续分配对显存碎片敏感，
# 会让可用的 batch size 上限变低。False 则放开速度与显存优先。
DETERMINISTIC = True

# DataLoader 进程数。0 = 最严格单进程确定性；
# >0 时配合 seed_worker + generator 仍然完全可复现。
#
# 【该值按容器内存上限推算，不按 CPU 核数设定】
#   本容器 cgroup 内存上限 32GB;而每个 worker 的 RSS 约 1.5~1.9GB
#   (torch + CUDA 库 + 数据集对象各占一份)。
#   更关键的是 utils.build_loader 开了 persistent_workers=True,**训练集与验证集
#   各自常驻一批 worker** —— 所以实际进程数 ≈ 2 x NUM_WORKERS。
#   实测把 NUM_WORKERS 设为 8 时是 16 个 worker、合计约 30GB,只剩 2GB 余量,
#   一旦页面缓存冲高就被 cgroup OOM killer 杀进程(oom_kill 计数曾涨到 13,
#   表现为"队列每隔十几分钟静默死亡",排查了很久)。
#   4 -> 8 个 worker 约 14GB,留出足够余量;Stage3 的瓶颈在 GPU 计算,不在取数,
#   所以降到 4 对速度几乎没有影响。
NUM_WORKERS = 4

# --------------------------------------------------------------------------
# 主输出根目录 OUTPUT_ROOT 与三个阶段目录名，由本文件**所有**影响最终指标的
# 参数（DOM_* / MAE_* / SEG_* …）拼成，所以统一放在文件末尾的「第 I 节 输出
# 目录」里定义 —— Python 自上而下执行，这个位置还拿不到下面才定义的参数。
# 改任何参数都不用动它，目录名会自动跟着变。
# --------------------------------------------------------------------------

# 附加到「全局层」目录名后面的标签（分层见第 I 节）。非空时（如 '_smoke'）
# 会把整棵输出树挪到另一个全局层目录下，冒烟测试的输出和正式实验彻底分开，
# 避免冒烟产生的小断点被正式训练当成断点续训的起点
RUN_TAG = ''

# 日志/断点频率
LOG_EVERY_STEPS = 20      # 每多少 step 往 stdout + train_log.jsonl 写一次
SAVE_EVERY_EPOCH = True   # 每 epoch 结束保存 last.pt（断点续训用）
SAVE_EVERY_STEPS = 1200   # 每 N step 刷新一次 last.pt(≈15 分钟 @0.757s/step)。
                          #   域适应每轮 65 分钟、本机又常被外部整树杀(一天 7 次),
                          #   逐轮保存=一次失手白扔一小时;周期刷新后最多损失 ~15 分钟。
KEEP_LAST_N = 3           # 保留最近 N 个 step 快照，超出自动清理
# 'auto'=存在 last.pt 就续训 | 'none'=强制从头训 | '<路径>'=指定断点文件
RESUME = 'auto'


# ==========================================================================
# B. 数据路径
# ==========================================================================

# ---- Stage 1 / Stage 2 源域: RxRx1 (6通道荧光, 512x512, uint8 png) ----
RXRX1_ROOT = '/bigdat2/user/zhangc/organ/datasets/rxrx1/rxrx1-images/images'
RXRX1_METADATA = '/bigdat2/user/zhangc/organ/datasets/rxrx1/rxrx1-metadata/metadata.csv'
# 取哪 3 个通道合成网络输入。RxRx1 通道语义:
# w1=Hoechst(核) w2=ConA(内质网) w3=Syto14(RNA) w4=WGA(高尔基/膜) w5=Phalloidin(肌动蛋白) w6=MitoTracker(线粒体)
# 默认 (核, 内质网, 肌动蛋白) —— 形态学信息最互补的三通道
RXRX1_CHANNELS = [0, 1, 4]
# 只用 metadata 里 dataset=='train' 的 site(81224)；False 则用全部 125510
RXRX1_TRAIN_SPLIT_ONLY = True
# 用多少比例的 RxRx1（1.0=全量）。调小便于快速调试
RXRX1_SUBSET_RATIO = 1.0

# ---- RxRx1 预解码缓存 ----
# 本机 NFS 单文件延迟极高（实测每个 PNG 约 0.5 秒，单进程只有 0.7 img/s），
# 直接读原图会让 GPU 饿死。先跑 prepare_rxrx1_cache.py 把像素顺序写成一个
# 大文件，之后训练走内存映射，速度提升一到两个数量级。
RXRX1_CACHE = True        # True=存在缓存就用（没有则回落到直接读原图）
# 必须放在本地盘！实测把 memmap 写到 NFS(/bigdat2) 会把预解码从 37.9 site/s
# 拖到 9.9 site/s（写回阻塞），写本地盘 /data 则零开销
RXRX1_CACHE_DIR = '/home/zhangc/workspace/organ_cache/rxrx1'
RXRX1_CACHE_SIZE = 256    # 缓存边长。256 约 15.9GB；想完整保留增强细节可设 512(约 64GB)
CACHE_WORKERS = 48        # 预解码的并行进程数（纯 I/O 密集，实测到 32 仍近线性扩展）
CACHE_LIMIT = 0           # >0 时只预解码前 N 个 site，用于快速验证

# ---- Stage 2 目标域: OOC 明场 (3072 张, 6 种细胞系, good/bad) ----
OOC_ROOT = '/bigdat2/user/zhangc/organ/datasets/ooc/OOC_image_dataset'
OOC_DATASHEET = '/bigdat2/user/zhangc/organ/datasets/ooc/OOC_datasheet.xlsx'
# 目录结构: {split}/{good|bad}/*.png，split ∈ {train, val, test}
OOC_SPLITS = ['train', 'val', 'test']

# ---- Stage 3: HTS-Seg 微流控芯片球体 ----
HTS_ROOT = '/bigdat2/user/zhangc/organ/datasets/hts-seg'
# 'tiled_images'(2971张, 论文口径) | 'cleaned_tiled_images'(2180张, 已剔除伪影)
HTS_SUBSET = 'cleaned_tiled_images'
# 送入网络的分辨率。取 224 与 baseline 一致，便于指标对齐（原始 tile 为 512）
HTS_IMG_SIZE = 512#224



# ==========================================================================
# C. Stage 2 域适应：源域 / 目标域（固定 rxrx1 -> ooc）
# ==========================================================================
#
#     DOM_SOURCE='rxrx1'  荧光源域（6 通道里取 3 通道）
#     DOM_TARGET='ooc'    明场目标域
# 域差异最大，最能体现"域适应"的必要性。
#
DOM_SOURCE = 'rxrx1'
DOM_TARGET = 'ooc'

# OOC 的 'Decision 1/2 (good/bad)' 作为辅助分类任务一起训练。
# OOC 是本工程唯一带这类标签的域（1=good, 2=bad）
DOM_AUX_CLS = True

# 各域取哪些数据参与域适应。注意：
#   - RxRx1 域适应阶段只用 train split，与预训练保持一致
#   - OOC 用**全部三个 split**：域适应是无监督的（重建损失 + 域判别器，不用标签），
#     没有"用测试集"的泄漏问题；原设置只取 train，等于把约 30%（≈900 张）
#     器官芯片图像闲置。本工程的目标就是"最大化利用每一张芯片图像"，
#     所以这里全部纳入。OOC 的 good/bad 标签仍只用于辅助分类分支。
OOC_SPLITS_FOR_DOMAIN = ['train', 'val', 'test']
DOM_RXRX1_TRAIN_SPLIT_ONLY = True

# 每个 epoch 结束后在固定的留出集上评估一次（源域/目标域各一个留出集，
# 每个留出集取 DOM_VAL_SAMPLES 张，都走 eval 变换、不增强）。没有这个的话
# 日志里只有训练 loss，判断不了域适应到底有没有起作用。设 0 关闭。
DOM_VAL_SAMPLES = 256


# ==========================================================================
# D. 数据集划分（与 baseline HTS_Seg-main 对齐）
# ==========================================================================

# baseline 用 train_test_split(paths, test_size=0.2, random_state=42)
# 数据集划分方式:
#   'random'        复刻 baseline 的 train_test_split(test_size=0.2, random_state=42)
#                   —— 同分布划分,与论文数字可比
#   'condition'     训练集只取 control/,**测试集只取 treatment/**;
#   'condition_rev' 反过来。
#   为什么需要:'random' 是同分布划分,预训练的价值与跨条件泛化能力都体现不出来
#   (实测 10% 数据就有 Dice 0.9814,已经饱和)。而 HTS-Seg 的掩码按实验条件分在
#   control/treatment 两个目录里,这就是一个现成的**药物处理分布偏移**测试 ——
#   训练时没见过处理过的球体,测试时直接上,测的是跨条件泛化。
#   这比"换一个别的实验室的数据集"更贴近比赛"真实场景、真实转化"的取向,
#   而且数据完全来自器官芯片本身。
SEG_SPLIT_MODE = 'random'
TEST_RATIO = 0.2
VAL_RATIO = 0.1      # 从剩余训练集里再切 10% 做验证（选最优权重/早停）
SPLIT_SEED = 42      # 与 baseline 的 random_state 一致，测试集完全同分布


# ==========================================================================
# E. Stage 1 —— MAE 自监督预训练
# ==========================================================================

MAE_INPUT_SIZE = 512#224      # 输入分辨率（与后续阶段统一）
MAE_PATCH_SIZE = 16       # patch 边长 -> 224/16 = 14x14 = 196 个 patch
MAE_ENC_DIM = 768         # 编码器隐层维度（ViT-Base 口径）
MAE_ENC_DEPTH = 12        # 编码器 Transformer block 数
MAE_ENC_HEADS = 12        # 编码器注意力头数
MAE_DEC_DIM = 512         # MAE 解码器隐层维度（轻量，仅预训练用）
MAE_DEC_DEPTH = 8         # 解码器层数
MAE_DEC_HEADS = 16        # 解码器头数
MAE_MASK_RATIO = 0.75     # 掩码比例，MAE 原文最优值
MAE_NORM_PIX_LOSS = True  # 按 patch 内均值方差归一化后算损失（MAE 原文做法）
# 预训练语料来源：
#   'rxrx1'  大规模 2D 细胞荧光图（125,510 site）—— **不是器官芯片数据**，
#            只提供通用细胞形态表征
#   'oocmix' 只用器官芯片图像：OOC 芯片明场(3,072) + HTS-Seg 芯片球体(2,180)
#            = 5,252 张。用于回答"本赛道下非要非芯片数据吗"——两臂除语料外
#            配置完全相同，差异可直接归因到预训练数据来源。
# 注意：'rxrx1' 是默认值且**不进目录名**（历史实验的路径不能变），只有 'oocmix'
# 会追加后缀，落到独立的一棵输出树里。
MAE_DATA = 'rxrx1'

# 实测（GPU 24GB，bf16，probe_memory.py）：显存 ≈ 1.5 + 0.048*bs (GB)
#   bs=128->7.7GB  bs=192->10.8GB  bs=256->13.9GB
# 想冲满 ~20GB 就设 384，但要注意这台机器是共享的，
# 实际能用的显存取决于别人占了多少；OOM 时 OOMAdaptiveStep 会自动切细 batch
# 显存探针在本卡(24GB)实测的可用值。**必须与 MAE 实际训练用的值一致**:
# 它进 Stage1 目录名,DIR_STAGE2/3 都嵌在它下面 —— 一旦不一致,下游会去另一个
# 空目录找 encoder.pt,静默退化成"随机初始化"(实测发生过:一次 63 小时的 MAE 预训练因此作废)。
MAE_BATCH_SIZE = 64
MAE_EPOCHS = 100#20           # 先跑 20 轮验证全流程；确认无误后改 100 并断点续训
MAE_MAX_STEPS = 0         # >0 时以 step 数为准提前结束（与 MAE_EPOCHS 取先到者）
MAE_LR = 1.5e-4           # MAE 论文 ViT-Base 推荐学习率
MAE_WEIGHT_DECAY = 0.05   # AdamW 解耦权重衰减
MAE_BETAS = (0.9, 0.95)   # MAE 原文用 beta2=0.95
MAE_WARMUP_EPOCHS = 5     # 线性 warmup 轮数
MAE_MIN_LR = 1e-6         # 余弦退火下界
MAE_GRAD_CLIP = 1.0       # 梯度范数裁剪，0=不裁
MAE_AMP = True            # bf16 混合精度，省显存约 40%
MAE_ACCUM_STEPS = 1       # 梯度累积步数（等效放大 batch）
MAE_SAVE_DECODER = False  # 导出下游权重时是否保留 MAE 解码器

# 早停：连续 MAE_EARLY_STOP_PATIENCE 个 epoch 的验证重建损失，都没比历史最优
# 再低 MAE_EARLY_STOP_MIN_DELTA 以上，就认为已收敛并停止。0 = 关闭。
#
# 这条曲线上起作用的是 min_delta，不是 patience —— 实测 100 epoch(bs128) 的
# val_loss 末段基本是噪声级下降：epoch 91~99 每轮只改善 0.00004~0.0007，
# 全程最大连续不改善仅 7 轮，最优值出现在最后一轮(epoch 99)。
#   * min_delta=0 时，就算 patience 放到 10 也要跑到 epoch 99 才触发 → 等于没加。
#   * 反之，想靠调小 patience 省时间在这里同样无效：不改善的连续轮数天然到不了 10。
#   * 真正有效的是 min_delta=1e-3：约在 epoch 90 触发，省约 10 轮，
#     代价是放弃末段约 0.0007 的 val_loss。
# 例：patience=10 + min_delta=1e-3 → 停在 epoch ~90；patience=10 + min_delta=0
#     → 停在 epoch 99（与不早停几乎无差别）。
MAE_EARLY_STOP_PATIENCE = 10
MAE_EARLY_STOP_MIN_DELTA = 1e-3


# ==========================================================================
# F. Stage 2 —— 域适应
# ==========================================================================

# 每步源域/目标域各取 BATCH_SIZE 张，显存按 2x 计。
# 实测：显存 ≈ 1.6 + 0.094*bs (GB)；bs=64->7.6GB  bs=128->13.6GB。冲 20GB 设 192
DOM_BATCH_SIZE = 16#64#128
DOM_EPOCHS = 30
DOM_LR = 2e-5             # 比预训练小一个量级，避免破坏已学表征
DOM_WEIGHT_DECAY = 0.05
DOM_WARMUP_EPOCHS = 2
DOM_RECON_WEIGHT = 1.0    # 重建损失权重
DOM_DOMAIN_WEIGHT = 0.5   # 域判别损失权重
DOM_GRL_LAMBDA = 1.0      # 梯度反转层强度（常数，不做动态调度）
DOM_AUX_WEIGHT = 0.3      # OOC good/bad 辅助分类损失权重（仅 DOM_AUX_CLS=True）
DOM_AMP = True
DOM_GRAD_CLIP = 1.0

# 早停：判据是留出集上的目标域重建损失 recon_tgt（越小越好，与 Stage1 的
# val_loss 同口径）。连续 DOM_EARLY_STOP_PATIENCE 个 epoch 没比历史最优再低
# DOM_EARLY_STOP_MIN_DELTA 以上就停。0 = 关闭。
# 注意 DOM_VAL_SAMPLES=0（不建留出集）时没有指标可判，早停会自动失效并告警。
DOM_EARLY_STOP_PATIENCE = 8
DOM_EARLY_STOP_MIN_DELTA = 0.0


# ==========================================================================
# G. Stage 3 —— HTS-Seg 任务微调（球体分割）
# ==========================================================================

# 显存大头是分割头把 14x14 上采样回 224x224 的中间特征图，约 0.45GB/样本。
# 实测：显存 ≈ 1.2 + 0.45*bs (GB)；bs=16->8.5GB  bs=32->15.8GB。冲 20GB 设 40
SEG_BATCH_SIZE = 32        # 2026-10-05 用户要求把卡吃满:实测显存≈1.2+0.45*bs(GB),
                          #   bs4 只吃 ~3GB、bs32 吃 ~15.8GB(24GB 卡上为同机其它进程留出余量)。
                          #   注意:RxRx1 参考臂 s3_base 是 bs4,本臂与它的对比带一个
                          #   batch/学习率口径差,报告里必须注明。
SEG_EPOCHS = 60
SEG_LR_ENCODER = 1e-4     # 预训练编码器用更小学习率
                          #   bs4->32 按 sqrt 规则放大(×2.83),取保守的 ×2
SEG_LR_DECODER = 1e-3     # 随机初始化的分割头用更大学习率(随 batch 同步放大)
SEG_WEIGHT_DECAY = 0.05
SEG_WARMUP_EPOCHS = 3
SEG_GRAD_CLIP = 1.0
SEG_AMP = True

SEG_FREEZE_LAYERS = 0     # 冻结编码器前 N 层（0=不冻）。小数据集防过拟合可设 6
SEG_LOSS = 'bce_dice'     # 'bce' | 'dice' | 'bce_dice' | 'bce_dice_bnd'(+边界项)
SEG_DECODER = 'deconv'    # 分割头上采样方式: 'deconv' | 'bilinear'
# 编码器初始化:
#   'auto'    按 Stage2 -> Stage1 顺序找预训练编码器(找不到就随机初始化并告警)
#   'scratch' **强制**随机初始化。数据效率实验的对照组必须用它 ——
#             否则一旦 Stage1/2 训好,对照组会静默加载预训练权重,
#             整个"预训练 vs 随机初始化"的对比就变成两组都用了预训练。
SEG_INIT = 'auto'
# 只用训练集的这个比例(1.0=全用)。用于**数据效率实验**:证明"只有数千张芯片
# 图像时预训练的价值"——同分布测试集上 s1s2s3/s2s3/s3 只差 0.0004,看不出迁移
# 的作用;把训练集砍到 10%/25% 再比,预训练的差距才会显出来。
# 子集是**嵌套**的(同一 permutation 取前 k 个),所以 10% ⊂ 25% ⊂ 50% ⊂ 100%。
SEG_TRAIN_FRACTION = 1.0
SEG_BCE_POS_WEIGHT = 1.0  # BCE 正类权重，>1 可缓解球体/背景像素不均衡
SEG_DICE_WEIGHT = 1.0     # dice 项权重（仅 'bce_dice' 生效）
# ---- 边界项(仅 SEG_LOSS 含 'bnd' 时生效) ----
SEG_BND_MODE = 'band'     # 'band'=边界带加权BCE | 'kervadec'=符号距离边界损失
SEG_BND_WEIGHT = 1.0      # 边界项总权重
SEG_BND_SIGMA = 3.0       # band: 高斯带宽度(像素);越小越集中在轮廓线
SEG_BND_ALPHA = 5.0       # band: 边界带峰值权重(1+alpha)
SEG_BND_CLIP = 20.0       # kervadec: 符号距离截断(像素),不限界会被远处背景主导
# 连通域过滤阈值(像素)。
#   * SEG_EVAL_MIN_COMPONENT_PX: 评估侧**额外**输出"过滤后"边界指标
#     (两侧都去掉小于该值的碎点),不改动训练/验证动力学,只增加报告口径。
#   * SEG_POST_MIN_COMPONENT_PX: 推理侧后处理,去掉预测里的碎点。
#     **默认 0(关闭)** —— 它会影响验证集 Dice 进而影响最优权重的选择,
#     开着会让"边界损失消融"的 A/B/C/D 四组不在同一口径下比较。
#     要在受控条件下评估它的效果,用 eval_tta.py 或单独跑一次对比。
# 依据:实测 GT 里 91.7% 的连通域 <20px、只占面积 0.1%,真球体占 99.77%;
# 这些单像素噪点会让 max-Hausdorff 完全失真。
SEG_EVAL_MIN_COMPONENT_PX = 50
SEG_POST_MIN_COMPONENT_PX = 0
SEG_THRESHOLD = 0.5       # 预测概率 > 该值判为正类（与 baseline 同值）
# 验证/测试时的 batch size。推理没有梯度、显存占用小得多，可以比训练大；
# 设 0 表示沿用 SEG_BATCH_SIZE。主推理路径带 OOM 自动对半切(_forward_oom_safe)，
# 但**别指望它兜住一切**：patch8(4096 token)那次就是画图分支直接 `model(img)`
# 跑满 64 → 单次注意力要 4.5GB → 训练跑完却在收尾崩溃(eval_seg.py 已修)。
# 降到 16 是 512px 输入下的稳妥值：少掉一轮轮 OOM 重试的空转，也不再逼近显存上限。
EVAL_BATCH_SIZE = 16

# 早停：判据是验证集前景 Dice（越大越好）。连续 SEG_EARLY_STOP_PATIENCE 个
# epoch 没比历史最优再高 SEG_EARLY_STOP_MIN_DELTA 以上就停。0 = 关闭。
# HTS 的验证集只有 238 张图，Dice 逐轮抖动相对大，所以 min_delta 保持 0、
# 靠放大 patience 到 12 吸收抖动，避免把正常波动当成收敛。
SEG_EARLY_STOP_PATIENCE = 12
SEG_EARLY_STOP_MIN_DELTA = 0.0

# --------------------------------------------------------------------------
# Stage3 的训练随机种子（**只影响模型初始化 / 数据打乱 / 增强**,不影响数据划分）。
# 为什么单列一个参数:
#   数据划分种子(SPLIT_SEED)必须固定,否则测试集变了、所有实验不可比;
#   但训练种子必须能变,否则无法做**多种子重复**,也没法给出误差棒。
#   论文 Table 2 里三步迁移的增益(0.0004 Dice)恰好等于 1 个标准误 —— 单种子下
#   这种差异无法区分于噪声。跑 3 个种子做概率平均,既能拿到更稳的指标
#   (集成通常 +0.001~0.003 Dice 并改善边界),也能报出 mean±std。
# 42 = 与既有实验一致(默认值不进目录名,'--SEG_SEED 43' 等才追加后缀)。
SEG_SEED = 42


# ==========================================================================
# H. 其它
# ==========================================================================

AUTO_EVAL_AFTER_TRAIN = True   # Stage 3 训完自动跑评估 + 论文表格对齐报告
PROBE_MEMORY = False           # True 时先跑显存探针确定 batch size 再训练


# ==========================================================================
# I. 输出目录（依赖上面**全部**参数，必须放在文件末尾）
# ==========================================================================
#
# 目录名 = 实验身份。凡是会改变最终指标（分割 Dice/IoU）的
# 参数，都按「名字的语义缩写 + 取值」编进路径，不同设定自动落到不同目录，
# 不会再出现"换了参数重跑、却被 RESUME='auto' 当成断点续训接上"的情况。
#
# 分层（自外向内，每层只装自己那一层的参数，**该层目录本身就是那一阶段的输出目录**）：
#
#   <_ROOT_BASE>/<split-…>/     ← 全局 + 数据划分 + RUN_TAG；_cache/、summary/ 在这层
#                   └── s1-…/   ← Stage1 参数；train_mae.py 的输出直接放这里
#                       └── s2-…/  ← Stage2 参数；train_domain.py 的输出直接放这里
#                           └── s3-…/  ← Stage3 参数；train_seg.py 的输出放这里
#
# 为什么 Stage2/3 的参数层要嵌在 Stage1 **下面**：下游依赖上游导出的 encoder.pt，
# 所以换上游参数时下游必须换目录（跟着重跑）；反之换 Stage2 参数（DOM_* 等）
# 不应该动 Stage1 —— 否则一次 Stage2 调参就会把已训好的 Stage1 输出"挪走"，
# Stage2 找不到 encoder.pt，静默退化成随机初始化。实测出现过该问题：
# DOM_BATCH_SIZE 128→64 之后日志出现「未找到 Stage1 编码器，Stage2 将从随机
# 初始化开始」，而 100 epoch 的编码器还躺在旧路径下。
#
# 分层的另一个原因：约 50 个参数塞进单个目录名会超过单组件 255 字节的 NAME_MAX
# 上限（ENAMETOOLONG）。按 全局 / Stage1 / Stage2 / Stage3 切成 4 段后，实测每段
# 55 / 140 / 89 / 145 字节，整条路径约 460 字符，都远低于上限。
#
# 归属规则（双向，两边都要守）：
#   * 影响最终指标的参数 → 必须进目录名，不同取值必须产生不同路径。
#     例：MAE_MASK_RATIO 0.75 vs 0.5 → 路径必须不同。
#   * 反之，**不影响**指标的纯运行期旋钮 → 必须不进目录名，否则同一个实验会因为
#     --NUM_WORKERS 2 这种速度旋钮被拆成两个目录，白白重跑一次 81k site × 100 epoch。
#     例：NUM_WORKERS 8 vs 2、EVAL_BATCH_SIZE 64 vs 16、--DEVICE_ID 6 vs 4
#         → 生成的 OUTPUT_ROOT 必须完全相同。
#   * 边界情况：早停参数（MAE_/DOM_/SEG_EARLY_STOP_*）**进**目录名。
#     它不改训练动力学、看着像省时间的速度旋钮，但它决定停在哪个 epoch，
#     也就决定最终落盘的是哪一份权重 → 属于"影响最终指标"，进名字。
#     反向确认：如果将来加的是"只影响日志/落盘频率"的旋钮（LOG_EVERY_STEPS、
#     SAVE_EVERY_STEPS 这类），那才是必须排除的 —— 它们连落盘内容都不改。
#
# 有意排除的运行期旋钮（理由见 README 第 4 节）：
#   NUM_WORKERS / CACHE_WORKERS / CACHE_LIMIT / DEVICE_ID / DEVICE /
#   EVAL_BATCH_SIZE / MAE_AMP / DOM_AMP / SEG_AMP / LOG_EVERY_STEPS /
#   SAVE_EVERY_EPOCH / SAVE_EVERY_STEPS / KEEP_LAST_N / RESUME /
#   AUTO_EVAL_AFTER_TRAIN / PROBE_MEMORY / DOM_VAL_SAMPLES / MAE_SAVE_DECODER
# 数据根目录（RXRX1_ROOT / OOC_ROOT / HTS_ROOT / RXRX1_CACHE_DIR）也不进名字：
# 完整路径太长，而像素内容已由 RXRX1_CHANNELS + RXRX1_CACHE_SIZE 的缓存标签覆盖。

_ROOT_BASE = '/bigdat2/user/zhangc/organ_v5'


def _fmt(v):
    """把参数值压成短标签：bool->1/0，小浮点走科学计数法，序列按元素类型拼接。"""
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        if v != 0 and abs(v) < 0.01:                      # 1.5e-4 而不是 0.00015
            m, e = f'{v:.6e}'.split('e')
            return f'{m.rstrip("0").rstrip(".")}e{int(e)}'
        return f'{v:g}'
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return ''.join(str(x) for x in v)             # [0, 1, 4] -> '014'
        if all(isinstance(x, str) for x in v):
            return '.'.join(x[:3] for x in v)             # ['train'] -> 'tra'
        return '.'.join(_fmt(x) for x in v)               # (0.9, 0.95) -> '0.9.0.95'
    return str(v).replace('/', '-')


def _split_tag(tag=None):
    """全局 + 数据划分 + RUN_TAG。这些参数三阶段共用，变了就得全量重跑。

    注意:SEG_SPLIT_MODE='random' 时**不进**目录名 —— 早期实验（A/B/C/D 消融、
    数据效率）都是在没有这个字段时落盘的，一旦把它编进名字，那些结果的路径会
    全部改变、被当成"没跑过"而重跑。只有非默认的划分模式才追加后缀。
    """
    tag = RUN_TAG if tag is None else tag
    mode = '' if SEG_SPLIT_MODE == 'random' else f'-{SEG_SPLIT_MODE}'
    return (f'split-{HTS_SUBSET}{mode}-tr{_fmt(TEST_RATIO)}'
            f'-va{_fmt(VAL_RATIO)}-s{SPLIT_SEED}-seed{SEED}'
            f'-det{_fmt(DETERMINISTIC)}{tag}')


def _stage1_tag():
    """Stage 1（MAE 预训练）。编码器质量的来源，直接决定 Stage 3 的指标。

    MAE_DATA='rxrx1'（默认）不进名字：A/B/C/D 消融、数据效率等已完成的实验都是
    在那之前落盘的，一旦编进目录名，它们的路径会全部改变、被当成没跑过而重跑。
    'oocmix' 是新增的对照臂，追加后缀即可（落到独立输出树）。
    """
    data = '' if MAE_DATA == 'rxrx1' else f'-{MAE_DATA}'
    return (f's1-rxrx1-c{_fmt(RXRX1_CHANNELS)}'
            f'-tr{_fmt(RXRX1_TRAIN_SPLIT_ONLY)}-sub{_fmt(RXRX1_SUBSET_RATIO)}'
            f'-cache{RXRX1_CACHE_SIZE if RXRX1_CACHE else 0}'
            f'-vitb{MAE_PATCH_SIZE}-in{MAE_INPUT_SIZE}'
            f'-dim{MAE_ENC_DIM}x{MAE_ENC_DEPTH}x{MAE_ENC_HEADS}'
            f'-dec{MAE_DEC_DIM}x{MAE_DEC_DEPTH}x{MAE_DEC_HEADS}'
            f'-m{_fmt(MAE_MASK_RATIO)}-np{_fmt(MAE_NORM_PIX_LOSS)}'
            f'-ep{MAE_EPOCHS}u{MAE_MAX_STEPS}-bs{MAE_BATCH_SIZE}'
            f'-lr{_fmt(MAE_LR)}-wd{_fmt(MAE_WEIGHT_DECAY)}-b{_fmt(MAE_BETAS)}'
            f'-wu{MAE_WARMUP_EPOCHS}-mn{_fmt(MAE_MIN_LR)}'
            f'-gc{_fmt(MAE_GRAD_CLIP)}-acc{MAE_ACCUM_STEPS}'
            f'-es{MAE_EARLY_STOP_PATIENCE}d{_fmt(MAE_EARLY_STOP_MIN_DELTA)}{data}')


def _stage2_tag():
    """Stage 2（域适应）。域组合固定为 rxrx1 -> ooc，但仍进名字：它决定
    Stage2 的全部输入，换掉就该换目录，不能让旧域的断点被 RESUME='auto' 接上。"""
    return (f's2-{DOM_SOURCE}2{DOM_TARGET}-aux{_fmt(DOM_AUX_CLS)}'
            f'-ocsp.{_fmt(OOC_SPLITS_FOR_DOMAIN)}'
            f'-rxtr{_fmt(DOM_RXRX1_TRAIN_SPLIT_ONLY)}'
            f'-ep{DOM_EPOCHS}-bs{DOM_BATCH_SIZE}-lr{_fmt(DOM_LR)}'
            f'-wd{_fmt(DOM_WEIGHT_DECAY)}-wu{DOM_WARMUP_EPOCHS}'
            f'-rc{_fmt(DOM_RECON_WEIGHT)}-dm{_fmt(DOM_DOMAIN_WEIGHT)}'
            f'-aw{_fmt(DOM_AUX_WEIGHT)}-grl{_fmt(DOM_GRL_LAMBDA)}'
            f'-gc{_fmt(DOM_GRAD_CLIP)}'
            f'-es{DOM_EARLY_STOP_PATIENCE}d{_fmt(DOM_EARLY_STOP_MIN_DELTA)}')


def _stage3_tag():
    """Stage 3（球体分割）。这些参数 Stage 1/2 都不读，所以只挂在 s3 层的
    目录名上 —— 改 SEG_* 时上游编码器原地复用，不必重跑 MAE。"""
    trseed = '' if SEG_SEED == SEED else f'-trseed{SEG_SEED}'
    tag = (f's3-img{HTS_IMG_SIZE}-seg-ep{SEG_EPOCHS}bs{SEG_BATCH_SIZE}'
           f'-lr{_fmt(SEG_LR_ENCODER)}.{_fmt(SEG_LR_DECODER)}'
           f'-wd{_fmt(SEG_WEIGHT_DECAY)}wu{SEG_WARMUP_EPOCHS}'
           f'-gc{_fmt(SEG_GRAD_CLIP)}frz{SEG_FREEZE_LAYERS}-{SEG_LOSS}'
           f'-{SEG_DECODER}-{SEG_INIT}-frac{_fmt(SEG_TRAIN_FRACTION)}'
           f'-pos{_fmt(SEG_BCE_POS_WEIGHT)}dw{_fmt(SEG_DICE_WEIGHT)}')
    if 'bnd' in SEG_LOSS:
        tag += (f'-bnd{SEG_BND_MODE}w{_fmt(SEG_BND_WEIGHT)}'
                f's{_fmt(SEG_BND_SIGMA)}a{_fmt(SEG_BND_ALPHA)}')
    tag += (f'-th{_fmt(SEG_THRESHOLD)}'
            f'-es{SEG_EARLY_STOP_PATIENCE}d{_fmt(SEG_EARLY_STOP_MIN_DELTA)}{trseed}')
    return tag


def output_root(tag=None):
    """当前参数对应的「全局层」目录：三个阶段目录都在它下面，_cache/（数据集
    清单缓存）与 summary/（报告）也在这一层。tag 都写成函数（不是模块级常量），
    这样 CLI 覆盖 --MAE_EPOCHS 之后调 refresh_output_dirs() 能拿到新值；
    report.py 用 tag 参数复现"带 RUN_TAG 的那棵树"的路径。"""
    return os.path.join(_ROOT_BASE, _split_tag(tag))


def stage_dir_names():
    """三个阶段目录，相对 output_root() 的路径（逐层嵌套，见本节开头的分层图）。
    report.py 也用它，避免各自硬编码字符串。"""
    s1, s2, s3 = _stage1_tag(), _stage2_tag(), _stage3_tag()
    return (s1, os.path.join(s1, s2), os.path.join(s1, s2, s3))


def _stage_dirs(tag=None):
    root = output_root(tag)
    return tuple(os.path.join(root, n) for n in stage_dir_names())


def refresh_output_dirs():
    """CLI 覆盖了任何会进目录名的参数之后调用，重算 OUTPUT_ROOT 与三个阶段目录。"""
    global OUTPUT_ROOT, DIR_STAGE1, DIR_STAGE2, DIR_STAGE3
    OUTPUT_ROOT = output_root()
    DIR_STAGE1, DIR_STAGE2, DIR_STAGE3 = _stage_dirs()


OUTPUT_ROOT = output_root()

# 子目录名（resume 依赖它定位 last.pt，勿随意改）
DIR_STAGE1, DIR_STAGE2, DIR_STAGE3 = _stage_dirs()
