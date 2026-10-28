# xiph RNNoise 0.2 (48kHz) 复现训练完整流程

目标：用官方公开数据 + 官方代码，在训练服务器上复现 48kHz RNNoise 0.2 模型，
确认效果后再迁移到 16kHz。

前提：
- 数据已下载（tts_speech_48k.sw / background_noise_v2.sw / foreground_noise_v2.sw /
  contrib_noise.sw / measured_rirs-v3.tar.gz）
- 训练服务器有 GPU + PyTorch，但还没有代码
- 本地已有 `xiph_rnnoise/` 完整仓库

---

## 阶段 0：把代码和数据传到训练服务器

训练服务器（示例 10.222.222.5，用户 oujie5）。先建目录：

```bash
# 在服务器上
ssh oujie5@10.222.222.5
mkdir -p ~/rnnoise48k/{data,rirs}
exit
```

把本地 xiph 代码推上去（在本地 WSL 执行）：

```bash
# 代码（排除 .git 和已编译产物，减小体积）
rsync -avP --exclude='.git' --exclude='*.tar.gz' \
  /mnt/e/server_code/rnnoise_16k/xiph_rnnoise/ \
  oujie5@10.222.222.5:~/rnnoise48k/xiph_rnnoise/

# 数据（假设本地数据在 E:\...\xiph_data\，按你实际路径改）
rsync -avP /mnt/e/server_code/rnnoise_16k/xiph_data/tts_speech_48k.sw \
  /mnt/e/server_code/rnnoise_16k/xiph_data/background_noise_v2.sw \
  /mnt/e/server_code/rnnoise_16k/xiph_data/foreground_noise_v2.sw \
  /mnt/e/server_code/rnnoise_16k/xiph_data/contrib_noise.sw \
  oujie5@10.222.222.5:~/rnnoise48k/data/

# RIR 压缩包
rsync -avP /mnt/e/server_code/rnnoise_16k/xiph_data/measured_rirs-v3.tar.gz \
  oujie5@10.222.222.5:~/rnnoise48k/rirs/
```

> tts_speech_48k.sw 有 63G，rsync -P 支持断点续传，中断可重跑。

---

## 阶段 1：服务器上编译 dump_features（特征生成工具）

```bash
ssh oujie5@10.222.222.5
cd ~/rnnoise48k/xiph_rnnoise/src

# 编译 dump_features（用官方 Makefile 里的方式，手动 gcc）
gcc -DTRAINING=1 -O3 -g -I. -I.. -I../include \
    dump_features.c denoise.c kiss_fft.c pitch.c celt_lpc.c rnn.c \
    rnn_data.c -o dump_features -lm

# 若报缺 rnnoise_data 相关符号，改用仓库自带的（已解压 rnnoise_data.c）：
# 检查 src/ 下有 rnnoise_data.c，把上面命令里的 rnn_data.c 换成 rnnoise_data.c 再试
ls -la dump_features
```

> ⚠️ dump_features.c 依赖的具体 .c 文件以实际编译报错为准（缺哪个符号补哪个）。
> 官方也提供 autotools：`cd .. && ./autogen.sh && ./configure && make`，
> 成功后 dump_features 在 src/ 下。

---

## 阶段 2：准备 RIR 列表（可选，但官方模型用了）

```bash
cd ~/rnnoise48k/rirs
tar xf measured_rirs-v3.tar.gz          # 解出一堆 .f32 (32-bit float LE)
# 生成 RIR 文件清单（绝对路径，每行一个）
find $PWD -name '*.f32' > rir_list.txt
wc -l rir_list.txt                      # 确认条数
```

---

## 阶段 3：生成训练特征 features.f32

dump_features 用法（从源码确认）：
```
dump_features [-rir_list list] <speech> <noise> <fg_noise> <output> <count>
```
- speech = 干净语音，noise = 背景噪声，fg_noise = 前景噪声
- count = 序列数（官方建议 ≥10000，推荐 200000+）；每序列 2000 帧

背景噪声可以先把 background + contrib 拼一个大文件（都是 raw 48k s16le）：
```bash
cd ~/rnnoise48k/data
cat background_noise_v2.sw contrib_noise.sw > bg_all.sw
```

### 3a. 不带 RIR（先跑通）

```bash
cd ~/rnnoise48k/xiph_rnnoise/src
./dump_features \
    ~/rnnoise48k/data/tts_speech_48k.sw \
    ~/rnnoise48k/data/bg_all.sw \
    ~/rnnoise48k/data/foreground_noise_v2.sw \
    ~/rnnoise48k/data/features.f32 \
    200000
```

### 3b. 带 RIR（正式训练，混响泛化更好）

```bash
./dump_features -rir_list ~/rnnoise48k/rirs/rir_list.txt \
    ~/rnnoise48k/data/tts_speech_48k.sw \
    ~/rnnoise48k/data/bg_all.sw \
    ~/rnnoise48k/data/foreground_noise_v2.sw \
    ~/rnnoise48k/data/features.f32 \
    200000
```

### 3c. 加速（多进程并行，官方脚本）

```bash
cd ~/rnnoise48k/xiph_rnnoise
bash scripts/dump_features_parallel.sh src/dump_features \
    data/tts_speech_48k.sw data/bg_all.sw data/foreground_noise_v2.sw \
    data/features.f32 200000 rirs/rir_list.txt
```

> features.f32 会很大（200000 序列 × 2000 帧 × 98 维 × 4 字节 ≈ 150G+），
> 确认磁盘空间。可先用小 count（如 20000）验证全流程再放大。

---

## 阶段 4：安装 PyTorch 训练依赖

```bash
cd ~/rnnoise48k/xiph_rnnoise/torch
# 权重导出工具（训练后导出 C 权重要用）
pip install -e weight-exchange
pip install torch numpy tqdm
```

---

## 阶段 5：训练

train_rnnoise.py 用法（从源码确认）：
```
train_rnnoise.py <features.f32> <output_folder> [选项]
```
关键选项：
- `--cond-size 128`（Conv 条件维度，默认 128）
- `--gru-size 384`（GRU 隐藏层，默认 384，这就是官方 0.2 规格）
- `--epochs 200`（默认，需约 75000 次权重更新）
- `--batch-size 128` / `--lr 1e-3` / `--lr-decay 5e-5`（默认，对齐官方）
- `--sparse`（稀疏化训练，若要复现 little/稀疏模型才加）
- `--initial-checkpoint`（预加载，从头训不用）

```bash
cd ~/rnnoise48k/xiph_rnnoise/torch/rnnoise

# 标准复现（gru=384, 200 epochs）
python3 train_rnnoise.py \
    ~/rnnoise48k/data/features.f32 \
    ~/rnnoise48k/output_48k \
    --gru-size 384 --cond-size 128 \
    --epochs 200 --batch-size 128 --lr 1e-3 --lr-decay 5e-5

# 输出：output_48k/checkpoints/rnnoise_1.pth ... rnnoise_200.pth
```

> epochs 数要让权重更新 ≈ 75000 次。更新次数 = epochs × (序列数/batch_size)。
> 200000 序列 / 128 ≈ 1562 步/epoch，约 48 epoch 就到 75000 步。
> 数据少时相应加大 epochs。跑够后取 loss 最低的 checkpoint。

---

## 阶段 6：导出 C 权重

dump_rnnoise_weights.py（torch/rnnoise/ 下）：
```bash
cd ~/rnnoise48k/xiph_rnnoise/torch/rnnoise
python3 dump_rnnoise_weights.py --quantize \
    ~/rnnoise48k/output_48k/checkpoints/rnnoise_200.pth \
    ~/rnnoise48k/rnnoise_c

# 产出 rnnoise_c/rnnoise_data.c 和 rnnoise_data.h
```

---

## 阶段 7：编译带新权重的 demo 并验证

```bash
# 把导出的权重覆盖到 src/
cp ~/rnnoise48k/rnnoise_c/rnnoise_data.c ~/rnnoise48k/xiph_rnnoise/src/
cp ~/rnnoise48k/rnnoise_c/rnnoise_data.h ~/rnnoise48k/xiph_rnnoise/src/

# 编译 demo
cd ~/rnnoise48k/xiph_rnnoise
gcc -O2 -Wall -std=c99 -Iinclude -Isrc \
    examples/rnnoise_demo.c \
    src/denoise.c src/rnn.c src/pitch.c src/kiss_fft.c src/celt_lpc.c \
    src/nnet.c src/nnet_default.c src/parse_lpcnet_weights.c \
    src/rnnoise_data.c src/rnnoise_tables.c \
    -lm -o rnnoise_demo_mymodel

# 跑测试音频（48k raw s16le）
./rnnoise_demo_mymodel noisy_48k.pcm out_48k.pcm
```

把结果和官方 rnnoise_demo（用官方权重）在同一批音频上对比 PESQ/听感，
确认你的复现模型达到官方水平。

---

## 阶段 8：确认后迁移到 16kHz

48k 复现确认无误后，再做 16k：
1. 所有 speech/noise/RIR 用 `resample_poly` 或 ffmpeg 从 48k → 16k
2. 用你 `training/rnnoise_16k/dump_features_16k.c`（16k 版特征提取）重新生成特征
3. 用 `training/rnnoise_16k/train_rnnoise_16k.py` 训练 16k 模型
4. 导出 → 集成到 `rnnoise/` C 引擎

---

## 关键提醒

- **无法逐位复现**：官方 README 明说结果依赖数据混合、训练时长、随机种子，
  只能复现到"同等 PESQ 水平"，不是同一个权重文件。
- **磁盘**：features.f32 极大，先小 count 验证再放大。
- **48k vs 16k 特征维度不同**：xiph 是 98 维（65 特征 + 32 gain + 1 vad），
  你的 16k 版是 68 维（45 + 22 + 1），两套 dump_features 不能混用。
- **先验证 pipeline 再堆数据**：阶段 3 先用 count=20000 跑通 dump→train→export→demo
  全链路，确认没报错，再上 200000 正式训练。
