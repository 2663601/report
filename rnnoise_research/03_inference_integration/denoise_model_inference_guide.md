# 降噪模型推理指南

对单个 PCM 文件跑多个降噪模型做对比。所有命令在 WSL 终端执行。

输入格式统一为 **16kHz 16bit mono signed little-endian PCM**。

---

## 1. RNNoise v3.1（自研，C 推理）

直接处理 16kHz PCM，无需重采样。

```bash
# 可执行文件位置
DEMO=/mnt/e/server_code/rnnoise_16k/rnnoise/demo_v31_full

# 用法: demo_v31_full <input.pcm> <output.pcm> [mode]
# mode: 0=aoplay(默认), 1=aitalk
$DEMO input.pcm output_v31.pcm 0
```

---

## 2. RNNoise v0.1（原始开源，48kHz）

v0.1 工作在 48kHz，需要先 16k→48k 重采样，处理后再 48k→16k 降回来。

```bash
# 可执行文件位置（从 xiph/rnnoise v0.1 tag 编译）
DEMO=/tmp/rnnoise01_demo

# 如果不存在，编译方式：
# git clone https://github.com/xiph/rnnoise.git /tmp/rnnoise_official
# cd /tmp/rnnoise_official && git checkout v0.1
# gcc -O2 -Iinclude -Isrc examples/rnnoise_demo.c \
#     src/denoise.c src/kiss_fft.c src/pitch.c src/celt_lpc.c \
#     src/rnn.c src/rnn_data.c -o /tmp/rnnoise01_demo -lm

# 重采样 + 处理 + 降采样
ffmpeg -y -f s16le -ar 16000 -ac 1 -i input.pcm -f s16le -ar 48000 -ac 1 /tmp/input_48k.pcm 2>/dev/null
$DEMO /tmp/input_48k.pcm /tmp/output_v01_48k.pcm
ffmpeg -y -f s16le -ar 48000 -ac 1 -i /tmp/output_v01_48k.pcm -f s16le -ar 16000 -ac 1 output_v01.pcm 2>/dev/null
```

---

## 3. RNNoise v0.2 xiph（开源新版，48kHz）

同样 48kHz，和 v0.1 流程一样。

```bash
# 可执行文件位置
DEMO=/mnt/e/server_code/rnnoise_16k/xiph_rnnoise/rnnoise_demo

# 如果不存在，编译方式：
# cd /mnt/e/server_code/rnnoise_16k/xiph_rnnoise
# bash build_demo.sh

# 重采样 + 处理 + 降采样
ffmpeg -y -f s16le -ar 16000 -ac 1 -i input.pcm -f s16le -ar 48000 -ac 1 /tmp/input_48k.pcm 2>/dev/null
$DEMO /tmp/input_48k.pcm /tmp/output_v02_48k.pcm
ffmpeg -y -f s16le -ar 48000 -ac 1 -i /tmp/output_v02_48k.pcm -f s16le -ar 16000 -ac 1 output_v02.pcm 2>/dev/null
```

---

## 4. GTCRN（PyTorch 推理）

需要 PyTorch 环境。代码在 `E:\server_code\rnnoise_16k\gtcrn`。

```bash
# 参考脚本
bash /mnt/e/server_code/rnnoise_16k/training/rnnoise_16k/inference/_run_hp7_gtcrn_dfnet.sh
```

单文件推理的核心 Python 代码：

```python
import sys, os, numpy as np, torch
sys.path.insert(0, '/mnt/e/server_code/rnnoise_16k/gtcrn')
from gtcrn import GTCRN

# 加载模型
model = GTCRN().eval()
ckpt = torch.load('/mnt/e/server_code/rnnoise_16k/gtcrn/checkpoints/model_trained_on_dns3.tar', map_location='cpu')
state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
model.load_state_dict(state)

# 读取 16kHz s16le PCM
raw = np.fromfile('input.pcm', dtype=np.int16)
audio = raw.astype(np.float32) / 32768.0

# STFT + 推理 + ISTFT
win = torch.hann_window(512).pow(0.5)
x = torch.from_numpy(audio)
spec = torch.stft(x, 512, 256, 512, win, return_complex=False)
with torch.no_grad():
    out = model(spec[None])[0]
out_c = torch.view_as_complex(out.contiguous())
enh = torch.istft(out_c, 512, 256, 512, win)

# 保存
out_int16 = np.clip(enh.numpy() * 32768, -32768, 32767).astype(np.int16)
out_int16.tofile('output_gtcrn.pcm')
```

### GTCRN 两个预训练权重

`gtcrn/checkpoints/` 下有两个官方权重，可换着跑对比：

| 权重 | 训练数据 | 特点 |
|------|------|------|
| `model_trained_on_dns3.tar` | DNS3 Challenge | 噪声类型杂、泛化广，默认用这个 |
| `model_trained_on_vctk.tar` | VCTK | 偏干净朗读语音，压制更保守 |

两者结构完全相同（GTCRN 48.2K 参数），只换 `torch.load` 的路径即可。

### wav 输入 + 非 16k 重采样

脚本模板读的是 16k s16le PCM；若输入是 **wav 或非 16kHz**，用 soundfile 读 + scipy 重采样（本地 WSL 无 librosa，用 `resample_poly`）：

```python
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
data, sr = sf.read('in.wav', dtype='float32', always_2d=True)
data = data.T[0]                        # 取单声道
if sr != 16000:
    g = gcd(sr, 16000)
    data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
# 之后 data 作为 16k 波形进 STFT
```

两个权重一次跑完的脚本见 `test_data/baby_cry/_run_gtcrn.py`。

### 实测记录：baby_cry (2026-10)

- 输入 `test_data/baby_cry/4-185575-B.wav`（44100Hz，重采样到 16k = 80016 帧）
- 输出 `test_data/baby_cry/out/4-185575-B_gtcrn_{dns3,vctk}.wav`，out_peak 分别 0.612 / 0.698（vctk 压得更保守）。
- ⚠️ **方向提醒**：GTCRN 是语音增强（保人声、压非语音噪声），**婴儿哭声会被当噪声压掉**。
  若目的是"提取/保留哭声"，这个模型方向相反；若目的是"把哭声当噪声去除"，才合适。

---

## 5. DeepFilterNet3（PyTorch 推理，48kHz 内部）

需要 `deepfilternet` 包。代码在 `E:\server_code\rnnoise_16k\DeepFilterNet`。
内部工作在 48kHz，用 scipy 做重采样。

```bash
# 参考脚本（和 GTCRN 在同一个脚本里）
bash /mnt/e/server_code/rnnoise_16k/training/rnnoise_16k/inference/_run_hp7_gtcrn_dfnet.sh
```

核心 Python 代码：

```python
import sys, os, types, numpy as np, torch
from scipy.signal import resample_poly

# torchaudio 兼容 shim（必须在 import df 之前）
import torchaudio
if not hasattr(torchaudio, 'backend'):
    torchaudio.backend = types.ModuleType('torchaudio.backend')
    sys.modules['torchaudio.backend'] = torchaudio.backend
if not hasattr(torchaudio.backend, 'common'):
    common_mod = types.ModuleType('torchaudio.backend.common')
    class AudioMetaData:
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
    common_mod.AudioMetaData = AudioMetaData
    torchaudio.backend.common = common_mod
    sys.modules['torchaudio.backend.common'] = common_mod

from df.enhance import enhance, init_df

model, df_state, _ = init_df()
df_sr = df_state.sr()  # 48000

# 读取 16kHz s16le PCM
raw = np.fromfile('input.pcm', dtype=np.int16)
data = raw.astype(np.float32) / 32768.0

# 16k -> 48k -> enhance -> 48k -> 16k
up = resample_poly(data, df_sr, 16000).astype(np.float32)
audio_t = torch.from_numpy(up).unsqueeze(0)
with torch.no_grad():
    enh = enhance(model, df_state, audio_t)
enh16 = resample_poly(enh.squeeze(0).numpy(), 16000, df_sr).astype(np.float32)

out_int16 = np.clip(enh16 * 32768, -32768, 32767).astype(np.int16)
out_int16.tofile('output_dfnet.pcm')
```

---

## 6. WebRTC Transient Suppressor + Pitch Recovery

```bash
cd /mnt/e/server_code/rnnoise_16k/audio_processing/webrtc_transient
bash build_pitch.sh
./transient_pitch_demo input.pcm output_transient.pcm
```

---

## 快速对比脚本

对一个 PCM 文件跑全部 5 个模型：

```bash
INPUT=/mnt/e/server_code/rnnoise_16k/test_data/hp7_data/ai_origin.pcm
OUTDIR=/mnt/e/server_code/rnnoise_16k/test_data/hp7_data

# 1) v3.1
/mnt/e/server_code/rnnoise_16k/rnnoise/demo_v31_full "$INPUT" "$OUTDIR/ai_origin_v31_out.pcm" 0

# 2) v0.1 + v0.2 (需要重采样)
bash /mnt/e/server_code/rnnoise_16k/training/rnnoise_16k/inference/_run_hp7_rnnoise_v01_v02.sh

# 3) GTCRN + DFNet
bash /mnt/e/server_code/rnnoise_16k/training/rnnoise_16k/inference/_run_hp7_gtcrn_dfnet.sh
```

输出文件：
- `ai_origin_v31_out.pcm` — RNNoise v3.1
- `ai_origin_v01.pcm` — RNNoise v0.1
- `ai_origin_v02.pcm` — RNNoise v0.2 xiph
- `ai_origin_gtcrn.pcm` — GTCRN
- `ai_origin_dfnet.pcm` — DeepFilterNet3

---

## 注意事项

- v0.1/v0.2 的重采样脚本中输入路径硬编码在 `_run_hp7_rnnoise_v01_v02.sh` 里，换文件需要改脚本里的 `INPUT` 变量
- GTCRN/DFNet 脚本同理，输入路径在 `_run_hp7_gtcrn_dfnet.sh` 的 `INPUT_PCM` 变量
- DFNet 需要 torchaudio shim（脚本里已包含），如果 torchaudio 升级可能需要调整
- 所有 PCM 输入必须是 16kHz 16bit mono s16le，不支持 WAV 头
