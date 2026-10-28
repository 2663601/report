# RNNoise 16kHz 降噪引擎项目答辩文档

## 一、项目概述

### 1.1 项目背景

在嵌入式音视频设备（对讲门铃、摄像头）场景中，需要一个运行在 ARM Cortex-A7 级别芯片上的实时单通道语音降噪模块。原有方案（V1 aoplay，基于 Mozilla rnnoise 早期版本 16.9K 参数）降噪能力不足（PESQ 仅 1.703），无法满足产品需求。

### 1.2 项目目标

基于 xiph/rnnoise 最新架构（v0.2），设计并实现一个面向 16kHz 采样率的极小模型降噪引擎：
- 参数量 <10K，ARM 每帧推理 <120µs（10ms 帧预算的 1.2%）
- PESQ 显著优于 V1（目标 >1.85，实测达 1.89）
- 保持 `YS_RNN_Interface.h` 接口不变，可无缝替换 VQE 中旧版 NS 模块

### 1.3 最终成果 (V3.1)

| 指标 | V1 旧版 (aoplay) | V3.1 本项目 | 提升 |
|------|:---:|:---:|:---:|
| 参数量 | 16,883 | 9,351 | -45% |
| PESQ (50条DNS2020) | ~1.6 | 1.959 | +0.36 |
| x86 每帧耗时 | 30.1µs | 20.3µs | 33% 加速 |
| x86 NN 推理 | 9.2µs | 1.9µs | 4.8x 加速 |
| ARM NEON 每帧 (qemu) | 531µs | 416µs | 22% 加速 |
| ARM NN 推理 (qemu) | 182µs | 87µs | 2.1x 加速 |
| ARM 真实硬件 (估计) | — | ~30-35µs | 帧预算 0.3% |
| 全维度 NEON 对齐 | ✗ | ✓ (16 倍数) | — |
| Pitch filter | 无 (注释掉) | 有 (梳状滤波) | 谐波增强 |
| 1帧 look-ahead | 无 | 有 (delayed_X) | 瞬态更佳 |
| Int8 量化 SNR | — | 32~46 dB | 听感无损 |

---

## 二、技术方案与核心原理

### 2.1 信号处理 Pipeline

```
PCM (16kHz, 10ms/帧, 160 samples)
  │
  ▼ 高通滤波 (去直流)
  │
  ▼ 分析窗 (320pt, 50% overlap) + FFT → 频谱 X[k]
  │
  ▼ 三角窗加权求和 → 22 band ERB 能量 Ex[m]
  │
  ▼ Pitch 检测 (降采样+LPC白化+自相关+多级搜索)
  │     └→ 基频延迟帧频谱 P[k], 逐band互相关 Exp[m]
  │
  ▼ 特征提取: 45维 = DCT(log(Ex)) + DCT(Exp) + pitch_index
  │
  ▼ RNN 推理 → 22 band 增益 gains[m]
  │
  ▼ Pitch Filter: X += r·P (梳状滤波增强谐波)
  │
  ▼ Gain 平滑 + 应用: X *= g
  │
  ▼ IFFT + 合成窗 + Overlap-Add → 输出 PCM
```

### 2.2 特征设计 (45 维)

| 索引 | 维度 | 含义 | 计算方式 |
|------|------|------|----------|
| [0..21] | 22 | BFCC (频谱包络) | log10(band_energy) → DCT |
| [22..43] | 22 | Pitch 相关性 DCT | 归一化互相关(X,P) → DCT |
| [44] | 1 | 归一化基频周期 | 0.01*(pitch_index - 300) |

相比 V1 的 37 维特征（含 delta/delta2/spec_variability），V3.1 去除了手工时序特征，让 GRU 自行学习时序关系，特征更简洁信息量更大。

### 2.3 模型结构 (V3.1, 9351 参数)

```
Conv1d(45→16, k=3, tanh)          2,160 参数
  ↓
Conv2d(16→16, k=3, tanh)          768 参数    ← 扩展感受野至5帧
  ↓
GRU1(16→16)                       1,536 参数
  ↓
GRU2(16→16)                       1,536 参数
  ↓
GRU3(16→16)                       1,536 参数
  ↓
Concat([conv2, gru1, gru2, gru3]) = 64维
  ↓
Dense(64→22, sigmoid) → gains     1,430 参数
Dense(64→1, sigmoid)  → VAD       65 参数
                                  ─────────
                                  总计 9,351
```

核心设计原则：
- 所有维度为 16 的倍数，NEON SIMD 零尾循环
- Skip connection: 所有层输出直达 Dense，信息无损传递
- Int8 量化 GRU 权重（scale=1/256），Conv/Dense 保持 float

### 2.4 关键 DSP 模块原理

**三角窗 Band Energy**：相邻 band 之间用线性权重 (1-frac)/frac 分配功率，形成三角形滤波器组，避免 band 边界增益跳变。

**Pitch 检测**：4x 降采样 + LPC 白化 + 粗搜自相关 + 精搜微调 + 倍频修正。LPC 白化基于源-滤波器模型，移除声道共振峰干扰，让自相关峰只对应真正基频。

**Pitch Filter**：频域梳状滤波器 `X += r·P`。在基频整数倍频率处增强（相位一致叠加），谐波间衰减（相位随机抵消）。混合比例 r 由 pitch 相关性和 RNN 增益共同决定。

**1 帧延迟 Look-ahead**：当前帧特征推理出的 gain 应用到上一帧频谱（delayed_X），让 RNN 能看到"下一帧发生了什么"，改善语音起止瞬态。

---

## 三、实现过程

### 3.1 训练阶段

1. **数据生成**：用 C 程序 (`dump_features_16k.c`) 从语音/噪声/RIR wav 三路混合生成训练数据（Viterbi VAD、A 加权 SNR、随机频响/削波/量化增强），每 sequence 2000 帧
2. **模型训练**：PyTorch，AdamW，感知 loss `(pred^0.25 - target^0.25)²`，LambdaLR 持续衰减，GRU 正交初始化，skip connection 架构
3. **知识蒸馏**：DFNet/GTCRN 大模型输出的 teacher gain 作为辅助监督
4. **多版本迭代**：V3(gru16, 8.5K) → V3.1(+conv2 扩展感受野, 9.3K, PESQ 1.959) → V4(gru32, 25.7K, 稀疏化)
5. **权重导出**：`dump_rnnoise_weights_16k.py` / `dump_rnnoise_weights_wexchange.py` 生成 int8 静态数组 `.c` 文件（兼容 xiph LinearLayer 格式，scale=1/256）

### 3.2 性能优化阶段

1. **FFT 替换**：自研 Stockham 混合基 320pt FFT (4×4×4×5) 替代 kiss_fft，为 NEON 向量化铺路
2. **Pitch 搜索 NEON**：`xcorr_kernel` / `celt_inner_prod` / `dual_inner_prod` 使用 `vmlaq_f32` 4-wide FMA，宏 `PITCH_XCORR_NEON` 控制开关，**BIT-EXACT**
3. **GRU 推理 NEON**：xiph `vec.h` 的 `cgemv8x4` 向量化矩阵乘，int8 权重 × float 输入混合精度
4. **内存零分配**：所有 state 预分配在 `YS_Rnnoise_Create` 时的静态 buffer 中，pipeline 内部 malloc 约 4KB

### 3.3 工程集成阶段

1. 保持 `YS_RNN_Interface.h` 四个接口函数签名不变（`Get_memsize` / `Create` / `Process_Frame` / `GetVersion`）
2. `rnnoise_16k_pipeline.c` 封装完整 pipeline（HP → 特征 → RNN → pitch_filter → gain_smooth → apply → OLA）
3. VAD 模块 (`YS_VAD_RNN_Interface`) 完全不动，降噪模块独立替换
4. 编译脚本适配多平台交叉编译（mstar8627DE, Edge10Max 等），ARM 静态链接
5. qemu-arm 验证流程：交叉编译 → qemu 执行 → 对比 x86 输出 SNR > 45dB

### 3.4 V3.1 性能实测（x86 各模块耗时）

| 模块 | 耗时/帧 | 占比 |
|------|---------|------|
| HP filter | 0.9 µs | 4.4% |
| 特征提取 (FFT+pitch+BFCC) | 13.4 µs | 66.3% |
| RNN 前传 (int8 GRU) | 1.9 µs | 9.3% |
| Pitch filter | 1.2 µs | 5.7% |
| Gain 应用 | 0.3 µs | 1.7% |
| OLA 合成 | 2.6 µs | 12.7% |
| **总计** | **20.3 µs** | 100% |

### 3.5 V3.1 ARM NEON 实测（qemu Cortex-A7）

| 模块 | V3.1 耗时 | aoplay 耗时 | 加速比 |
|------|-----------|-------------|--------|
| HP filter | 5.0 µs | 12.4 µs | 2.5x |
| 特征提取 | 242.7 µs | 270.8 µs | 1.1x |
| NN 推理 | 87.1 µs | 182.1 µs | 2.1x |
| Pitch filter | 22.8 µs | — (无) | — |
| Gain + OLA | 58.1 µs | 65.8 µs | 1.1x |
| **总计** | **415.7 µs** | **531.2 µs** | **1.28x** |

---

## 四、克服的困难与解决方案

### 4.1 FFT 集成：kiss_fft 隐含 1/N 缩放

**问题**：替换为自研 Stockham FFT 后，输出 SNR = -41dB，降噪完全失效。  
**根因**：kiss_fft 在 forward FFT 内部对输出除以了 N=320，而标准 FFT 不除 N。频域能量相差 N²=102400 倍，RNN 特征值域完全偏移。  
**解决**：forward 输出手动除 N 对齐 kiss_fft 行为，inverse 相应不再额外缩放。验证两版输出 SNR > 52dB。

### 4.2 训练数据 NaN：RIR wav 文件直接 fread(float)

**问题**：使用 RIR 训练时频繁出现 loss=NaN。  
**根因**：RIR 文件是 wav 格式（含 44B header + int16 采样），C 代码 `load_rir()` 用 `fread(float*)` 直接读取，wav header 被误解释为极大 float32 值，卷积后信号溢出。  
**解决**：编写 `convert_rir_wav_to_f32.py` 预处理所有 RIR 为 raw float32 格式；增加 `USE_RIR` 编译开关方便排查。

### 4.3 增益平滑策略：xiph 能量补偿在小模型上失效

**问题**：对齐 xiph 的 gain smoothing（floor + 能量补偿）后 PESQ 反而下降 0.04。  
**根因**：语音起振帧 `Ex_cur >> Ex_prev`，能量补偿把 lastg 压低导致 floor 失效。xiph 大模型（5 帧感受野）增益平稳不依赖 floor；V3.1 小模型（3 帧感受野）增益本就偏抖，floor 是主要平滑来源。  
**解决**：经 50 条 DNS2020 A/B 实测，只保留 floor（`max(g, 0.6*lastg)`），去掉能量补偿。

### 4.4 服务器训练评估不一致

**问题**：训练服务器报 PESQ ~1.0-1.2，本地评估 ~1.85，差距巨大导致无法在服务器上选 best model。  
**根因**：服务器无 C 共享库，fallback 到 Python 特征提取（无 pitch filter），与 C 推理路径不一致。  
**解决**：`run_server.sh` 启动时自动编译 `libdenoise16k.so`，训练评估优先加载 C 库，保证训练/部署口径一致。

### 4.5 Stockham FFT 索引推导错误

**问题**：Stockham DIT 公式从论文到代码翻译出错，N=16 即失败。  
**解决**：先用 Python (numpy) 实现完整算法，逐级打印中间值与 numpy.fft 对比验证，再逐行对齐 C。Radix-5 蝶形 y[2]/y[3] 互换问题也通过独立单元测试定位。

---

## 五、答辩可能问到的问题与回答

### Q1: 为什么选 RNNoise 架构而不是 DCCRN/Conv-TasNet 等端到端模型？

RNNoise 架构（手工特征 + 轻量 RNN + 频域增益）有几个在嵌入式场景不可替代的优势：
- **极低参数量**：9K 参数 vs DCCRN 的数百 K，内存/Flash 占用差一个数量级
- **确定性推理耗时**：纯 GRU 前向无动态卷积/attention，每帧计算量固定
- **频域 gain mask 物理可解释**：每个 band 的增益有明确含义，便于调试和后处理
- **无 encoder-decoder 延迟**：只有 1 帧 look-ahead (10ms)，而端到端模型通常需要 32-64ms

### Q2: 三角窗 band energy 相比直接硬切割有什么好处？

硬切割在 band 边界处会产生增益阶跃（相邻 band 增益不同时），听感上表现为"金属感"伪影。三角窗让每个 FFT bin 的功率按位置比例分配给相邻两个 band，保证后续 `interp_band_gain` 做逆插值时频率响应平滑连续。本质上是让 band 级增益控制退化为"软控制"，减少频谱不连续。

### Q3: Pitch filter 什么时候不应该使用？

当 pitch 检测不可靠时（噪声段、非语音、pitch_correlation 很低），pitch filter 的混合比例 r 会趋近 0，自动关闭。代码中 `if (Exp[i] > g[i]) r=1; else r = f(Exp, g)` 的逻辑确保了只在"有明确谐波结构且噪声较强"时才混合 pitch 信号。如果盲目使用会引入"回声感"（把噪声段的无关信号叠加进来）。

### Q4: 为什么 LPC 白化用 4 阶而不是 10-16 阶？

这里的目的不是精确建模声道（那需要 10-16 阶），而只是粗略去除频谱包络大趋势，让自相关的基频峰从共振峰旁瓣中凸出来。4 阶意味着只有 5 次 MAC/样本，计算极低。且这是在已 2x 降采样后的信号上做的（只需覆盖 0-4kHz），4 阶足够描述 1-2 个主要共振峰。

### Q5: 1 帧延迟 look-ahead 的代价和收益分别是什么？

代价：10ms 额外算法延迟（加上编解码器延迟，总端到端延迟增加 10ms）。  
收益：RNN 在做帧 N-1 的增益决策时能看到帧 N 的特征，对语音起止瞬态判断更准确——避免语音开头被误杀、语音结尾噪声残留。在实时通信场景中 10ms 人耳无感。

### Q6: 模型只有 9K 参数，如何保证降噪效果？

三个关键设计：
1. **特征层已做了大量信号处理**（ERB 分带、DCT 去相关、pitch correlation），RNN 拿到的是高度压缩的信息
2. **Skip connection** 让所有层输出直达 Dense，避免信息在串行 GRU 中衰减
3. **训练策略对齐 xiph**：感知 loss (γ=0.25)、Viterbi VAD、数据增强（RIR/削波/量化/三路混合）让小模型每个参数都学到最大化信息

### Q7: int8 量化 GRU 精度损失如何？

实测 int8 GRU (scale=1/256) 与 float32 GRU 输出 cosine similarity > 0.999，PESQ 差异 < 0.01。原因：GRU 的 sigmoid/tanh 激活值域在 [-1,1]，int8 的 256 级量化在这个范围内精度足够。关键是 bias 保持 float32 不量化。

### Q8: 为什么特征提取占 66% 耗时，有优化空间吗？

主要来自 pitch 自相关搜索（256 个延迟 × 160 采样的内积）和 320pt FFT。优化方向：
- Pitch 搜索 NEON 4 路展开（已做，ARM 加速约 3x）
- Ne10 Stockham FFT NEON 向量化（已集成框架，待 NEON butterfly 实现）
- 缩小 pitch 搜索范围（如根据上一帧 pitch 做 ±20% 局部搜索）

### Q9: 与 xiph 官方 rnnoise (3.5M 参数) 相比差距多少？

DNS2020 测试集上：xiph PESQ=2.012, V3.1 PESQ=1.959, 差距仅 0.053。考虑到参数量差 375 倍（3.5M vs 9.3K），这个差距完全合理。V3.1 在高 SNR 场景（≥13dB）甚至接近或反超 xiph，说明小模型在简单场景下效率更高、语音失真更少。

### Q10: 项目的创新点在哪里？

1. **架构适配**：将 xiph v0.2 的 48kHz/32band/3.5M 架构成功压缩到 16kHz/22band/9.3K，保留了 skip connection、Conv1d 前端等核心设计
2. **NEON 全对齐设计**：所有维度为 16 倍数，推理零尾循环
3. **自研 Stockham FFT**：320pt 混合基（4×4×4×5），为 NEON 向量化铺路
4. **训练-部署一致性**：C 共享库同时服务于训练评估和板端推理，消除 Python/C 特征不一致问题
5. **工程化闭环**：从训练数据生成、模型训练、权重导出、C 推理引擎到 VQE 集成，全链路打通

---

## 六、项目总结

本项目完成了从学术理论（源-滤波器模型、ERB 感知频带、梳状滤波器）到工程落地（NEON 优化、int8 量化、VQE 集成）的全链路工作。V3.1 最终在 9,351 参数约束下实现了 PESQ 1.959 的降噪效果（xiph 3.5M 参数为 2.012，差距仅 0.053），相比旧版 aoplay 提升 +0.36；ARM NEON 上每帧 416µs（实际硬件估计 30-35µs），NN 推理加速 2.1x，已通过 qemu 验证和接口适配，具备产品化部署条件。

### 版本演进路线

```
V1 (aoplay, 16.9K, PESQ~1.6)
 → 问题：降噪弱、特征设计老旧(delta/spec_variability)
 
V3 (8.5K, PESQ 1.89)
 → 基于 xiph v0.2 架构适配 16kHz, Conv1d+3×GRU(16)+skip

V3.1 (9.3K, PESQ 1.959) ← 当前发布版本
 → +Conv2 扩展感受野(3→5帧)
 → +Pitch filter (梳状滤波增强谐波)
 → +1帧 delayed look-ahead
 → +Gain smoothing (α=0.6 floor only)
 → +Int8 GRU 量化 (SNR 32~46dB, 听感无损)
 → +NEON pitch xcorr 优化 (BIT-EXACT)
 → +Stockham FFT 320pt (为 NEON butterfly 铺路)
 → 训练-评估一致性: C 共享库服务训练/部署双路径
```


---

## 七、语音谐波补偿策略：基于 Exp 的 VAD 检测与频域 Comb 补偿

### 7.1 背景与动机

RNNoise 在低信噪比场景下，RNN 输出的 gain 会将语音帧中低 SNR 的 band 大幅压低，导致语音谐波结构被部分抹除，听感上表现为"语音发闷/失真"。xiph 原版通过 gain apply 前的 pitch filter（梳状滤波器）增强谐波来缓解，但我们的 V3.1 小模型感受野仅 3-5 帧，gain 波动更大，仅靠 gain 前的 pitch filter 不够。

核心思路：在 gain apply 之后，利用 pitch 延迟帧 P 中保留的谐波信息，**有选择地补回被 gain 过度压制的谐波成分**。关键在于"有选择地"——需要一个可靠的帧级 VAD 和逐 band 的谐波置信度来门控补偿，避免在噪声帧混入噪声。

### 7.2 Exp[b] 的计算原理

Exp[b] 是 rnnoise 特征提取阶段（`rnn_compute_frame_features`）里已有的计算产物，表示当前帧频谱 X 与基频延迟频谱 P 在第 b 个 band 上的归一化互相关系数。

#### 计算过程

1. **FFT**：对当前帧加窗后做 320pt FFT，得到频谱 X[k]（k=0..160）

2. **Pitch 延迟帧构造**：利用 pitch 检测得到的基频周期 T，从历史采样中取出延迟 T 个采样点的信号，同样加窗 FFT 得到 P[k]

3. **逐 band 能量与互相关**：
   ```
   Ex[b]  = Σ_{k∈band_b} |X[k]|²           // 当前帧 band 能量
   Ep[b]  = Σ_{k∈band_b} |P[k]|²           // 基频延迟帧 band 能量
   Sxp[b] = Σ_{k∈band_b} Re(X[k] · P[k]*)  // X 和 P 的实部互相关
   ```
   其中求和按三角窗加权（与 band energy 一致），band_b 覆盖的 FFT bin 由 ERB 频带表决定。

4. **归一化**：
   ```
   Exp[b] = Sxp[b] / sqrt(Ex[b] · Ep[b])
   ```
   范围 [0, 1]（负值 clamp 到 0）。

#### 物理含义

Exp[b] 本质是频域短时归一化互相关（Normalized Cross-Correlation, NCC），逐 band 计算：

- **Exp[b] ≈ 1**：该 band 的 X 和 P 高度相关——存在稳定的谐波结构。语音的基频 f0 及其倍频 2f0, 3f0, ... 会在对应 band 产生高 Exp 值。
- **Exp[b] ≈ 0**：无周期性，可能是噪声、气音、或非谐波成分。
- 不受绝对能量影响（归一化了），只反映"形状相似度"。

Exp[b] 零额外计算开销——它已经在特征提取中被计算并用于 RNN 的输入特征（DCT(Exp) 是 22 维 pitch correlation 特征的来源），我们只是复用它做 VAD 和补偿门控。

### 7.3 基于 Exp 的中低频 VAD 检测

#### 方法

不依赖 RNN 的 vad_prob 输出，而是直接从 Exp[b] 构造一个信号处理级别的 VAD 判决：

1. **选取中低频 band**：band 1~12（约 100~2000Hz），覆盖人声基频（85~300Hz）及前几个倍频谐波的主要分布区间。跳过 band 0（DC/极低频，易受风噪干扰）。

2. **计数法判决**：统计这些 band 中 `Exp[b] > 0.6` 的个数（exp_count）。阈值 0.6 表示"该 band 有明确的周期性成分"。

3. **多 band 一致性判据**：`exp_count ≥ 6`（12 个 band 中至少一半有强周期性）判为语音帧。

4. **帧级平滑（拖尾）**：
   - 语音帧：`exp_speech_flag` 立即拉到 1.0（快攻，不丢语音起始）
   - 非语音帧：每帧衰减 0.125，8 帧（80ms）归零（慢释放，避免补偿突然截断产生杂音）

#### 原理与优势

- **语音的核心声学特征是谐波结构**：浊音段有基频 f0 及其整数倍频泛音，这些会在多个 band 上同时产生高 Exp[b]。
- **计数法比均值更抗干扰**：噪声可能偶尔在某 1-2 个 band 上碰巧有周期性（如空调嗡鸣），但不会同时在 6 个以上 band 都有高 Exp。均值法容易被单个异常 band 拉高，计数法要求"多数一致"。
- **不依赖 RNN 模型**：纯信号处理特征，鲁棒性不受模型质量/训练数据影响。
- **零额外计算**：Exp[b] 已经在特征提取中算好，VAD 判决只需一个 for 循环（12 次比较 + 计数）。

#### 实测性能

在实际测试音频上：
- 纯噪声段误检率：0.4%（极少误触发）
- 语音段检测率：约 35%（含大量静音段和气音段。纯浊音段检测率更高，清辅音/气音段不检测——这是预期行为，因为清辅音没有谐波结构，不需要也不应该做谐波补偿）

### 7.4 Exp-gated 频域谐波 Comb 补偿

#### 算法

在 gain apply 之后、OLA 合成之前，当 `exp_speech_flag > 0` 时执行：

```c
// 逐 band 计算混合系数
for (b = 0; b < NB_BANDS; b++) {
    mix[b] = Exp[b] * scale * strength;
    clamp(mix[b], 0, strength);
}

// 插值到 FFT bin 级别
interp_band_gain(mixf, mix);

// 频域叠加 pitch 延迟帧的谐波
for (k = 0; k < FREQ_SIZE; k++) {
    X'[k].re += mixf[k] * P[k].re;
    X'[k].im += mixf[k] * P[k].im;
}

// 能量归一化：防止补偿后超过原始信号能量
for (b) {
    target = Ex[b];  // 上限 = 原始帧能量
    if (newE[b] > target) scale_down;
}
```

#### 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `Exp[b]` | 0~1 | 逐 band 谐波置信度（周期性越强越大） |
| `scale` (exp_speech_flag) | 0~1 | 帧级门控：语音=1，拖尾期间渐降，噪声=0 |
| `strength` | 0.3 | 总强度上限：最多混入 30% 的 P 信号 |
| 归一化 target | Ex[b] | 补偿后 band 能量不超过原始信号能量 |

#### 为什么用 Exp[b] 而不用 r[b]（xiph pitch filter 公式）

xiph 原版 pitch filter 的混合比例 r[b] 公式为：

```
r[b] = sqrt( Exp²·(1-g²) / (0.001 + g²·(1-Exp²)) ) × sqrt(Ex/Ep)
```

这个公式在 **gain apply 之前** 使用是合理的——此时 X 和 P 都是原始频谱，`sqrt(Ex/Ep)` 做能量匹配有物理意义。

但在 **gain apply 之后**，X 已经被 gain 压过，能量远小于原始：
- `sqrt(Ex/Ep)` 的 Ex 是原始能量（未压过），而此时的 X 频谱是压过的，能量不匹配
- gains 低的 band r 更大（公式设计如此——低 gain 意味着"被压了多少"），但这些 band gains 低的原因可能正是噪声占主导，r 大反而把 P 中的噪声成分混进来了

Exp[b] 的优势：
- **纯粹的周期性度量**，不含能量缩放项
- Exp[b] 高说明该 band 有谐波，低说明没有——不会因为能量差异导致噪声混入
- 物理意义清晰，调参直观（只需调 strength 一个参数）

#### 迭代过程与教训

1. **第一版**（r[b] + norm=Ex×g²）：效果不明显。原因：归一化 target 设为 `Ex×g²`（增益后能量），补进来的谐波全被归一化抹掉了。
2. **第二版**（r[b] + norm=Ex）：语音处噪声明显增加。原因：r[b] 在 gain apply 后语义不对，`sqrt(Ex/Ep)` 把 P 中的噪声成分也放大混入了。
3. **第三版**（r[b] + 无归一化）：噪声更明显。没有归一化约束，噪声混入无上限。
4. **当前版本**（Exp[b] 直接做混合 + norm=Ex）：用 Exp[b] 替代 r[b]，避免能量缩放带来的噪声引入。归一化到 Ex 确保不超过原始能量。

### 7.5 答辩 Q&A 补充

#### Q11: 为什么不直接用 RNN 输出的 vad_prob 做门控？

可以用，但有两个考虑：
1. vad_prob 的可靠性依赖模型训练——小模型 vad 输出的召回率和准确率不一定稳定
2. Exp-based VAD 是纯信号处理特征，与 RNN 模型完全解耦，鲁棒性更好
3. 实际上两者可以级联使用（如 `vad_prob > 0.5 && exp_speech_flag > 0`），目前只用 Exp VAD 是因为实测够用且更简单

#### Q12: 补偿强度 strength=0.3 是怎么定的？

经验调参。0.3 意味着即使 Exp[b]=1（最强谐波）、exp_speech_flag=1（确认有语音），最多也只混入 30% 的 P 信号。实测中：
- 0.3：语音自然度有轻微改善，不引入可感知噪声
- 0.5：改善更明显，但在某些噪声类型下有轻微可察觉的残留噪声
- 1.0：明显引入噪声（P 本身含噪）

最终 strength 需要结合实际产品场景和主观听测确定。

#### Q13: Exp-gated comb 补偿对计算量的影响？

增量很小：
- Exp VAD 判决：12 次比较 + 1 次条件判断，忽略不计
- Comb 补偿（仅语音帧执行）：22 次乘法（band）+ `interp_band_gain`（161 次插值）+ 161 次复数加法 + 能量归一化（22 次 `compute_band_energy` + 161 次乘法）
- 估计增量 ~2-3µs/帧（x86），占总帧耗时 10-15%
- 噪声帧完全跳过（`exp_speech_flag=0` 直接不进入补偿块），平均增量更低

#### Q14: 这个补偿放在 gain apply 之后而不是之前，有什么特殊考虑？

gain apply 之前已经有 xiph 原版的 pitch filter（`X += r·P`），它负责增强谐波让 RNN 看到更干净的输入。如果 gain apply 前就"补偿"，等于在 RNN 判决之前就改了输入，可能干扰 RNN 的降噪判断。

放在 gain apply 之后是"后处理"逻辑：RNN 已经做了降噪决策，我们只在确认有语音谐波的帧上，把被过度压制的成分适度补回来。这样不影响 RNN 本身的工作。

#### Q15: 如果 pitch 检测错误（比如估计的基频 T 不对），Exp[b] 会怎样？

pitch 检测错误时，P[k] 和 X[k] 不对齐，Exp[b] 会很低（接近 0），补偿自然不会触发。这是 Exp[b] 做门控的天然优势——它本身就包含了 pitch 可靠性的信息。如果基频估计不对，Exp 低，VAD 不触发，comb 不补偿，系统安全退化为纯 gain 降噪。


### 7.6 方案二：Exp + RNN vad_prob 联合门控

#### 动机

纯 Exp VAD 只能检测浊音段（有谐波结构的语音），对清辅音（/s/, /f/, /sh/ 等）、气音、弱起始音节的检测率较低——因为这些段没有周期性，Exp[b] 天然很低。而 RNN 的 vad_prob 输出是一个全局语音概率，训练时同时学习了浊音和清辅音的模式，理论上覆盖范围更广。

单独用 vad_prob 做门控的问题：vad_prob 在某些噪声类型下可能给出偏高的值（模型泛化不够），导致噪声帧误触发补偿。

因此采用联合策略：Exp VAD 和 vad_prob 取 OR（并集），各自有严格的独立门限，互补检测。

#### 实现

在原有 Exp VAD 判决之后，增加 vad_prob 的连续帧计数逻辑：

```c
// RNN vad_prob 连续高帧计数
if (vad_prob > VAD_JOINT_THRESH)       // 默认 0.9
    vad_high_count++;
else
    vad_high_count = 0;
int vad_triggered = (vad_high_count >= VAD_JOINT_MIN_FRAMES);  // 默认 2 帧

// OR 逻辑：Exp 或 vad_prob 任一触发 → 拉起
if (exp_triggered || vad_triggered)
    exp_speech_flag = 1.0;
else
    exp_speech_flag -= 0.125;  // 8帧衰减
```

#### 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `VAD_JOINT_THRESH` | 0.9 | vad_prob 触发阈值。设 0.9 是因为要求 RNN 非常确信是语音才额外触发，避免噪声误检 |
| `VAD_JOINT_MIN_FRAMES` | 2 | 连续帧要求。单帧 vad_prob 高可能是瞬态误判，要求连续 2 帧（20ms）以上才触发 |
| `ENABLE_VAD_JOINT` | 1 | 编译宏，=0 可关闭联合模式回退到纯 Exp VAD |

#### vad_prob 的来源

vad_prob 来自 V3.1 降噪模型 RNN 推理的副产物——`compute_rnn_16k()` 前传时同时输出 22 个 band gains 和 1 个 vad_prob（Dense(64→1, sigmoid)），零额外计算。不是外部独立 VAD 模块，而是降噪模型自身的语音概率判断。

#### 两条路径的互补性

| 检测路径 | 检测对象 | 优势 | 盲区 |
|----------|----------|------|------|
| Exp VAD | 浊音（有基频谐波） | 纯信号处理，不依赖模型，零额外计算 | 清辅音/气音/弱起始 |
| vad_prob | 全语音（含清辅音） | RNN 学习了完整语音模式 | 某些噪声类型可能误判 |
| 联合(OR) | 全语音 | 互补覆盖 | 两者都误判时才误触发（概率极低） |

联合门控让补偿的触发范围更完整，不仅限于浊音段。而两个路径各自的严格门限（Exp: ≥6 band 同时高，vad_prob: >0.9 连续 2 帧）确保了噪声段极低的误触发率。

#### 答辩 Q&A 补充

**Q16: vad_prob>0.9 的门限为什么这么高？**

0.9 是为了确保只有 RNN 非常确信是语音时才通过 vad_prob 路径触发补偿。sigmoid 输出 >0.9 意味着 logit > 2.2，这在噪声段很难达到。如果设 0.5（常规 VAD 阈值），在低 SNR 下噪声帧也可能误触发，导致噪声混入。0.9 + 连续 2 帧的双重约束，使误检率接近 0。

**Q17: 为什么要求连续 2 帧而不是 1 帧？**

单帧 vad_prob 峰值可能来自瞬态事件（关门声、碰撞声、噪声突变），这些不是语音。要求连续 2 帧（20ms）是语音的最小特征时长——人说话最短的音素（如爆破音释放）也至少持续 20-30ms。1 帧（10ms）的瞬态事件通不过这个滤波。

**Q18: 联合模式的计算开销？**

一次浮点比较（vad_prob > 0.9）+ 一次整数自增（vad_high_count++）+ 一次整数比较（>= 2），总计 3 条指令，完全可忽略。vad_prob 本身已在 RNN 推理时算好，不增加任何推理开销。
