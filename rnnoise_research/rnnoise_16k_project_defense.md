# RNNoise 16kHz 降噪引擎项目答辩文档

## 一、项目概述

### 1.1 项目背景

在嵌入式音视频设备（对讲门铃、摄像头）场景中，需要一个运行在 ARM Cortex-A7 级别芯片上的实时单通道语音降噪模块。原有方案（V1 aoplay，基于 Mozilla rnnoise 早期版本 16.9K 参数）降噪能力不足（PESQ 仅 1.703），无法满足产品需求。

### 1.2 项目目标

基于 xiph/rnnoise 最新架构（v0.2），设计并实现一个面向 16kHz 采样率的极小模型降噪引擎：
- 参数量 <10K，ARM 每帧推理 <120µs（10ms 帧预算的 1.2%）
- PESQ 显著优于 V1（目标 >1.85，实测达 1.89）
- 保持 `YS_RNN_Interface.h` 接口不变，可无缝替换 VQE 中旧版 NS 模块

### 1.3 最终成果

| 指标 | V1 旧版 (aoplay) | V3.1 本项目 | 提升 |
|------|:---:|:---:|:---:|
| 参数量 | 16,883 | 9,351 | -45% |
| PESQ (DNS2020) | 1.703 | 1.89 | +0.19 |
| x86 每帧推理 | 28.7µs | 18.1µs | 1.6x 加速 |
| NN 推理耗时 | 9.5µs | 1.2µs | 8.3x 加速 |
| 全维度 NEON 对齐 | ✗ | ✓ (16 倍数) | — |

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
Conv1d(45→16, k=3, tanh)          2,176 参数
  ↓
Conv2d(16→16, k=3, tanh)          784 参数    ← 扩展感受野至5帧
  ↓
GRU1(16→16)                       1,632 参数
  ↓
GRU2(16→16)                       1,632 参数
  ↓
GRU3(16→16)                       1,632 参数
  ↓
Concat([conv2, gru1, gru2, gru3]) = 64维
  ↓
Dense(64→22, sigmoid) → gains     1,430 参数
Dense(64→1, sigmoid)  → VAD       65 参数
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
4. **权重导出**：`dump_rnnoise_weights_16k.py` 生成 int8 静态数组 `.c` 文件（兼容 xiph LinearLayer 格式）

### 3.2 性能优化阶段

1. **FFT 替换**：自研 Stockham 混合基 320pt FFT 替代 kiss_fft，为后续 NEON 向量化铺路
2. **Pitch 搜索 NEON**：`pitch_xcorr` 内层循环 4 路展开 + NEON intrinsic
3. **GRU 推理 NEON**：`cgemv8x4` 向量化矩阵乘，int8 权重 × float 输入混合精度
4. **内存零分配**：所有 state 预分配在 `YS_Rnnoise_Create` 时的静态 buffer 中

### 3.3 工程集成阶段

1. 保持 `YS_RNN_Interface.h` 四个接口函数签名不变
2. `rnnoise_16k_pipeline.c` 封装完整 pipeline（特征+RNN+pitch_filter+gain+OLA）
3. 编译脚本适配多平台交叉编译（mstar8627DE, Edge10Max 等）
4. Profile 工具集成，逐模块计时输出

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

DNS2020 测试集上：xiph PESQ=2.012, V3.1 PESQ=1.89, 差距 0.12。考虑到参数量差 375 倍（3.5M vs 9.3K），这个差距完全合理。V3.1 在高 SNR 场景（≥13dB）甚至反超 xiph（因为小模型更保守，语音失真更少）。

### Q10: 项目的创新点在哪里？

1. **架构适配**：将 xiph v0.2 的 48kHz/32band/3.5M 架构成功压缩到 16kHz/22band/9.3K，保留了 skip connection、Conv1d 前端等核心设计
2. **NEON 全对齐设计**：所有维度为 16 倍数，推理零尾循环
3. **自研 Stockham FFT**：320pt 混合基（4×4×4×5），为 NEON 向量化铺路
4. **训练-部署一致性**：C 共享库同时服务于训练评估和板端推理，消除 Python/C 特征不一致问题
5. **工程化闭环**：从训练数据生成、模型训练、权重导出、C 推理引擎到 VQE 集成，全链路打通

---

## 六、项目总结

本项目完成了从学术理论（源-滤波器模型、ERB 感知频带、梳状滤波器）到工程落地（NEON 优化、int8 量化、VQE 集成）的全链路工作。在 9.3K 参数约束下实现了 PESQ 1.89 的降噪效果，相比旧版提升 +0.19，推理加速 1.6x，已具备产品化部署条件。
