# RNNoise V3.1 VQE 集成指南

## 概述

V3.1 降噪引擎（9351 参数，int8 量化 GRU，16kHz）已集成到 `rnnoise/` 目录，保持 `YS_RNN_Interface.h` 接口不变，可直接替换 VQE 项目中的旧版 NS 模块。

## 模型信息

| 项目 | 值 |
|------|-----|
| 模型 | `model_c/v3.1/best_pesq_1.959.pth` |
| 参数量 | 9351 |
| 网络结构 | Conv1d(45→16,k3) + Conv2d(16→16,k3) + 3×GRU(16) + Dense(64→22) |
| 量化 | GRU int8 + scale，Conv/Dense float |
| 采样率 | 16kHz |
| 帧长 | 160 samples (10ms) |
| NB_BANDS | 22 |
| 信号处理 | HP filter + 特征提取 + pitch filter + gain smoothing(α=0.6) + OLA |

## 性能实测

| 平台 | 每帧耗时 | 实时倍率 | 帧预算占比 |
|------|----------|---------|-----------|
| x86 AVX2 (-march=native) | 18.1 us | 552x | 0.2% |
| x86 无 SIMD (-O2) | 20.7 us | 483x | 0.2% |
| ARM NEON Cortex-A7 (qemu) | ~450 us | ~22x | 4.5% |
| ARM 真实硬件 (估计) | ~30-35 us | ~300x | 0.3% |

各模块耗时占比（x86 实测）：
- 特征提取（FFT+pitch+BFCC）: 66.3%
- OLA 合成: 12.7%
- RNN 前传: 9.3%
- Pitch filter: 5.7%
- HP filter: 4.4%
- Gain 应用: 1.7%

## 接口调用方式

接口与旧版完全一致（`YS_RNN_Interface.h` 不变）：

```c
#include "YS_RNN_Interface.h"

// 1. 获取内存大小
int memsize = YS_Rnnoise_Get_memsize();

// 2. 分配内存 + 创建
unsigned char *buf = (unsigned char *)calloc(1, memsize);
YS_RNN_ParamIn param_in = {0};
param_in.sample_rate = 16000;
param_in.frame_len = 160;
param_in.mode = 0;        // 0=对讲, 1=预览 (V3.1 统一模型，mode 暂不区分)
param_in.howl_flag = 0;   // 暂不使用
void *handle = NULL;
YS_Rnnoise_Create(&param_in, buf, &handle);

// 3. 逐帧处理 (每帧 160 samples = 10ms)
short in_pcm[160], out_pcm[160];
YS_RNN_ParamOut param_out = { .data_out_pcm = out_pcm };
param_in.data_in_pcm = in_pcm;
int vad_p = YS_Rnnoise_Process_Frame(&param_in, &param_out, (unsigned char *)handle);
// vad_p: VAD 概率 * 32768

// 4. 获取版本号
int ver = YS_RNN_GetVersion();  // 0x0C20350C (v3.1.0, 2026-08-12)
```

## 需要的源文件

集成到 VQE 的 `ns/` 模块时，需要以下文件（全部在 `rnnoise/` 目录下）：

### 核心文件（必须）
| 文件 | 说明 |
|------|------|
| `YS_RNN_Interface.c` | 对外接口适配层 |
| `YS_RNN_Interface.h` | 公共头文件（不改） |
| `rnn_16k.c` / `rnn_16k.h` | 网络前传 |
| `rnnoise_16k_pipeline.c` / `rnnoise_16k_pipeline.h` | 完整 pipeline |
| `rnnoise_data_16k.c` / `rnnoise_data_16k.h` | 权重数据（int8） |
| `denoise_16k.c` / `denoise_16k.h` | 特征提取 + OLA 合成 |
| `rnnoise_tables_16k.c` | ERB band 表等常量 |

### VAD 模块（保持不动，原样编译）
| 文件 | 说明 |
|------|------|
| `YS_VAD_RNN_Interface.c` / `YS_VAD_RNN_Interface.h` | VAD 对外接口 |
| `rnn_vad.c` / `rnn_vad.h` | VAD 网络前传 |
| `rnn.c` / `rnn.h` | VAD GRU/Dense 计算 |
| `rnn_data.c` / `rnn_data.h` / `rnn_data_ns.h` | VAD 权重数据 |
| `kiss_fft_vad.c` / `kiss_fft_vad.h` / `_kiss_fft_guts_vad.h` | VAD 专用 FFT（与降噪 kiss_fft 不冲突） |

### xiph nnet 框架（必须）
| 文件 | 说明 |
|------|------|
| `nnet.c` / `nnet.h` | GRU/Dense/Conv 通用计算 |
| `nnet_default.c` / `nnet_arch.h` | 默认标量实现 |
| `parse_lpcnet_weights.c` | 权重加载 |

### 信号处理（必须）
| 文件 | 说明 |
|------|------|
| `kiss_fft.c` / `kiss_fft.h` / `_kiss_fft_guts.h` | 320pt FFT (16kHz) |
| `pitch.c` / `pitch.h` | pitch 搜索 |
| `celt_lpc.c` / `celt_lpc.h` | LPC (pitch 依赖) |

### 公共头文件
| 文件 | 说明 |
|------|------|
| `common.h` / `arch.h` / `opus_types.h` | 基础类型定义 |
| `denoise.h` | shim，指向 `denoise_16k.h` |

### NEON 优化（ARM 平台，从 xiph_rnnoise/src/ 引用）
| 文件 | 说明 |
|------|------|
| `vec.h` | GEMV 实现（含 NEON 分支） |
| `x86/nnet_sse4_1.c` | x86 SSE4.1（可选） |
| `x86/nnet_avx2.c` | x86 AVX2（可选） |

## 编译命令

### ARM 交叉编译（板端部署）
```bash
arm-linux-gnueabihf-gcc -O2 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard \
    -I<rnnoise_dir> -I<xiph_src_dir> \
    -DRTCD_ARCH=c -DDISABLE_DEBUG_FLOAT \
    -static -o rnnoise_v31_demo \
    demo.c YS_RNN_Interface.c rnn_16k.c rnnoise_16k_pipeline.c \
    rnnoise_data_16k.c denoise_16k.c rnnoise_tables_16k.c \
    nnet.c nnet_default.c parse_lpcnet_weights.c \
    kiss_fft.c pitch.c celt_lpc.c -lm
```

### x86 编译（本地测试）
```bash
gcc -O2 -march=native \
    -I<rnnoise_dir> -I<xiph_src_dir> \
    -DRTCD_ARCH=c -DDISABLE_DEBUG_FLOAT \
    -o rnnoise_v31_demo \
    demo.c YS_RNN_Interface.c rnn_16k.c rnnoise_16k_pipeline.c \
    rnnoise_data_16k.c denoise_16k.c rnnoise_tables_16k.c \
    nnet.c nnet_default.c parse_lpcnet_weights.c \
    kiss_fft.c pitch.c celt_lpc.c -lm
```

### 关键编译宏
| 宏 | 作用 |
|-----|------|
| `-DRTCD_ARCH=c` | 运行时 CPU 分发基准架构 |
| `-DDISABLE_DEBUG_FLOAT` | 去除 float 权重副本，强制走 int8 路径 |
| `-D__ARM_NEON__` | ARM 编译器自动定义，启用 NEON GEMV |
| `-march=native` | x86 启用 AVX2/SSE4.1 |

## qemu-arm 验证流程

```bash
# 交叉编译 (静态链接)
arm-linux-gnueabihf-gcc ... -static -o demo_arm

# qemu 模拟执行
qemu-arm ./demo_arm input.pcm output.pcm 0 1

# 对比 x86 输出验证正确性
# 期望 SNR > 30dB (实测 40~50dB)
```

## 权重更新流程

当训练出新模型后，更新权重：
```bash
cd training/rnnoise_16k/
python3 dump_rnnoise_weights_wexchange.py <model.pth> inference/ --quantize

# 拷贝到集成目录
cp inference/rnnoise_data_16k.c /path/to/vqe/ns/rnnoise_data_16k.c
cp inference/rnnoise_data_16k.h /path/to/vqe/ns/rnnoise_data_16k.h

# 重编译即可，其他文件不变
```

## 与旧版对比

| 项目 | 旧版 v0.1 | V3.1 |
|------|-----------|------|
| 采样率 | 16kHz (8kHz 训练) | 16kHz |
| 参数 | ~50K | 9351 |
| GRU 结构 | Dense(42→96)+GRU(96) | Conv×2+GRU(16)×3 |
| 量化 | int8 GEMV | int8 GRU + float Conv |
| pitch filter | 无 | 有 (xiph 式) |
| gain smoothing | α=0.95 IIR | α=0.6 floor |
| 1 帧延迟 | 无 | 有 (对齐 xiph) |
| PESQ (50条DNS2020) | ~1.6 | 1.959 |
| NEON | 仅 GEMV | GEMV + pitch 循环展开 |

## 注意事项

1. **VAD 模块不动**：`YS_VAD_RNN_Interface.c/h` 和 `rnn_vad.c` 保持原样，V3.1 只替换降噪
2. **kiss_fft 不冲突**：VAD 用 `kiss_fft_vad.c/h`，V3.1 用 `kiss_fft.c/h`，符号不冲突
3. **howl_flag**：当前未使用，直接忽略。后续可加入：当 howl_flag≥2 时 gain 平方加强抑制
4. **mode**：V3.1 只有一个模型，对讲/预览统一。如需差异化，训练两套权重替换 `rnnoise_data_16k.c`
5. **延迟**：V3.1 有 1 帧（10ms）算法延迟，比旧版多 10ms。可通过 `#define RNNOISE_16K_USE_DELAY 0` 关闭
6. **内存**：`YS_Rnnoise_Get_memsize()` 返回 1152 bytes（handle），pipeline 内部 malloc 约 4KB


## 模型参数量与计算量对比

### 网络结构对比

| 层 | aoplay (旧-对讲) | aitalk (旧-预览) | V3.1 |
|----|-----------------|-----------------|------|
| input | Dense(38→16) | Dense(38→32) | Conv1d(45→16,k=3) |
| conv2 | — | — | Conv1d(16→16,k=3) |
| vad_gru | GRU(16→16) | GRU(32→32) | — (内置于dense_out) |
| noise_gru | GRU(70→16) | GRU(102→32) | — |
| denoise_gru | GRU(70→32) | GRU(102→32) | 3×GRU(16→16) |
| output | Dense(32→18) | Dense(32→18) | Dense(64→22) + Dense(64→1) |
| NB_BANDS | 18 | 18 | 22 |

### 参数量对比

| 层 | aoplay | aitalk | V3.1 |
|----|--------|--------|------|
| input_dense/conv1 | 624 | 1,248 | 2,160 |
| conv2 | — | — | 768 |
| vad_gru | 1,632 | 6,336 | — |
| noise_gru | 4,224 | 13,056 | — |
| denoise_gru | 9,984 | 13,056 | 4,608 (3×GRU) |
| output | 594+17=611 | 594+33=627 | 1,430+65=1,495 |
| **总计** | **17,075** | **34,323** | **9,351** |

### 计算量对比 (MACs/帧)

| 模块 | aoplay | aitalk | V3.1 |
|------|--------|--------|------|
| Dense/Conv | 608 | 1,216 | 2,160+768=2,928 |
| GRU (核心) | 15,744 | 47,232 | 4,608 |
| Output Dense | 576 | 576 | 1,408+64=1,472 |
| **总 NN MACs** | **~16,928** | **~49,024** | **~9,008** |
| **相对V3.1** | 1.9x | 5.4x | 1.0x |

### 实测性能 (x86, -O2, int8 GRU)

V3.1 各模块耗时分布：

| 模块 | 耗时/帧 | 占比 |
|------|---------|------|
| HP filter | 0.9 us | 4.4% |
| 特征提取 (FFT+pitch+BFCC) | 13.4 us | 66.3% |
| RNN 前传 | 1.9 us | 9.3% |
| Pitch filter | 1.2 us | 5.7% |
| Gain 应用 | 0.3 us | 1.7% |
| OLA 合成 | 2.6 us | 12.7% |
| **总计** | **20.3 us** | 100% |

aoplay (旧版) 各模块耗时分布 (x86, -O2, float32)：

| 模块 | 耗时/帧 | 占比 |
|------|---------|------|
| HP filter | 1.4 us | 4.7% |
| 特征提取 (FFT+pitch+BFCC) | 16.3 us | 54.2% |
| NN 推理 | 9.2 us | 30.7% |
| 增益应用+插值 | 0.4 us | 1.3% |
| OLA 合成 | 2.7 us | 9.1% |
| **总计** | **30.1 us** | 100% |

V3.1 vs aoplay 对比：
- 总耗时：20.3 vs 30.1 us（V3.1 快 33%）
- NN 推理：1.9 vs 9.2 us（V3.1 快 4.8x，int8 量化 + 小 GRU 的效果）
- 特征提取：13.4 vs 16.3 us（V3.1 稍快，得益于 xiph 优化的 pitch 实现）
- 注：aoplay 无 pitch filter（旧版注释掉了），V3.1 有 pitch filter 但总耗时仍更低

### 平台性能汇总

| 平台 | V3.1 每帧耗时 | 实时倍率 | 帧预算占比 |
|------|-------------|---------|-----------|
| x86 AVX2 (-march=native) | 18.1 us | 552x | 0.18% |
| x86 标量 (-O2) | 20.7 us | 483x | 0.21% |
| ARM NEON Cortex-A7 (qemu) | 435 us | 23x | 4.3% |
| ARM 真实硬件 (估计) | ~30-35 us | ~300x | 0.3% |

### qemu-arm 对比 (howling_test_ns-2.pcm, Cortex-A7 NEON)

| 模块 | aoplay (float+NEON) | V3.1 (int8+NEON) 估算 |
|------|--------------------|-----------------------|
| HP filter | 12.4 us (2.3%) | ~19 us (4.4%) |
| 特征提取 (FFT+pitch) | 270.8 us (51.0%) | ~288 us (66.3%) |
| NN 推理 | 182.1 us (34.3%) | ~40 us (9.3%) |
| Pitch filter | — (注释掉了) | ~25 us (5.7%) |
| 增益应用 | 7.8 us (1.5%) | ~7 us (1.7%) |
| OLA 合成 | 58.0 us (10.9%) | ~55 us (12.7%) |
| **总计** | **537 us** | **435 us** |

关键结论：
- V3.1 在 ARM NEON 上比 aoplay 快约 19%（435 vs 537 us）
- NN 推理部分 V3.1 快 4.5x（~40 vs 182 us），得益于 int8 量化 + 小 GRU
- 但特征提取 V3.1 稍重（288 vs 271），因为 320pt FFT（vs aoplay 160pt）+ 完整 pitch search
- V3.1 额外有 pitch filter（~25us），aoplay 无
- 总体省出的 NN 计算量被更重的信号处理部分消耗掉大部分，最终差距约 19%

### int8 量化精度验证

| 文件 | max_diff (int16) | rms_diff | SNR (float vs int8) |
|------|-----------------|----------|---------------------|
| CB30_nearin | 74 | 1.37 | 39.2 dB |
| s20_nearin | 469 | 17.11 | 40.2 dB |
| s20_howling | 309 | 5.77 | 32.4 dB |
| howling-0611 | 115 | 2.23 | 46.1 dB |
| howling_test_ns-2 | 29 | 1.07 | 43.6 dB |

结论：int8 量化几乎无损（SNR 32~46dB），听感无差别。

### NEON 正确性验证 (x86 vs ARM qemu)

| 指标 | 结果 |
|------|------|
| SNR (x86 vs ARM) | 45.8 dB |
| Max sample diff | 178 |
| 判定 | PASS (negligible) |

差异来自 int8 GEMV 不同 SIMD 实现的浮点累加顺序差异，数学结果等价。


## V3.1 ARM NEON 实测详细分析 (qemu-arm, Cortex-A7)

### 测试条件
- 音频: `howling_test_ns-2.pcm` (883 帧, 8.83s)
- 平台: qemu-arm 模拟 Cortex-A7 + NEON VFPv4
- V3.1: int8 GRU + NEON GEMV, best_pesq_1.959.pth
- aoplay: float + NEON sgemv_accum16 (原始方式, 不带 OPT_INT8_GEMV)

### V3.1 各模块实测耗时 (ARM NEON qemu)

| 模块 | V3.1 耗时/帧 | 占比 |
|------|-------------|------|
| HP filter | 5.0 us | 1.2% |
| 特征提取 (FFT+pitch+BFCC) | 242.7 us | 58.4% |
| RNN 推理 (int8 NEON) | 87.1 us | 20.9% |
| Pitch filter | 22.8 us | 5.5% |
| Gain 应用 | 6.7 us | 1.6% |
| OLA 合成 | 51.4 us | 12.4% |
| **总计** | **415.7 us** | 100% |

### aoplay vs V3.1 ARM NEON 对比

| 模块 | aoplay (float+NEON) | V3.1 (int8+NEON) | 差异 |
|------|--------------------|--------------------|------|
| HP filter | 12.4 us | 5.0 us | V3.1 快 2.5x |
| 特征提取 | 270.8 us | 242.7 us | V3.1 快 10% |
| NN 推理 | 182.1 us | 87.1 us | **V3.1 快 2.1x** |
| Pitch filter | — (无) | 22.8 us | V3.1 额外增加 |
| Gain 应用 | 7.8 us | 6.7 us | 相当 |
| OLA 合成 | 58.0 us | 51.4 us | V3.1 快 11% |
| **总计** | **531.2 us** | **415.7 us** | **V3.1 快 22%** |

### 分析结论

1. V3.1 在 ARM NEON 上整体比 aoplay 快 22%（416 vs 531 us/帧）
2. NN 推理部分快 2.1 倍（87 vs 182 us），得益于更小的 GRU(16) + int8 量化
3. 特征提取是主瓶颈（占 58%），V3.1 用 320pt FFT（vs aoplay 160pt）但 xiph 实现稍优
4. V3.1 额外有 pitch filter（23us），aoplay 无——但总耗时仍更低
5. 实际 ARM 硬件上差距可能更小（qemu 不模拟 NEON 并行加速效果）

## Pitch NEON 优化 (PITCH_XCORR_NEON)

### 实现

在 `rnnoise/pitch.h` 中增加了 pitch 自相关搜索的 NEON 优化实现，通过宏 `PITCH_XCORR_NEON` 控制开关。

优化的函数：
- `xcorr_kernel()`: 4 路滑动点积，使用 `vmlaq_f32` 4-wide FMA
- `celt_inner_prod()`: 内积，使用 NEON 4-wide 累加 + 水平求和
- `dual_inner_prod()`: 双内积，使用 NEON 双累加器

### 使用方式

```bash
# 启用 pitch NEON 优化
-DPITCH_XCORR_NEON

# 不加此宏则走原始标量路径（完全向后兼容）
```

### 验证结果

- 正确性: **BIT-EXACT**（优化前后输出完全一致）
- qemu 性能: 无法体现加速（qemu 逐指令翻译，不模拟 SIMD 并行）
- **实际板端测试: pitch 搜索部分提升约 12%**

### 后续优化方向

| 优化方案 | 预期节省 | 难度 | 是否影响算法 |
|---------|---------|------|------------|
| Pitch xcorr NEON（已实现） | ~60-70 us | 低 | 无（BIT-EXACT） |
| FFT 替换 Ne10/CMSIS-DSP | ~50 us | 中 | 无 |
| 去掉 pitch filter | ~23 us + 间接省 pitch 搜索 | 低 | PESQ -0.04 |
| 去掉 1 帧延迟 | 无性能变化，减少 10ms 延迟 | 低 | 轻微 |


## 扩容方案与开源 0.2 逐层结构对比

为改善 V3.1 效果，评估「扩大建模能力 + 启用稀疏训练」。目标：ARM NEON 板端速度与 aoplay 旧版（531 us/帧）持平、可略多。以下把候选扩容配置 **gru64/3层/sparse66.6%** 与开源 rnnoise 0.2（cond=128, gru=384）逐层对比。两者结构完全同构：

```
Conv1(in→cond,k3,tanh) → Conv2(cond→gru,k3,tanh) →
GRU1(gru) → GRU2(gru) → GRU3(gru) →
cat[conv2_out, gru1, gru2, gru3] = 4*gru →
Dense_out(4*gru→22) + VAD_dense(4*gru→1)
```

### 结构尺寸对照

| 层 | V3.1 现状 | gru64 扩容 | 开源 0.2 |
|----|-----------|-----------|---------|
| Conv1 | 45→16, k3（单conv） | 45→32, k3 | 45→128, k3 |
| Conv2 | 无 | 32→64, k3 | 128→384, k3 |
| GRU×3 | 16→16 | 64→64 | 384→384 |
| cat_size | 4×16=64 | 4×64=256 | 4×384=1536 |
| Dense_out | 64→22 | 256→22 | 1536→22 |
| VAD_dense | 64→1 | 256→1 | 1536→1 |
| 时间感受野 | 3 帧 | 5 帧（two_conv） | 5 帧 |

### 逐层参数量 / 计算量对比（稠密）

| 模块 | gru64 参数 | gru64 MAC/帧 | 开源0.2 参数 | 开源0.2 MAC/帧 | 开源/gru64(参数) |
|------|-----------|-------------|-------------|---------------|-----------------|
| Conv1 | 4,352 | 4,320 | 17,408 | 17,280 | 4.0x |
| Conv2 | 6,208 | 6,144 | 147,840 | 147,456 | 23.8x |
| GRU×3 | 74,880 | 73,728 | 2,661,120 | 2,654,208 | 35.5x |
| Dense_out | 5,654 | 5,632 | 33,814 | 33,792 | 6.0x |
| VAD_dense | 257 | 256 | 1,537 | 1,536 | 6.0x |
| **总计** | **91,351** | **90,080** | **2,861,719** | **2,854,272** | **31.3x** |

（参考：V3.1 现状 8,567 参数 / 8,240 MAC / 0.82 MMAC/s）

### 逐层分析

- **Conv1（前端特征压缩）**：差距最小（4x）。输入是固定的 45 维特征，信息量有限，32 通道已能承载大部分，这层不是瓶颈，无需盲目放大。
- **Conv2（升维到 GRU 宽度）**：差距 23.8x。代价 = cond×gru×k，随 GRU 宽度线性放大。
- **GRU×3（核心时序建模）**：差距最大（35.5x），是两者参数的绝对主体（开源占 93%，gru64 占 82%）。GRU 参数≈2×hid²，宽度 64 vs 384 = 6 倍 → 参数 6²≈36 倍。**这是建模能力差距的核心，也是缩小差距的唯一有效杠杆。**
- **Dense_out / VAD_dense（输出头）**：差距 6x，即 cat_size 256 vs 1536 的线性比；输出固定 22 band + 1 vad。

### 稀疏（66.6%）后有效计算对比

稀疏（SPARSE_PARAMS_V4，8×4 块，对应 cgemv8x4 NEON 内核）只作用于 GRU 权重，conv/dense 保持稠密。

| 指标 | gru64 | 开源 0.2 |
|------|-------|---------|
| GRU 稠密 MAC/帧 | 73,728 | 2,654,208 |
| GRU 稀疏后有效 MAC/帧 | 24,625 | 886,505 |
| 全模型有效 MAC/帧 | 40,977 | 1,086,569 |
| 有效算力 @100帧/s | 4.1 MMAC/s | 108.7 MMAC/s |

稀疏后总有效算力差距从 31.7x 缩小到 **26.5x**（conv/dense 不稀疏，在小模型里占比相对更大）。

### 速度预算复核（ARM NEON Cortex-A7）

按 V3.1 实测 NN ≈ 0.0097 us/有效MAC 线性外推：

| 配置 | 有效 MAC/帧 | NN 耗时估算 | 总耗时估算(含固定329us) | vs aoplay(531us) |
|------|------------|-----------|----------------------|-----------------|
| V3.1 现状 | ~9K | 87 us | 416 us | 快 22% |
| gru48 甜点/sparse66% | ~23K | ~220 us | ~550 us | ≈持平 ✅ |
| **gru64/sparse66.6%** | **40,977** | **~397 us** | **~726 us** | **慢 37% ❌ 超预算** |
| gru64/sparse80% | ~26K | ~250 us | ~580 us | 慢 9%（可接受） |

固定开销地板（与模型无关）：特征提取 243 + OLA 51 + pitch filter 23 + HP 5 + gain 7 ≈ 329 us/帧，占 V3.1 总耗时 58%，无论模型多大都下不去。

### 结论

1. **差距集中在 GRU 与 Conv2**，均由 GRU 宽度 64 vs 384（6 倍→二次方 36 倍）决定；Conv1 和输出头只差 4~6 倍，不是瓶颈。
2. **gru64 + sparse66.6% 会超 aoplay 速度约 37%**；要用 gru64 必须把稀疏率提到 75~80% 才能压回 ~580 us，但高稀疏率有质量损失风险，需实验验证。
3. **稳在 aoplay 速度内的实际上限是 gru48 甜点档**（cond32, two_conv, sparse66.6%，~550 us，稠密参数 ~40K，比 V3.1 大 4 倍）。
4. 即便扩到 gru64，也仅为开源 0.2 的 1/31 参数、1/26.5 有效算力——嵌入式小模型无法在绝对质量上追平开源大模型，但相对 V3.1 的 ~5 倍有效算力提升应能带来可感知的效果改善。
5. 结构改变后旧 checkpoint 无法微调，必须从头训练。扩容后务必用 `rnnoise/demo/` 的 ARM profile 脚本实测确认未超预算（上表为线性外推估算）。
