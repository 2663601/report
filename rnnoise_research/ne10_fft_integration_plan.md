# Ne10 FFT 320pt 集成方案（无 malloc）

## 目标
替换 `denoise_16k.c` 中的 kiss_fft（320pt complex FFT）为 Ne10 NEON 优化版本，满足：
- 零 malloc（twiddle table 编译期静态化）
- 用宏 `USE_NE10_FFT` 控制开关
- 不加宏时走原始 kiss_fft 路径

## Ne10 源码位置
`E:\server_code\rnnoise_16k\Ne10\modules\dsp\`

## 关键文件
- `NE10_fft.c` — alloc/init 入口
- `NE10_fft_generic_float32.neonintrinsic.c` — NEON 蝶形核心
- `NE10_fft_bfly.h` — radix-2/3/4/5 蝶形宏
- `NE10_fft.neonintrinsic.h` — NEON 辅助宏
- `NE10_fft_common_varibles.h` — twiddle 相关

## 320pt FFT 分解
320 = 4 × 4 × 4 × 5（Ne10 mixed-radix 路径）
- 需要的 twiddle factors: N/4=80 个复数 + N/16=20 个 + N/64=5 个 + radix-5 twiddle
- 总 twiddle 表约 200 个 ne10_fft_cpx_float32_t（= 400 float = 1.6KB）

## 实施步骤

### Step 1: 预计算 twiddle table（Python）
```python
# gen_ne10_twiddle_320.py
import numpy as np
N = 320
# Ne10 mixed-radix twiddle generation logic...
# 输出: static const float twiddle_320[...] = {...};
```
输出到 `rnnoise/ne10_fft_twiddle_320.h`

### Step 2: 裁剪 Ne10 核心代码
从 Ne10 源码抽取以下到 `rnnoise/ne10_fft_neon.c`：
- `ne10_mixed_radix_butterfly_float32_neon()` — NEON 蝶形
- `ne10_fft_c2c_1d_float32_neon()` — complex FFT
- `ne10_fft_r2c_1d_float32_neon()` / `ne10_fft_c2r_1d_float32_neon()` — real FFT

去掉所有 malloc 调用，改为引用静态 twiddle 表。

### Step 3: 适配接口
在 `denoise_16k.c` 中加 `#ifdef USE_NE10_FFT` 分支：
```c
#ifdef USE_NE10_FFT
#include "ne10_fft_neon.h"
// forward_transform: ne10_fft_r2c_1d_float32_neon(out, in, &static_cfg_320)
// inverse_transform: ne10_fft_c2r_1d_float32_neon(out, in, &static_cfg_320)
#else
// 原始 kiss_fft 路径
#endif
```

### Step 4: 编译验证
- ARM 交叉编译 + qemu-arm 验证输出正确性
- 对比 kiss_fft 版本输出（允许微小浮点差异 SNR>60dB）

### Step 5: 板端测试
- 对比特征提取耗时（预期 FFT 部分加速 3-5x）

## 约束
- 不允许 malloc/calloc/realloc
- twiddle 表必须为 static const（ROM 空间）
- FFT state 结构体可以放在外部传入的 alloc_buf 上（如需 scratch buffer）
- Ne10 的 NEON FFT 核心需要约 320×2×sizeof(float)=2560 bytes 的 scratch buffer

## 编译宏
```
-DUSE_NE10_FFT          # 启用 Ne10 FFT
-DPITCH_XCORR_NEON      # 启用 pitch NEON（已完成）
```

## 新对话 prompt
"按照 docs/ne10_fft_integration_plan.md 的方案，从 Ne10 源码裁剪 320pt real FFT NEON 实现，集成到 rnnoise/denoise_16k.c，用 USE_NE10_FFT 宏控制。Ne10 源码在 E:\server_code\rnnoise_16k\Ne10。要求零 malloc，twiddle 表静态化。"
