# RNNoise 三档模型 VQE 集成指南(V3.1 / V4BIG / V5)

## 一句话概述

在原 V3.1 集成基础上,新增 **V4BIG**(~91K 参数,int8+sparse GRU) 和 **V5**(~2.86M,验证用) 两档,通过 CMake option + 编译宏切换。VQE 侧代码结构不变,只需**改一份 CMakeLists 分支** + **拷更新的源文件**。

---

## 三档对比

| 档次 | 编译宏 | 参数量 | 结构 | 权重文件 | 用途 |
|------|--------|--------|------|----------|------|
| **V3.1**(默认) | 无 | ~9.3K | Conv(45→16→16) + GRU16×3 + cat=64 | `rnnoise_data_16k_v31.c` | 当前部署基线 |
| **V4BIG** | `-DUSE_MODEL_V4BIG=ON` | ~91K | Conv(45→32→64) + GRU64×3 + cat=256 | `rnnoise_data_16k_v4big.c` | 推荐新档,int8+sparse |
| **V5** | `-DUSE_MODEL_V5=ON` | ~2.86M | Conv(45→128→384) + GRU384×3 + cat=1536 | `rnnoise_data_16k_v5.c` | xiph-full,仅验证数据/策略 |

三档共用同一份 `pipeline / rnn_16k / nnet / kiss_fft / pitch / vec` 代码,通过 `MODEL_GRU*_INPUT/RECURRENT` 宏访问不同规格的权重 struct,**pipeline 主逻辑一行不用改**。

---

## 性能实测(x86 WSL gcc -O3, 单核, hp7 howling 8.83s 数据)

| 档次 | best_ms | µs/frame | RTF | 实时倍率 |
|------|---------|----------|------|---------|
| V3.1 | 75 | 84.9 | 0.85% | 117.8× |
| **V4BIG** | **73** | **82.7** | **0.83%** | **121.0×** |
| V5 | 220 | 249.2 | 2.49% | 40.1× |

V4BIG 稀疏结构在 x86 上实测和 V3.1 相当甚至略快(索引路径开销小,且模型大 10x 表达力更强)。ARM 上需再实测。

---

## VQE 集成 3 步走

### 步骤 1:更新 CMakeLists.txt(仅改一处)

在原 `4.4 rnnoise 两个版本` 段落基础上,加 `USE_MODEL_V4BIG` option + elseif 分支:

```cmake
# 4.4 rnnoise 多档模型选择
option(USE_RNNOISE_NEW "Use optimized rnnoise_new V3.1" OFF)
option(USE_MODEL_V5    "Use V5 large model (gru=384, sparse) instead of V3.1" OFF)
option(USE_MODEL_V4BIG "Use V4BIG model (gru=64, int8+sparse, ~91K)"           OFF)  # ← 新增

if(USE_RNNOISE_NEW)
    # 根据模型版本选择权重文件
    if(USE_MODEL_V5)
        set(RNNOISE_WEIGHT_SRC src/rnnoise_new/rnnoise_data_16k_v5.c)
        add_definitions(-DUSE_MODEL_V5)
        message(STATUS "rnnoise model : V5 (gru=384, sparse)")
    elseif(USE_MODEL_V4BIG)                                                          # ← 新增分支
        set(RNNOISE_WEIGHT_SRC src/rnnoise_new/rnnoise_data_16k_v4big.c)
        add_definitions(-DUSE_MODEL_V4BIG)
        message(STATUS "rnnoise model : V4BIG (gru=64, int8+sparse, ~91K)")
    else()
        set(RNNOISE_WEIGHT_SRC src/rnnoise_new/rnnoise_data_16k_v31.c)
        message(STATUS "rnnoise model : V3.1 (gru=16, default)")
    endif()

    # 三档共用的最小自包含文件集(降噪 + VAD, 共 18 个 .c)
    set(RNNOISE_LIST
        src/rnnoise_new/YS_RNN_Interface.c
        src/rnnoise_new/rnn_16k.c
        src/rnnoise_new/rnnoise_16k_pipeline.c
        ${RNNOISE_WEIGHT_SRC}
        src/rnnoise_new/denoise_16k.c
        src/rnnoise_new/rnnoise_tables_16k.c
        src/rnnoise_new/nnet.c
        src/rnnoise_new/nnet_default.c
        src/rnnoise_new/parse_lpcnet_weights.c
        src/rnnoise_new/kiss_fft.c
        src/rnnoise_new/pitch.c
        src/rnnoise_new/celt_lpc.c
        src/rnnoise_new/YS_VAD_RNN_Interface.c
        src/rnnoise_new/rnn_vad.c
        src/rnnoise_new/kiss_fft_vad.c
        src/rnnoise_new/rnn_ns.c
        src/rnnoise_new/rnn.c
        src/rnnoise_new/rnn_data.c
    )
    message(STATUS "rnnoise         : --- rnnoise_new (self-contained)")
else()
    aux_source_directory(src/rnnoise RNNOISE_LIST)
    message(STATUS "rnnoise         : --- rnnoise (original)")
endif()
```

**改动点仅 2 行**:
- `option(USE_MODEL_V4BIG ...)` — 新增
- if/elseif 结构里插一段 V4BIG 分支

`RNNOISE_LIST` 里的 18 个源文件路径不用动(VQE 侧是 `src/rnnoise_new/` 平铺目录)。

---

### 步骤 2:拷贝更新后的源文件到 `src/rnnoise_new/`

**必须更新**(有实质代码变化):

| 文件 | 变化说明 |
|------|---------|
| `rnnoise_data_16k.h` | 加 V4BIG 分支,dispatch header 强制 V3.1/V4BIG/V5 三选一 |
| **`rnnoise_data_16k_v4big.c`** | **新文件**,V4BIG int8+sparse 权重(1.1MB) |
| `rnnoise_16k_pipeline.c` | `ENABLE_EXP_COMB` 默认改为 0(vad+comb 融合补偿默认关闭,省 CPU) |
| `YS_RNN_Interface.c` | 启动日志按 `USE_MODEL_*` 编译宏分档输出 |
| `rnn_vad.c` | 3 处 pitch 函数改用 `rnn_pitch_downsample/search/remove_doubling`(算法零变化,只是符号名对齐) |

**建议同步**(保持整个模块版本一致,避免签名/头不匹配):

- 主降噪路径:`rnn_16k.{c,h}`, `rnnoise_16k_pipeline.h`, `denoise_16k.{c,h}`, `denoise.h`, `rnnoise_tables_16k.c`, `rnnoise_compat.h`, `parse_lpcnet_weights.c`
- NN kernel:`nnet.{c,h}`, `nnet_arch.h`, `nnet_default.c`
- FFT / pitch:`kiss_fft.{c,h}`, `_kiss_fft_guts.h`, `pitch.{c,h}`, `celt_lpc.{c,h}`
- 基础头:`arch.h`, `common.h`, `opus_types.h`, `vec.h`, `vec_avx.h`, `vec_neon.h`, `tansig_table.h`, `timing.h`
- 对外接口头:`YS_RNN_Interface.h`, `YS_RNN_Interface_core.{c,h}`, `YS_VAD_RNN_Interface.h`
- VAD 路径:`rnn_vad.h`, `rnn.{c,h}`, `rnn_data.{c,h}`, `kiss_fft_vad.{c,h}`, `_kiss_fft_guts_vad.h`
- 老降噪(如果 VQE 仍在用):`rnn_ns.{c,h}`, `rnn_data_ns.h`

**权重档次三选一**(按 CMake 选择):`rnnoise_data_16k_v31.c` / `rnnoise_data_16k_v4big.c` / `rnnoise_data_16k_v5.c`。三份都拷进去,CMake 自动选。

---

### 步骤 3:VQE 编译命令

```bash
# 默认 V3.1(与之前一致,零迁移成本)
cmake -DUSE_RNNOISE_NEW=ON ...

# 切 V4BIG(推荐新档,~91K 参数, int8+sparse)
cmake -DUSE_RNNOISE_NEW=ON -DUSE_MODEL_V4BIG=ON ...

# V5(仅数据/策略验证, 不部署)
cmake -DUSE_RNNOISE_NEW=ON -DUSE_MODEL_V5=ON ...
```

**运行时日志**(YS_RNN_Interface.c 启动会打印一行确认):
- V3.1: `[rnnoise_v31] V3.1 engine created (~9351 params, gru=16, int8 GRU, pitch filter)`
- V4BIG: `[rnnoise_v31] V4BIG engine created (~91K params, cond=32, gru=64, int8+sparse GRU, pitch filter)`
- V5: `[rnnoise_v31] V5 engine created (~2.86M params, cond=128, gru=384, int8+sparse GRU, pitch filter)`

---

## 融合补偿(vad+comb)开关

V3.1 时代 pipeline 里有个 `ENABLE_EXP_COMB` 后处理增益补偿块(用 Exp VAD 拖尾门控 + pitch band 谐波混合),会额外消耗 CPU 且和一些下游模块行为不完全对齐。

**本次集成默认关闭**(`ENABLE_EXP_COMB=0`),关掉后:
- 每帧省 ~30 次 float 比较 + 160×2 次 mixf 乘加(约 5-10% 单帧降噪 CPU)
- `st->exp_speech_flag` 拖尾状态机同步跳过
- VQE 走干净的"NN gain + pitch filter + OLA"三段路径,与老 v0.1 行为一致

**如需 A/B 对比重开**(仅调优用,不建议部署):

```bash
cmake ... -DCMAKE_C_FLAGS="-DENABLE_EXP_COMB=1"
```

---

## 接口(不变)

```c
#include "YS_RNN_Interface.h"          // 降噪
#include "YS_VAD_RNN_Interface.h"      // VAD(独立老 v0.1 接口)

/* 降噪 */
int   memsize = YS_Rnnoise_Get_memsize();
YS_RNN_ParamIn  in  = { .sample_rate=16000, .frame_len=160, .mode=0, .howl_flag=0 };
YS_RNN_ParamOut out;
void *handle = NULL;
unsigned char *buf = calloc(1, memsize);
YS_Rnnoise_Create(&in, buf, &handle);
in.data_in_pcm  = pcm_in_short;
out.data_out_pcm = pcm_out_short;
YS_Rnnoise_Process_Frame(&in, &out, (unsigned char *)handle);

/* VAD(独立 handle, 与降噪并行) */
int vad_mem = YS_Audio_VAD_RNN_GetMemSize();
unsigned char *vbuf = calloc(1, vad_mem);
void *vhandle = NULL;
YS_VAD_RNN_ParamIn  vin  = { .samplerate=16000, .framelen=160,
                             .E_floor=-40.f, .vad_prob_floor=0.5f,
                             .data_in_pcm=pcm_in_short };
YS_VAD_RNN_ParamOut vout;
YS_Audio_VAD_RNN_Create(&vin, vbuf, &vhandle);
YS_Audio_VAD_RNN_Process(&vin, &vout, vhandle);
// vout.vad_flag  : 0/1
// vout.E_out     : 帧能量 dB
```

**降噪的编译宏切档,VAD 完全不受影响**,VAD 依赖老 v0.1 权重(`rnn.c`+`rnn_data.c`+`kiss_fft_vad.c`)独立编入。

---

## 换 pth 快速流程

VQE 侧不动,只需在 rnnoise 主仓库 rerun dump 脚本 → cp 权重 .c:

```bash
# 例:换 V4BIG 权重为新的 pth
bash training/rnnoise_16k/_dump_v4big.sh path/to/new.pth
cp /tmp/v4big_export/rnnoise_data_16k_v4big.c rnnoise/            # rnnoise/ 侧覆盖
# 再 cp 到 VQE 侧 src/rnnoise_new/
```

产物就是 `rnnoise_data_16k_v4big.c`(1.1MB int8+sparse 数组),CMake / 源码 / 编译宏都不变。

---

## 已知坑

1. **wexchange 生成 .h 与 dispatch header 冲突**  
   dump 后**不要**拷伴生 `rnnoise_data_16k_v4big.h`,只拷 .c。规格由 `rnnoise_data_16k.h` 的 `#ifdef USE_MODEL_V4BIG` 分支统一提供。dump 脚本 `_dump_v4big.sh` 已自动处理。

2. **rnn_vad.c pitch 函数命名**  
   老版本调 `pitch_downsample/search/remove_doubling`,新版本 `pitch.c` 加了 `rnn_` 前缀。已在源码里直接改成 `rnn_pitch_*`,算法逐 bit 等价。若发现 VAD 侧编译时找不到 `pitch_downsample`,说明 VQE 里还有其它历史文件在用老符号,加 `compat_aliases.c`(见 `rnnoise/legacy/`)做 wrapper。

3. **档次切换要 make clean 一次**  
   `USE_MODEL_V4BIG` 变化会改 `-D` 编译宏,若 CMake 缓存了旧 struct 布局的 .o,链接会出符号不匹配。建议:
   ```bash
   rm -rf build && cmake -DUSE_RNNOISE_NEW=ON -DUSE_MODEL_V4BIG=ON ..
   ```

4. **平台 SIMD 宏**  
   编译加 `-DRTCD_ARCH=c -DDISABLE_DEBUG_FLOAT`。ARM 侧 `-DARM_NEON_INTR` 之类看 VQE 现成 flag 沿用即可,`vec_neon.h` 会自动启用 NEON kernel。x86 侧 `-march=native` 会启用 `nnet_avx2.c` / `nnet_sse4_1.c`,但目录里已有分派,不用手动加。

---

## VAD 独立编译校验(可选)

VQE 集成 VAD 前想先验证依赖闭环,在 rnnoise 主仓库执行:

```bash
bash rnnoise/scripts/build_check_vad.sh
# 通过时打印: "VAD mem: 12672 bytes" + "=== VAD BUILD CHECK PASSED ==="
```

---

## 版本历史

| 日期 | 变更 |
|------|------|
| 2026-08-12 | V3.1 集成(YS_RNN_Interface,`ENABLE_EXP_COMB=1`) |
| 2026-08-30 | V5 分支加入(仅验证用,不部署) |
| 2026-09-22 | **V4BIG 集成,`ENABLE_EXP_COMB` 默认关闭,rnn_vad.c pitch 前缀对齐** |

---

## 反向问题速查

| 现象 | 原因 | 对策 |
|------|------|------|
| 链接 `pitch_downsample/search/remove_doubling` 未定义 | VQE 里还有老代码引用无前缀符号 | 加 `compat_aliases.c` wrapper 到源列表 |
| 运行日志显示 `V3.1 engine created`,期望 `V4BIG` | CMake 缓存了老 `-D` | `rm -rf build && cmake -DUSE_MODEL_V4BIG=ON ..` |
| 换权重后 pipeline crash 在 `linear_init` | .c 里 `init_rnnoise16k` 的字符串与 `rnnoise_data_16k.h` 里 struct 成员不匹配 | 检查 dump 出的 `.c` 是否 include 了 dispatch header;GRU 命名 v4big/v5 用 `gru1_input` |
| VAD 独立编译 undefined reference | 缺 pitch.c / celt_lpc.c | 加进 VAD 依赖列表(与降噪路径共享) |
| 帧耗时不降反升 | `USE_MODEL_V4BIG` 没进宏,还在跑 V3.1 稠密 | 看运行日志确认档次;`make VERBOSE=1` 确认 `-DUSE_MODEL_V4BIG` 上了 |
