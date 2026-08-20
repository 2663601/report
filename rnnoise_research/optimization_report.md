# RNNoise 推理优化对比分析报告

> 基于 xiph/rnnoise v0.2（2024-2025）与当前项目 `rnnoise/` 目录的代码对比分析

---

## 一、网络架构差异

| 特性 | 当前项目 `rnnoise/` | xiph v0.2 |
|------|-------------------|-----------|
| 前端 | Dense (input_dense) | **Conv1d×2**（kernel=3） |
| GRU 连接 | 串联 + input 拼接到每个 GRU | 纯串联（gru1→gru2→gru3） |
| 输出层输入 | 只用最后一个 GRU 输出 | **Skip Connection: conv2+gru1+gru2+gru3 拼接**（4x宽） |
| GRU 大小 | 32/32/32（最大 128） | **256/256/256**（MAX_NEURONS=1024） |
| 默认模型规模 | ~34K 参数 | ~几百万参数 |
| 训练框架 | Keras | **PyTorch + 内置稀疏化训练** |

### xiph v0.2 的核心网络流程（`rnn.c`）

```c
conv1(input) → conv2(tmp) → gru1(conv2) → gru2(gru1) → gru3(gru2)
cat = [conv2, gru1, gru2, gru3]     // Skip Connection
dense_out(cat) → gains
vad_dense(cat) → vad
```

### 当前项目网络流程（`rnn_ns.c`）

```c
input_dense(input) → vad_gru(dense_out) → vad_output(vad_gru)
concat(dense_out, vad_gru, input) → noise_gru
concat(vad_gru, noise_gru, input) → denoise_gru
denoise_output(denoise_gru) → gains
```

---

## 二、计算核心差异

| 特性 | 当前项目 | xiph v0.2 |
|------|---------|-----------|
| 矩阵乘法抽象 | `sgemv_accum16`（手动 16 路展开） | **`compute_linear` 统一接口** + RTCD 派发 |
| 稀疏矩阵 | ❌ | ✅ `sparse_sgemv8x4`（float） + `sparse_cgemv8x4`（int8） |
| int8 量化 GEMV | ❌（纯 float） | ✅ `cgemv8x4`（int8 权重 × float 输入） |
| x86 SSE4.1 | ❌ | ✅ `nnet_sse4_1.c` |
| x86 AVX2 | ❌（有 ifdef 但缺文件） | ✅ `nnet_avx2.c` |
| ARM NEON | ✅ `vec_neon.h`（float32x4 激活函数） | ✅ 类似（编译选项更完善） |
| RTCD 运行时检测 | ❌ 编译时选择 | ✅ 函数指针数组动态派发 |
| Conv1d 支持 | ❌ | ✅ `compute_generic_conv1d` |
| Conv2d 支持 | ❌ | ✅ `compute_conv2d` |
| GLU 激活 | ❌ | ✅ `compute_glu` |
| 模型文件加载 | ❌ 编译内嵌 | ✅ `parse_weights` + binary blob |

---

## 三、稀疏矩阵实现细节

xiph v0.2 支持两种稀疏 GEMV：

### 3.1 Float 稀疏（`sparse_sgemv8x4`）

- **块大小**：8 行 × 4 列
- **索引结构**：`idx[]` 数组记录每 8 行有多少个非零 4 列块，及其列位置
- **计算**：对非零块做 8×4 的展开乘法
- **适用**：float 权重 + 稀疏训练后

### 3.2 Int8 稀疏（`sparse_cgemv8x4`）

- 权重为 `opus_int8`（-128~127）
- 输入做 `x[i] = 127 + floor(0.5 + 127 * _x[i])` 量化
- 配合 `scale` 缩放因子还原
- **适用**：最小模型体积 + 最快计算

### 3.3 稀疏化训练（`GRUSparsifier`）

```python
sparse_params1 = {
    'W_hr': (0.3, [8, 4], True),   # reset gate 循环权重：70%稀疏
    'W_hz': (0.2, [8, 4], True),   # update gate 循环权重：80%稀疏
    'W_hn': (0.5, [8, 4], True),   # output 循环权重：50%稀疏
    'W_ir': (0.3, [8, 4], False),  # reset gate 输入权重：70%稀疏
    'W_iz': (0.2, [8, 4], False),  # update gate 输入权重：80%稀疏
    'W_in': (0.5, [8, 4], False),  # output 输入权重：50%稀疏
}
# 训练 step 6000→20000 期间逐步增加稀疏度，指数衰减
```

---

## 四、RTCD 运行时 CPU 检测机制

xiph v0.2 通过函数指针数组实现 zero-cost dispatch：

```c
// dnn_x86.h
extern void (*const RNN_COMPUTE_LINEAR_IMPL[OPUS_ARCHMASK + 1])(...);
#define compute_linear(linear, out, in, arch) \
    ((*RNN_COMPUTE_LINEAR_IMPL[(arch) & OPUS_ARCHMASK])(linear, out, in))
```

运行时 `arch` 变量由 CPU 检测决定（`x86cpu.c` 检测 SSE4.1/AVX2），同一个 binary 自动选最快实现。

---

## 五、训练管道差异

| 特性 | 当前项目 | xiph v0.2 |
|------|---------|-----------|
| 训练框架 | Keras/TF | **PyTorch** |
| 稀疏化训练 | ❌ | ✅ GRUSparsifier（训练中渐进稀疏化） |
| 稀疏块大小 | — | [8, 4]（8×4 块为单位） |
| 稀疏调度 | — | step 6000→20000 渐进，指数衰减 |
| 数据增强 | 随机 SNR + 低通 + biquad | +饱和量化 +前景噪声 +RIR 混响 |
| VAD 标签 | 能量阈值三值 | **Viterbi HMM 二值** |
| 损失函数 | mycost（sqrt MSE + 4次方） | 类似但调参不同 |
| 权重导出 | `dump_rnn_neon.py` → float C 数组 | `dump_rnnoise_weights.py` → binary blob |

---

## 六、可移植的优化清单

### Tier 1：不改网络结构，直接移植推理优化

| # | 优化 | 说明 | 难度 | 预估加速 |
|---|------|------|------|---------|
| 1 | **NEON sgemv 向量化** | 用 `vfmaq_n_f32` 替代 `sgemv_accum16` 标量展开 | 低 | 矩阵乘法 2-3x |
| 2 | **稀疏矩阵推理** | 移植 `sparse_sgemv8x4`，配合训练端剪枝 | 中 | 30-50% 计算减少 |
| 3 | **int8 量化 GEMV** | 移植 `cgemv8x4`，权重 int8、输入做 127 量化 | 中 | 模型体积÷4 + 更快乘法 |
| 4 | **Swish 激活函数** | 加入 `ACTIVATION_SWISH`（v0.2 新增） | 低 | 可能改善效果 |

### Tier 2：改网络结构，需要重新训练

| # | 优化 | 说明 | 难度 | 预估效果 |
|---|------|------|------|---------|
| 5 | **Skip Connections** | 拼接 conv2+gru1+gru2+gru3 输出到 Dense | 中 | 降噪质量提升 |
| 6 | **Conv1d 前端** | 替代 Dense input_dense | 中-高 | 更好的局部特征提取 |
| 7 | **稀疏化训练** | 移植 GRUSparsifier 到 PyTorch 训练 | 中 | 配合稀疏推理减计算 |
| 8 | **Viterbi VAD 标签** | 移植 `dump_features.c` 中的 viterbi_vad | 低 | 更准确的训练标签 |

### Tier 3：基础设施升级

| # | 优化 | 说明 | 难度 | 收益 |
|---|------|------|------|------|
| 9 | **二进制模型加载** | 移植 `parse_weights` + `WeightArray` 机制 | 高 | 运行时切换模型 |
| 10 | **RTCD 机制** | 多平台 binary + 运行时 CPU 检测 | 高 | 一个 binary 适配所有 CPU |
| 11 | **PyTorch 训练迁移** | 用 xiph 的 `torch/rnnoise/` 为起点 | 高 | 更好的训练生态 |

---

## 七、推荐实施路径

```
Phase 1（立即，1-2 天）:
  ├── 移植 Viterbi VAD 到 denoise.c（改训练数据，不影响推理）
  └── sgemv 用真正的 NEON vfmaq_n_f32 替代标量展开

Phase 2（1-2 周）:
  ├── 引入 Skip Connections（改 compute_rnn + 重新训练）
  ├── 训练时加 GRU 稀疏化（移植 GRUSparsifier）
  └── 推理时加 sparse_sgemv8x4（利用稀疏后的权重）

Phase 3（远期，可选）:
  ├── Conv1d 前端
  ├── int8 量化路径
  └── 二进制模型加载 + RTCD
```

---

## 八、关键代码位置参考

### xiph v0.2（克隆到 `xiph_rnnoise/`）

| 文件 | 作用 |
|------|------|
| `src/nnet.h` | 统一的 LinearLayer/Conv2dLayer 定义 + RTCD 宏 |
| `src/nnet.c` | Dense/GRU/Conv1d/GLU 的高层逻辑 |
| `src/nnet_arch.h` | 平台无关的 compute_linear/activation 实现模板 |
| `src/nnet_default.c` | C 标量 fallback |
| `src/x86/nnet_avx2.c` | AVX2 向量化版本 |
| `src/x86/nnet_sse4_1.c` | SSE4.1 向量化版本 |
| `src/vec.h` | sgemv/sparse_sgemv/cgemv 核心内核 |
| `src/vec_neon.h` | ARM NEON 激活函数 |
| `src/rnn.c` | RNNoise 模型的 compute_rnn（Skip Connection） |
| `torch/rnnoise/rnnoise.py` | PyTorch 模型定义（Conv1d + 3xGRU + Skip） |
| `torch/sparsification/` | GRU 稀疏化训练工具 |

### 当前项目

| 文件 | 作用 |
|------|------|
| `rnnoise/rnn_ns.c` | 推理引擎（sgemv_accum + compute_dense/gru） |
| `rnnoise/vec.h` | sgemv_accum16 标量展开 |
| `rnnoise/vec_neon.h` | NEON float32x4 激活函数 |
| `rnnoise/YS_RNN_Interface.c` | 业务层（多模型切换 + 增益平滑 + 啸叫联动） |
| `training/rnn_train_neon_dataset.py` | Keras 训练（32/32/32 轻量版） |

---

## 九、总结

当前项目的推理引擎基于 xiph/rnnoise v0.1 修改，已做了：
- float 权重（无量化开销）
- NEON 激活函数向量化
- 16 路循环展开

**相比 xiph v0.2 缺失的核心能力**：
1. **稀疏矩阵计算**（最大加速点）
2. **Skip Connections**（最大质量点）
3. **RTCD 多平台适配**
4. **Conv1d 前端**
5. **int8 量化路径**

**推荐的最小改动最大收益方案**：Phase 1 的 Viterbi VAD + NEON GEMV 优化 + Phase 2 的 Skip Connections + 稀疏化训练，可以在不大改代码架构的前提下同时提升**推理速度**和**降噪质量**。

---

*文档生成日期: 2026-05-25*
*参考: xiph/rnnoise main 分支（commit 70f1d25, Feb 2025）*
