# 48k gru64 vs 16k v4big 结构对比

对齐 GRU 宽度(都是 3×GRU64)后,48k xiph 缩小版(gru64/cond128)与 16k v4big(gru64/cond32)
的参数量、算力差异分析。**结论:GRU 主体完全相同,差异全在前端 Conv,v4big 是"同 GRU 骨架 + 更瘦的前端"。**

## 1. 结构对照

| | 48k gru64 (xiph RNNoise 类) | v4big (RNNoise16k_V4 类) |
|---|---|---|
| 代码路径 | `xiph_rnnoise/torch/rnnoise/rnnoise.py` | `training/rnnoise_16k/rnnoise_16k.py` |
| 采样率 / 帧长 | 48kHz / 10ms(480 样本) | 16kHz / 10ms(160 样本) |
| 输入特征维度 | 65(45 + 20 复数谱) | 45(2×22+1) |
| 输出 gain 维度 | 32 band | 22 band |
| Conv1 | in→**128**, k=3 | in→**32**, k=3 |
| Conv2 | 128→64, k=3 | 32→64, k=3 |
| GRU | 3×GRU64(相同) | 3×GRU64(相同) |
| cat_size | 4×64 = 256 | 4×64 = 256 |
| dense_out | 256→32 | 256→22 |
| vad_dense | 256→1 | 256→1 |

## 2. 逐层参数量(公式 + 数值)

- Conv1d(k=3): `in × out × 3 + out`
- GRU(input=hidden=64): `3 × (64×64 + 64×64 + 64) = 24,768`
- Linear: `in × out + out`

| 层 | 48k gru64 | v4big | 备注 |
|---|---:|---:|---|
| Conv1 | 65×128×3+128 = **25,088** | 45×32×3+32 = **4,352** | 48k 前端厚 5.8× |
| Conv2 | 128×64×3+64 = **24,640** | 32×64×3+64 = **6,208** | 48k 4.0× |
| GRU1 | 24,768 | 24,768 | 相同 |
| GRU2 | 24,768 | 24,768 | 相同 |
| GRU3 | 24,768 | 24,768 | 相同 |
| dense_out | 256×32+32 = **8,224** | 256×22+22 = **5,654** | |
| vad_dense | 257 | 257 | 相同 |
| **合计** | **≈132,513** | **≈90,775** | **48k 是 v4big 的 1.46×** |

> v4big 实测打印 91,351(与推算 90,775 略差是 padding/bias 计入方式细节,量级一致)。

## 3. 计算量(每帧 MAC)

Conv/GRU/Dense 都是乘加主导,**稠密下 MAC/帧 ≈ 参数量**(每帧算一次前向)。两者帧率都是 100 fps(10ms 帧)。

| 情况 | 48k gru64 | v4big | 倍率 |
|---|---:|---:|---:|
| 稠密 MAC/帧 | ≈132K | ≈91K | 1.46× |
| 稀疏 66.6% 后有效 MAC/帧 | ≈83K | ≈41K | **2.0×** |
| GRU 主体(3 层稠密) | 74,304 | 74,304 | 1.0× |
| 前端 Conv(Conv1+Conv2) | 49,728 | 10,560 | **4.7×** |

**稀疏后差距被前端 Conv 放大**:GRU 那 74K 参数被稀疏到约 1/3(≈25K),但 Conv 是稠密不参与稀疏。
- 48k 稀疏后:25088(conv1) + 24640(conv2) + 24768(gru) + 8224 + 257 ≈ **83K**
- v4big 稀疏后:4352 + 6208 + 24768 + 5654 + 257 ≈ **41K**

## 4. 差异定位:全部来自前端

| 差异来源 | 48k 多出的量 | 原因 |
|---|---:|---|
| Conv1 前端 | +20,736 | 48k cond=128 vs v4big cond=32; 输入 65 vs 45 |
| Conv2 前端 | +18,432 | 48k cond=128 输入 vs v4big cond=32 输入 |
| dense_out | +2,570 | 48k 32 band vs v4big 22 band |
| GRU 主体 | 0 | **完全相同** |
| **总差异** | **≈41,738** | |

48k 需要 cond=128 的宽 Conv 前端,是为了在 48k 宽频谱下压出足够信息量给 GRU。v4big 面向 16k
(带宽只有 8k Hz,信息量少),cond=32 足够,前端瘦下来 5 倍。

## 5. ARM 部署角度的意义

固定开销地板(特征提取+OLA+pitch+HP+gain)≈ **329 µs/帧**(与模型大小无关,占预算 58%)。
NN 部分按 ≈0.0097 µs/有效MAC 外推(V3.1 实测锚点):

| 模型 | 有效 MAC/帧 | NN 耗时(µs) | 总耗时(µs) | vs aoplay 531µs 预算 |
|---|---:|---:|---:|---|
| 48k gru64 稀疏 | ≈83K | ≈805 | ≈1134 | **超 113%** ❌ |
| v4big gru64 稀疏 | ≈41K | ≈398 | ≈727 | **超 37%** ⚠️(需再压稀疏率到 80%) |
| v4big gru48 稀疏 | ≈29K | ≈281 | ≈610 | 接近预算,可用 |
| V3.1 baseline | ≈8.5K(稠密) | ≈82 | ≈411 | ✅ |

结论:**同为 gru64,v4big 之所以能塞进 ARM 预算而 48k gru64 不行,主要因为省下的 4 万参数
几乎全在每帧要算的稠密 Conv 前端**——GRU 部分两者算力完全一样。

## 6. 一句话总结

> v4big 是 48k xiph 结构在 16k 场景的极简移植:GRU 骨架(3×GRU64)一模一样,
> 只把前端 Conv 从 cond=128 砍到 cond=32(-4.7×),换取参数量 -46%、稀疏后有效算力 -50%,
> 从而挤进 ARM Cortex-A7 的每帧预算。

## 7. 依据

- 48k RNNoise 类:`xiph_rnnoise/torch/rnnoise/rnnoise.py` line 58-70
- 16k RNNoise16k_V4 类:`training/rnnoise_16k/rnnoise_16k.py` `RNNoise16k_V4` line 244
- GRU 参数公式:`3 × (input×hidden + hidden×hidden + hidden)`(PyTorch nn.GRU 单层)
- ARM 速度预算锚点见 `03_inference_integration/rnnoise_v31_vqe_integration_guide.md`「扩容方案与开源 0.2 逐层结构对比」
