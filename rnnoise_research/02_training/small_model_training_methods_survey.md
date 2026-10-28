# 小模型性能提升方法调研（学术界 + 工业界）

> 日期：2026-08-11
> 适用范围：RNNoise 16k 这类 <30K 参数的极小语音增强模型的性能提升
> 说明：本文档汇总学术界与工业界针对小模型训练的主流方法，并结合本项目
>       （V3/V4，8.6K/25.7K 参数）已有的实测结论给出优先级建议。
>       所有引用来源见文末「参考文献」，内容已按合规要求改写。

---

## 0. 结论先行（TL;DR）

针对本项目「几 K~几十 K 参数、部署在 MCU/DSP、目标 PESQ/STOI」的场景，
按**投入产出比**排序，推荐优先级如下：

| 优先级 | 方法 | 预期收益 | 本项目落地状态 |
|---|---|---|---|
| P0 | 从强基线微调（而非从头训练加辅助 loss） | 已被本项目日志证实：辅助 loss 需在收敛基线上才起效 | 已采用 |
| P0 | 感知/任务对齐的 loss（PESQ、STOI、SI-SNR、过压制/平滑） | 直接对齐评估指标，本项目已接入 | 已采用 |
| P1 | 知识蒸馏（KD），尤其是**选择性/特征级**蒸馏 | 学术界小模型 SE 报告 PESQ/SI-SNR 可观提升 | 部分采用（DFNet/GTCRN gain 蒸馏） |
| P1 | 数据增强（SNR 课程学习、SpecAugment、混合噪声） | 对小模型泛化收益大、零推理成本 | 部分采用（多 SNR/RIR） |
| P2 | 压缩三件套顺序（剪枝 → QAT → KD）| 主要降体积，KD 放最后回收精度 | 稀疏化已有，QAT 未做 |
| P2 | 结构搜索/感受野调整（two-conv、kernel） | 本项目 v3.1 已验证 +conv 提升上下文 | 已采用 |

**一句话**：小模型的性能天花板由「结构容量 + 目标质量」共同决定。本项目当前最
可靠的增益来自**在收敛基线上做感知 loss 微调 + 高质量 teacher 蒸馏**，而不是堆结构。

---

## 1. 知识蒸馏（Knowledge Distillation, KD）

KD 是小模型提升最主流的手段：用一个大的 teacher 模型指导小 student 模型，让 student
逼近 teacher 的输出分布或中间表示，从而在不增加推理成本的前提下逼近大模型性能。

### 1.1 输出级蒸馏（logit / mask 蒸馏）
最经典形式：student 直接回归 teacher 的输出（对 SE 来说是 mask/gain 或增强频谱）。
本项目的 DFNet gain 蒸馏、GTCRN gain 蒸馏就属于这类（离线存 teacher gain，训练时回归）。

**要点**：teacher 质量决定上界。本项目实测 DFNet→22band oracle 的 PESQ 上界比 V3
高约 0.365，是选它当 teacher 的依据。但 teacher 输出与 student 结构差异过大时，
直接回归全部输出反而可能引入 student 学不动的目标。

### 1.2 特征级蒸馏（feature-based / hint）
不止对齐最终输出，还对齐中间层特征（FitNets 思路）。Interspeech 2024 的一项工作提出
**特征增强的蒸馏**，利用 teacher 中间潜在特征来训练更小的 student，用于语音增强。
另有 **Distil-DCCRN**（arXiv 2408.04267）：student 仅为 DCCRN 的约 30% 参数，通过
基于注意力的 KL 特征蒸馏（AT-KL），在 DNS 测试集上 PESQ/SI-SNR 反超原 DCCRN，
DNSMOS 也接近。这说明**特征级蒸馏在小 SE 模型上确有超越 teacher 的潜力**。

> 本项目的 `--distill-hint-weight`（FitNets hint）就是这个方向，但目前默认关闭（0）。
> 若 V5/GTCRN teacher 的中间维度能对齐，可尝试开启。

### 1.3 选择性蒸馏（重点，2025 新方向）
**DISPatch**（arXiv 2509.15922，Kim et al., 2025）提出：不要在所有时频区域一视同仁
地蒸馏，而是用一个「知识差距分数（Knowledge Gap Score）」找出**teacher 明显优于
student 的频谱 patch**，只在这些区域施加蒸馏 loss。理由是：teacher 也不是处处都好，
在 teacher 本身也差的区域强行对齐只会引入噪声监督。

**Dynamic Frequency-Adaptive KD**（arXiv 2502.04711）思路类似：按频带动态调整蒸馏
权重，报告在 SE 任务上超过其他 logit 级蒸馏方法。

> **对本项目的启示**：这正好呼应我们的诊断结论——V3 与 teacher 的差距主要在**帧间
> 增益抖动**而非均匀压制。可以把「只在 teacher 明显更好的帧/带上蒸馏」落地为：对
> teacher gain 蒸馏 loss 按 `|teacher - student|` 或按帧间差分加权，而不是全帧等权。
> 这比当前的全帧 MSE 更对症，且实现成本低（改 loss 加权即可）。

### 1.4 teacher 选择
arXiv 2511.02833 指出：最优 teacher 依 student 与任务而定，盲目用最大 teacher 未必最好。
本项目已踩过坑：V5（自训 384 GRU）听感差，不能当 teacher；应选 xiph 开源或 DFNet/GTCRN
这类客观指标经过验证的模型。

---

## 2. 任务/感知对齐的损失函数

小模型容量有限，用「更接近评估指标」的 loss 能把有限容量用在刀刃上。这条路本项目
已大量落地（PESQ、STOI、SI-SNR、over-suppress、gain-smooth），此处补充方法论。

- **可微感知 loss**：直接优化 PESQ/STOI 的可微近似，让训练目标对齐评估。注意权重
  要小（本项目实测 PESQ 权重 0.1 会反客为主导致过压制，最终定为 0.01）。
- **多分辨率 STFT loss（MR-STFT）**：谱收敛 + 对数幅度 L1，跨多个 FFT 分辨率求和，
  波形域常用，能同时约束粗细结构。
- **时域保真 loss（SI-SNR）**：保护谐波结构，抵消纯感知 loss 的过平滑。
- **结构化正则**：本项目的 over-suppress（只罚压过头）、gain-smooth（罚帧间跳变）
  都是把「已诊断出的具体损伤形态」写成 loss，属于最高性价比的定向修复。

> 方法论要点：**先诊断再加 loss**。本项目通过 `diag_gain_distribution.py` 定位到
> 「帧间波动 7.95dB vs 官方 5.03dB」才决定加 STOI/gain-smooth，避免了盲目堆 loss。

---

## 3. 数据增强与训练课程

对小模型，数据侧的改进往往比结构改进更划算，且零推理成本。

### 3.1 SpecAugment
arXiv 1904.08779（Park et al., 2019）：直接在特征（滤波器组系数）上做时间弯曲、
频带屏蔽、时间步屏蔽。原本用于 ASR，但屏蔽类增强对 SE 前端特征同样能提升泛化。
衍生的 **SpecMix**（arXiv 2108.03020）通过时频掩码混合两条样本，保留各自的谱相关性。

### 3.2 SNR 课程学习（Curriculum Learning）
EUSIPCO 2017 的 ACCAN（accordion annealing）：多阶段训练，先加入低至 0dB 的低 SNR
样本，再逐步加入更高 SNR（直到 50dB）。核心思想是**由难到易或由易到难地安排 SNR
分布**，改善噪声鲁棒性。

> **对本项目的启示**：当前训练是固定 SNR 分布混合。可尝试课程式：前若干 epoch 用
> 中高 SNR 让模型先学会「基本降噪 + 保语音」，后期再压低 SNR 增强难样本能力。
> 实现成本低（改数据生成的 SNR 采样策略），值得一试。

### 3.3 混合噪声 / RIR
本项目已有多噪声混合与 RIR 选项（当前 RIR 因破坏 clean 目标而关闭）。RIR 根因修好前
不要开——这条已在项目里踩过并记录。

---

## 4. 压缩三件套：剪枝、量化、蒸馏

工业界部署小模型的标准组合是**剪枝（Pruning）+ 量化（Quantization）+ 蒸馏（KD）**，
三者可单独用也可流水线组合。

### 4.1 顺序很重要
arXiv 2106.14681 与 Springer 2025（10.1007/s42835-025-02541-7）都研究了三者的流水线。
一个较一致的经验（arXiv 2604.04988）：
- **剪枝**主要作为「容量削减的预处理」，为后续低精度优化提供更鲁棒的起点；
- **INT8 量化感知训练（QAT）** 提供主要的运行时收益；
- **KD 放在最后**，在已经稀疏 + INT8 的受限空间里回收精度，且不改变部署形态。

### 4.2 量化感知训练（QAT）
多篇工作（arXiv 2409.01990、2502.00046）指出 QAT 能在显著降体积的同时保持精度，
4-bit 量化可大幅降能耗且精度损失很小。对本项目：WebRTC 的 RNN-VAD 已用 **int8 权重**
存储 rnnoise 量级的网络（约 4.6K 参数），证明这个量级 int8 掉点可忽略——这是本项目
后续压缩体积的一条已验证路径。

### 4.3 量化 + 蒸馏联合
arXiv 2509.20854（"Punching Above Precision"）提出可学习正则器 GoR，缓解量化与蒸馏
两路监督信号的冲突，提升小量化模型（SQM）收敛与性能，在分类/检测/LLM 压缩上均超过
现有 QAT-KD 方法。若本项目未来要做 INT8 + 蒸馏，这类「协调多监督信号」的思路可参考。

> 本项目现状：稀疏化（sparse）已有但微调时建议关闭（会在微调期重新清零权重、破坏
> 起点）。QAT 尚未做。建议顺序：先用浮点把感知/蒸馏 loss 调到最优 → 再做 INT8 QAT。

---

## 5. 结构与容量

小模型性能受结构容量硬约束，但在固定参数预算内仍有优化空间。

- **感受野 / 上下文**：本项目 v3.1 把 Conv 前端从一层加到两层（对齐 xiph 原版），
  感受野 3 帧→5 帧，仅 +约 780 参数（+9%），是低成本的结构增益。
- **kernel 大小**：convk5/convk7 提供更宽上下文（本项目已有消融开关）。
- **架构演进方向**：xiph rnnoise v0.1→v0.2 在**参数量降到 29%** 的同时 PESQ +0.138，
  靠的是删除 delta-cepstrum 特征改用 conv 前端。这说明**盲目加特征/加层不是正道**，
  结构效率比堆容量更关键（本项目已据此撤回「补 delta-cepstrum」的想法）。

---

## 6. 针对本项目的具体行动建议（按性价比）

1. **选择性蒸馏（最推荐，低成本高对症）**：把 DFNet/GTCRN gain 蒸馏 loss 从全帧等权
   改为「只在 teacher 明显更好的帧/带」加权，或改为**帧间差分蒸馏** `MSE(Δpred, Δteacher)`。
   直接命中已诊断的「帧间抖动」损伤。参考 DISPatch 的 Knowledge Gap Score 思想。
2. **SNR 课程学习**：改数据生成的 SNR 采样为课程式，零推理成本。
3. **特征级蒸馏**：尝试开启 `--distill-hint-weight`（当前默认 0），对齐 teacher 中间特征。
   参考 Distil-DCCRN 用 AT-KL 反超 teacher 的结果。
4. **验证机制而非只看指标**：每次改动后重跑 `diag_gain_distribution.py`，确认帧间波动
   从 7.95dB 向 5~6dB 靠拢——机制层面的验证比单看 PESQ 更可信。
5. **部署阶段再做 INT8 QAT**：浮点调优完成后再压 INT8，参考 WebRTC RNN-VAD 的 int8
   先例（该量级掉点可忽略）。

---

## 7. 参考文献

- DISPatch: Distilling Selective Patches for Speech Enhancement, arXiv:2509.15922 (Kim et al., 2025)
- Dynamic Frequency-Adaptive Knowledge Distillation for Speech Enhancement, arXiv:2502.04711
- A Small-footprint DCCRN Leveraging Feature-based Knowledge Distillation (Distil-DCCRN), arXiv:2408.04267
- Feature-augmentation based Knowledge Distillation for Speech Enhancement, Interspeech 2024 (gholami24)
- Comparison of KD Methods for Low-complexity Multi-microphone SE (FT-JNF), arXiv:2507.19208
- Principled Teacher Selection for Knowledge Distillation, arXiv:2511.02833
- SpecAugment: A Simple Data Augmentation Method for ASR, arXiv:1904.08779 (Park et al., 2019)
- SpecMix: A Mixed Sample Data Augmentation for Time-Frequency Features, arXiv:2108.03020
- ACCAN: A Curriculum Learning Method for Improved Noise Robustness in ASR, EUSIPCO 2017
- Model Compression via Pruning, Quantization, and KD, arXiv:2106.14681
- An Ordered Pipeline for Efficient Neural Network Compression, arXiv:2604.04988
- Pipeline of Pruning, KD, and Quantization for Model Compression, Springer 10.1007/s42835-025-02541-7
- Punching Above Precision: Small Quantized Model Distillation with Learnable Regularizer (GoR), arXiv:2509.20854
- Optimization Strategies for Resource Efficiency in Transformers & LLMs, arXiv:2502.00046
- Contemporary Model Compression on LLM Inference, arXiv:2409.01990

> 注：以上外部结论均为公开论文的方法性描述，内容已改写以符合引用规范；具体数值
> 收益以各论文原文与本项目自身实测（见 `rnnoise_16k_optimization_summary.md`）为准。
> Content was rephrased for compliance with licensing restrictions.

---

## 附录 A：六模型横向实测对标（2026-08-11 追加）

> 口径：50 条 DNS2020 test_set/synthetic/no_reverb，`--sample-seed 42 --use-c-features`
>       （C 特征提取 + pitch filter，与 C 推理对齐）。增益诊断由 `diag_gain_distribution.py`
>       在语音帧上统计逐 band 相对 clean 的 dB 增益（取能量高于中位数的帧）。
> 所有数字均为本机实测，来源可复现；无实测的项标注「未测」，不臆造。

### A.1 感知/客观指标

| 模型 | PESQ | STOI | SI-SNR | 参数量 | 架构/采样率 | 数据来源 |
|---|---|---|---|---|---|---|
| noisy（未处理） | 1.570 | 0.9124 | 9.21 | — | — | 本轮实测 |
| V3（best_pesq_ovs） | 1.8897 | 0.7495 | 9.32 | 8567 (~8.6K) | 纯 gain, 16k, 单层conv | 历史实测 |
| **V3.1**（best_pesq_1.959） | **1.931** | 0.7560 | 9.30 | ~9.4K | 纯 gain, 16k, 两层conv | 本轮实测 |
| V4（best_pesq_2.087） | 2.087 | 0.7506 | 7.60 | 25.7K | 纯 gain, 16k | 本轮实测 |
| V5（best） | 1.940 | 0.7873 | 12.25 | ~2.86M | 纯 gain, 16k | 本轮实测 |
| rnnoise v0.1 | 1.9305 | 0.9219 | 11.28 | 87.5K | 纯 gain, 48k | 历史实测 |
| **rnnoise v0.2**（xiph 主线） | **2.070** | 0.8064 | 12.58 | ~5.79M | 纯 gain, 48k | 本轮实测 |
| dfnet（DeepFilterNet3） | 2.5034 | 未测* | 未测* | ~2M | 复数谱 deep filtering, 48k | 历史实测 |

> *dfnet 的 STOI/SI-SNR 本轮未做 50 条汇总（历史 csv 有逐条数据，均值待补），
>  仅 PESQ 2.5034 为历史确证值。
> ⚠️ v0.2 参数量更正：早期文档误记为 25.7K（那实为 V4 参数）。本地 `xiph_rnnoise`
>  主线 `rnnoise_data.c`（78MB）权重元素实测约 **5.79M**（conv1→128 + conv2→384
>  + GRU384×3 + 32band），是百万级，不是 25.7K。

### A.2 增益分布诊断（语音帧，相对 clean 的 dB 增益）

| 模型 | 整体平均增益 | 带间不一致(std) | 帧间波动(std) |
|---|---|---|---|
| v0.1 | +0.09 dB | 1.13 dB | **5.03 dB** |
| dfnet | −3.43 dB | 1.75 dB | **5.61 dB** |
| V5 | −1.26 dB | 1.15 dB | 7.90 dB |
| v0.2 | −2.22 dB | 1.56 dB | 7.92 dB |
| V3（历史） | −2.53 dB | 1.35 dB | 7.95 dB |
| V3.1 | −3.18 dB | 1.17 dB | 8.14 dB |
| V4 | −5.57 dB | 1.72 dB | 8.35 dB |

### A.3 关键结论（修正了早期两处误判）

1. **V3.1 的 PESQ（1.931）已追平 rnnoise v0.1（1.9305）**，仅用其 ~1/9 参数（9.4K vs 87.5K）；
   V4 的 PESQ（2.087）甚至逼近 v0.2（2.070），但代价是 STOI/SI-SNR 双低（见下）。

2. **与真正开源对标 v0.2 的差距**（v0.2 = 5.79M / 48k，V3.1 = 9.4K / 16k，参数差 ~615 倍）：
   PESQ 差 0.14、STOI 差 0.05、SI-SNR 差 3.28dB。用 1/615 参数换来的差距其实不大，
   主要缺口在 SI-SNR（波形保真）和 STOI（可懂度）——这两项需要容量做精细重建。

3. **【更正一】v0.2 没有 deep filtering。** 早期误判 v0.2 靠 deep filtering 取胜、需用 v4df 补相位——
   经核读 `xiph_rnnoise/src/denoise.c`：v0.2 推理输出就是 32-band 实数 gain（`interp_band_gain`
   后 `X.r/X.i *= gf`，不改相位），唯一复数操作是 v0.1 就有的 pitch filter。**v0.2 与 V3.1
   是同类纯 gain 架构**，其优势纯来自「更大网络 + 32band + 48k」，不是相位重建。
   （只有 dfnet 才是真正的复数谱 deep filtering。）

4. **【更正二】帧间波动不是小模型的独有病根，性价比低。** 早期判断「帧间波动 8dB 是 STOI
   杀手、应上 gain-smooth/选择性蒸馏」被两个实测反例推翻：
   - **v0.2 帧间波动 7.92dB ≈ V3.1 的 8.14dB**，但 v0.2 的 STOI/SI-SNR 远高——同样的波动，质量差很多。
   - **V5（2.86M，V3.1 的 300 倍参数）帧间波动仍是 7.90dB**，加容量几乎压不动。
   → 帧间波动 ~7.9~8.4dB 是「16k 纯 gain + 本套特征/framing」体系的**固有常态**，
     不是 loss 能显著改善的，投入产出比低。只有 v0.1(5.03)/dfnet(5.61) 两项低：
     v0.1 靠 48k+干净训练，dfnet 靠复数谱重建，均非本体系可达。

5. **真正可改善且对症的是「减少过压制」。** V5 证明大模型能把整体增益从 V3.1 的 −3.18dB
   拉到 −1.26dB（越接近 0 越好，v0.1 是 +0.09），说明过压制是**容量/训练可改善项**。
   V4 −5.57dB 压得最狠，直接导致 SI-SNR 7.60（唯一低于 noisy）——这解释了 V4「PESQ 高但
   STOI/SI-SNR 塌」的本质：靠极端过压制换 PESQ。**对 V3.1/V4，`--oversuppress-weight`
   是最对症的工具**，比 gain-smooth 性价比高。

6. **架构层面的天花板**：要追上 v0.2 的 SI-SNR(12.58)/STOI(0.81)，本质缺的是容量 +
   更高采样率 + 更多 band，或 dfnet 式的相位重建能力——这些是结构差异，不是调 loss 能补的。

> 数据文件：`evaluation/results_50/eval_20260811_194156.csv`（v3.1 + v0.2）、
>          `results_50/audio/*_v3|_xiph_48k|_v5|_dfnet|_rnnoise01.wav`；
>          诊断脚本：`diag_gain_distribution.py`（已扩为 6 模型通用版）。

---

## 附录 B：rnnoise v0.1 vs v0.2 输入特征对比（2026-08-11 追加）

> 依据：v0.2 特征构成读自本地 `xiph_rnnoise/src/denoise.c` 的 `compute_frame_features`
>       与 `denoise.h`（`NB_FEATURES = 2*NB_BANDS+1`）；v0.1 特征构成依据 xiph rnnoise
>       v0.1 的 `denoise.c`（`NB_FEATURES = NB_BANDS + 3*NB_DELTA_CEPS + 2`）。
>       本地 xiph 为 16k 适配版（NB_BANDS=22, FRAME_SIZE=160），但特征"组织方式"与
>       v0.2 上游一致，故用于对比结构差异成立。

### B.1 特征维度总览

| 版本 | 频带数 | 特征维度公式 | 维度值 |
|---|---|---|---|
| v0.1 | 22 band | `NB_BANDS + 3*NB_DELTA_CEPS + 2` = 22 + 3×6 + 2 | **42** |
| v0.2 | 22 band | `2*NB_BANDS + 1` = 2×22 + 1 | **45** |

> 注：以上是 22-band 口径（本项目 16k 与 xiph 48k 主线均为 22 band）。band 数本身
>    v0.1/v0.2 相同；差异在"每帧带哪些特征"，不在 band 数。

### B.2 逐项特征构成对比

**v0.1（42 维）—— 显式塞入时间导数：**

| 段 | 内容 | 维度 |
|---|---|---|
| ① | BFCC（band 对数能量的 DCT，即倒谱系数） | 22 |
| ② | 前 6 个倒谱的一阶差分（Δ delta-cepstrum） | 6 |
| ③ | 前 6 个倒谱的二阶差分（ΔΔ delta-delta） | 6 |
| ④ | pitch 相关（前 6 band 的基音互相关 DCT） | 6 |
| ⑤ | pitch period + 一个特征（周期/gain 类） | 2 |
| 合计 | | **42** |

**v0.2（45 维）—— 去掉显式差分，改由 conv 前端学时序：**

| 段 | 内容 | 维度 | 代码位置 |
|---|---|---|---|
| ① | BFCC：`dct(features, Ly)`，Ly=band 对数能量 | 22 | `features[0..21]` |
| ② | pitch 互相关的 DCT：`dct(&features[NB_BANDS], Exp)` | 22 | `features[22..43]` |
| ③ | pitch period：`.01*(pitch_index-300)` | 1 | `features[44]` |
| 合计 | | **45** | |

### B.3 核心区别（三点）

1. **v0.1 有 delta / delta-delta 倒谱，v0.2 完全删除。**
   v0.1 用 ②③ 两段共 12 维显式编码"倒谱随时间的一/二阶变化"（手工时间导数）。
   v0.2 把它们全砍掉，改为让 **conv1（前端两层 Conv1d）从相邻帧自动学时序变化**。
   即：时间动态从"手工特征"变成"网络自学"。

2. **v0.2 把 pitch 互相关从 6 维扩到全 22 band。**
   v0.1 只取前 6 个 band 的基音互相关（`3*NB_DELTA_CEPS` 里的一部分口径），
   v0.2 对全部 22 band 都算 pitch 相关再做 DCT（`dct(&features[NB_BANDS], Exp)`，22 维），
   基音谐波结构的信息量更足。

3. **维度 42→45**：删 12 维差分、pitch 相关 6→22（+16），净变化即 −12+16−… 落到 45。
   本质是"减手工时间特征、增频域/基音信息 + 靠 conv 补时序"。

### B.4 对本项目的意义

- **本项目（V3/V3.1/V4/V5）用的是 v0.2 口径的 45 维特征**（`NB_FEATURES = 2*NB_BANDS+1`，
  见 `rnnoise_16k.py` 的 `NB_FEATURES`），即已对齐 v0.2、无 delta-cepstrum。
- 之前一度考虑"补 delta-cepstrum 特征"提升效果——**这是逆 v0.2 演进方向的**：v0.1→v0.2
  正是删掉 delta 改用 conv 前端，同时 PESQ 从 1.93 升到 2.07。所以补 delta 不可取，
  正确方向是保留 v0.2 特征 + 强化 conv 前端时序建模（v3.1 的两层 conv 即此思路）。
- v0.2 用 conv 学时序而非手工差分，也解释了为何 v3.1（两层 conv，感受野 5 帧）比
  v3（单层，3 帧）PESQ 更高：更宽的 conv 感受野替代了 v0.1 显式差分的作用。

> Content was rephrased for compliance with licensing restrictions.
