# mini-cog

最小认知架构，手机 + 免费 Colab 实现。

## 总路线

1. 混合序列引擎（SSM + Attention）
2. 外部记忆
3. 世界模型
4. 持续学习
5. 内在动机 + 元认知
6. 全局工作空间
7. 行动闭环

## 阶段 1：混合序列引擎 ✅

**架构**：3 SSM → 1 Attention → 3 SSM，d_model=128，约 1.2M 参数

**任务**：Copy（half_len=32，序列长 65，500 步）

**结果**（T4 GPU，seed=42）：

| 模型 | 参数量 | acc | 训练时间 |
|---|---:|---:|---:|
| 纯 SSM | 1,176,xxx | 0.0225 | 189s |
| 纯 Attention | 509,616 | 1.0000 | 12s |
| Hybrid (3:1:3) | 1,221,184 | 1.0000 | 162s |

**结论**：
- 纯 SSM 无精确检索能力，Copy 任务失败
- 纯 Attention 短序列高效，但受窗口限制
- Hybrid 兼顾流式与检索，序列变长时优势才显现

**已踩坑记录**：
- MambaBlock 串行扫描 → 改并行扫描，提速约 2.7x
- `_init_weights` 会覆盖 A_log，导致 SSM 失效
- Colab Python 3.13 + torch 默认无 CUDA，需切 T4 GPU

**下一步**：阶段 2 外部记忆