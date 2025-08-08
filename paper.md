# 基于Transformer双向编码器的文本要点抽取方法

摘要
- 本文提出一种面向结构化打分任务的文本要点抽取方法。方法以本地 BERT（Transformer 双向编码器）为语义编码 backbone，对输入摘要先进行句级切分与编码，再通过可学习的门控过滤（filter）在句级-特征级联合维度上完成重要性重加权，最后由多任务评估器（assessor heads）输出 5 个二分类目标：THREAT_up、THREAT_down、citizen_impact、PF_score、PF_US。该方法兼顾了句级解释性与端到端可训练性，并通过阶段性解冻策略稳定优化预训练模型参数。

关键词：BERT、Transformer、要点抽取、多任务学习、句级池化、门控过滤、迁移学习

## 1 方法论设计

1.1 问题定义与数据建模
- 输入：每条样本含一段英文摘要 `summary`，以及 5 个0/1标签（THREAT_up、THREAT_down、citizen_impact、PF_score、PF_US）。
- 目标：对摘要进行要点抽取与任务相关性评估，输出 5 维逻辑值（后经 Sigmoid 得到概率），用于二分类判定与下游分析。
- 数据来源与格式：Excel 文件位于 `data/analysis_result_simple_format_{3000|80000}_with_summary.xlsx`。数据加载、切分、标注读取见 `model/text_assessment/text_dataset.py`。

1.2 句级建模与语义表示
- 句子切分：对 `summary` 以正则 `[.;]` 进行句界划分，并剔除特殊符号（参见 `TextDataset.to_literal`）。
- Token 化与张量化：使用本地 BERT tokenizer（路径 `model/bert-base-uncased`），对每个句子独立 tokenization，padding 至 `MAX_LEN=256`，得到形状为 `[n_sentences, MAX_LEN]` 的 `input_ids` 与 `attention_mask`。
- 编码：逐句输入 BERT 获得 token 级隐状态，采用基于 mask 的平均池化得到句向量矩阵 `[n_sentences, hidden]`，从而保留句级可解释结构。

1.3 任务相关要点抽取（门控过滤）
- 设计思想：句级重要性不仅体现在句与句之间，也体现在句内不同特征维度的重要性上。为此，引入“特征维度的句级门控”，学习一个在 `[n_sentences, hidden]` 上逐元素的权重矩阵 `probs ∈ (0,1)`，对句向量进行加权平均，形成单一的文档级向量 `[hidden]`。
- 优势：
  - 相比仅句级注意力，细化到特征维度，有助于强调与任务更强相关的子空间维度；
  - 相比简单平均，更具选择性，有助于对噪声句或冗余特征降权，提升任务判别能力与稳健性。

1.4 多任务评估与损失设计
- 多头评估器（assessor heads）：为兼顾任务异质性，使用两路 MLP 头分别输出维度 `[3, 2]`，拼接为 5 维 logits，按顺序映射到 5 个标签。
- 损失函数：对每一维使用 `binary_cross_entropy_with_logits`，并基于类别不平衡引入 `pos_weight`（负正样本比值的平方根；见 `tools.get_pos_weights` 与 `compute_loss`）。总损失为 5 个子损失之和。
- 评价指标：对 5 维概率阈值化（0.5）后，统计逐维命中率的平均作为样本准确率；跨 batch 取均值得到 epoch 级准确率（见 `caculate_accuracy`）。

1.5 迁移学习与训练策略
- 采用“渐进解冻（progressive unfreezing）”稳定训练：先仅训练输出头，再依次解冻 `pooler`、最后一层、倒数第3-1层、倒数第7-3层，最终解冻全部 encoder 层与 embeddings（详见 `3000-main.py`、`80000-main.py`）。
- 优化器：AdamW，小学习率、分组参数配置与权重衰减，阶段间保存曲线与检查点到 `data/` 目录。

## 2 网络设计

2.1 主干编码器（Backbone）
- 使用本地 `BertModel` 加载（`PRETRAINED_MODEL_NAME='./model/bert-base-uncased'`）。隐藏维度 `hidden_size` 由配置文件确定。
- 随机种子统一在模块级设置，以保证可重复性（见 `__init__.py` 及各脚本的 `torch.manual_seed`）。

2.2 句级表示与池化
- Token 级隐状态经 mask 平均得到句向量：对每句在 token 维加权平均，避免 padding 干扰。
- 得到句向量矩阵 `S ∈ R^{n_sentences × hidden}`，作为后续门控与评估的输入。

2.3 门控过滤（Filter Head）
- 结构：两层前馈网络（含 LayerNorm、LeakyReLU、Dropout），输出与输入同维度的 Sigmoid 权重矩阵 `P ∈ (0,1)^{n_sentences × hidden}`。
- 融合：按元素乘法得到加权句向量并在句维求和，随后按权重和归一，形成文档级表示 `v ∈ R^{hidden}`。该过程在每个任务头内独立执行，支持任务特定的注意选择。

2.4 评估器（Assessor MLP）
- 结构：两层全连接 + ReLU + Dropout，将 `v` 投射到任务特定的分类维度；本项目配置为两路 MLP 分别输出 `[3]` 与 `[2]`，拼接后得到 `[5]` 维 logits。
- 输出映射顺序固定：`[THREAT_up, THREAT_down, citizen_impact, PF_score, PF_US]`。修改任务时需同步更新 `TextAssessor(n2=...)`、`compute_loss(...)`、`caculate_accuracy(...)` 中的顺序与键名。

2.5 前向与 Batch 约定
- Dataloader 的 `collate_fn` 返回“样本字典的列表”，`TextAssessor.forward` 会逐样本调用 `_forward`，再在 batch 维堆叠为 `[batch, 5]`。这与常见的 `[batch, seq_len]` 直接送入 encoder 的范式不同，属于本项目的关键约定，便于保持句级结构与解释性。

## 3 关键技术实现

3.1 数据管道与实现要点
- 数据集实现：`model/text_assessment/text_dataset.py`
  - 句级切分：`to_literal()` 使用正则分句；
  - 编码：按句 tokenize 并 padding 到 `MAX_LEN`；
  - 返回：字典含 `input_ids`、`attention_mask`（形如 `[n_sentences, MAX_LEN]`）及 5 个 float 标签；
  - `collate_fn`：原样返回列表，保持上层网络的逐样本处理逻辑。

3.2 模型与训练循环
- 核心模型：`model/text_assessment/text_assessor.py`
  - `TextAssessor` 含多个 `outputs` 头，每头均包含 `filter` 与 `assessor` 两部分；
  - `_forward` 实现：BERT 编码 → mask 均值句向量 → 门控过滤 → 任务评估；
  - `compute_loss`：逐维 BCEWithLogits，并注入 `pos_weight`（对不平衡敏感）；
  - `caculate_accuracy`：逐维阈值化命中率，平均为样本准确率；
  - `save/loads`：保存/加载包含结构参数与权重的包，便于脱离代码结构直接复现。
- 训练脚本：`3000-main.py`、`80000-main.py`
  - 设备自适应 CUDA；固定随机种子；
  - 以阶段为单位解冻 encoder 不同部分，分组设置学习率，训练若干 epoch；
  - 过程中保存损失与准确率曲线到 `data/*.svg`，并按阶段写入检查点 `data/*text_assessor_*.pth` 与最终 `data/text_assessor.pth`；
  - 注意：脚本中 `matplotlib.RcParams.update` 推荐替换为 `matplotlib.rcParams.update`。

3.3 类别不平衡与稳健性
- 类别权重：通过 `tools.get_pos_weights` 计算负正样比，损失中取其平方根以缓解梯度不稳定；
- 渐进解冻：从仅训练任务头到逐层解冻 encoder 与 embeddings，降低灾难性遗忘与过拟合风险；
- 丢弃与归一：Filter/Assessor 中的 Dropout 与 LayerNorm 提升泛化与数值稳定性。

3.4 推理与部署
- 单样本推理：`predict_one(text, model, tokenizer, max_len, device)`，直接返回 5 维概率；
- 模型加载：`TextAssessor.loads(save_model, device)` 可从 `data/text_assessor.pth` 恢复；
- 本地依赖：使用本地预训练模型与 tokenizer，路径见 `model/text_assessment/__init__.py` 的 `PRETRAINED_MODEL_NAME`，离线环境可直接运行。

附：实验与实现建议
- 环境与依赖：Python 3.11+；`pip install torch transformers pandas matplotlib tqdm openai`；
- 数据放置：Excel 于 `data/` 目录；
- 训练命令：`python 3000-main.py` 或 `python 80000-main.py`；
- 修改任务维度时，严格同步更新模型头、损失与指标函数的键名与顺序映射。

参考实现与代码路径
- `model/text_assessment/__init__.py`：SEED、PRETRAINED_MODEL_NAME、SAVED_MODEL、MAX_LEN；
- `model/text_assessment/text_dataset.py`：TextDataset 与 collate_fn；
- `model/text_assessment/text_assessor.py`：模型、训练/评估、推理；
- `3000-main.py`、`80000-main.py`：分阶段训练脚本；
- `tools.py`：类别权重与通用工具（DeepSeek API 样例未用于训练）。

结论
- 该方法以 BERT 为基础，通过句级-特征级门控实现“可解释的要点抽取 + 多任务评估”，结合渐进解冻与类别加权，在类不平衡、长摘要等场景下具备良好的可迁移性与鲁棒性。
