# 基于 Transformer 双向编码器的文本要点抽取方法（项目当前版本说明）

摘要
- 本文提出一种面向结构化打分任务的文本要点抽取方法。方法以本地 BERT（Transformer 双向编码器）为语义编码 backbone，对输入摘要先进行句级切分与编码，再通过可学习的门控过滤（filter）在句级-特征级联合维度上完成重要性重加权，最后由多任务评估器（assessor heads）输出 5 个二分类目标：THREAT_up、THREAT_down、citizen_impact、PF_score、PF_US。该方法兼顾句级解释性与端到端可训练性，并通过阶段性解冻策略稳定优化预训练模型参数。当前实现与实验均基于 `training-3-2-*.py` 脚本。

关键词：BERT、Transformer、要点抽取、多任务学习、句级池化、门控过滤、渐进解冻

## 1 方法论设计

### 1.1 问题定义与数据建模

#### (1) 问题描述
方法旨在对给定的英文摘要（summary）进行多标签分类，判断其在若干个任务维度上的相关性。在每个任务维度上，对应一个二元标签（0/1），其中：
- 1 表示摘要与该任务相关
- 0 表示不相关

#### (2) 输入
每条样本的输入包含两部分：
1. 文本输入（summary）：
   - 一段英文摘要，通常由若干句子组成，如：
     > "We propose a novel deep learning model for text classification, achieving state-of-the-art performance on benchmark datasets."
   - 可建模为词序列 \( \mathbf{x} = \{w_1, w_2, \dots, w_n\} \)，其中 \( w_i \) 是单词或子词（如 BERT 的 WordPiece）。
2. 标签（ground truth）：
   - 一个 m 维的二元向量 \( \mathbf{y} = (y_1, y_2, \dots, y_m) \)，其中 \( y_j \in \{0, 1\} \) 表示该摘要是否属于第 \( j \) 个类别。
   - 示例：\( m = 5, \mathbf{y} = (1, 0, 1, 0, 0) \) 表示该摘要在 5 个维度上，与 1、3 相关，但与 2、4、5 无关。

#### (3) 输出
模型的输出是一个 m 维逻辑值（logits）向量：
\[
\mathbf{z} = (z_1, z_2, \dots, z_m)
\]
其中：
- 每个 \( z_j \) 是一个实数（无约束范围），表示模型对第 \( j \) 个类别的原始置信度。
- 通过 Sigmoid 函数 \( \sigma(z_j) = \frac{1}{1 + e^{-z_j}} \) 转换为概率 \( p_j \in [0, 1] \)，用于最终的分类决策。
- 最终预测：
  \[
  \hat{y}_j = \begin{cases}
  1 & \text{if } p_j \geq \tau \\
  0 & \text{otherwise}
  \end{cases}
  \]
  其中 \( \tau \) 是阈值（通常取 0.5，但可在验证集上调优以优化 F1 或 Precision/Recall）。

#### (4) 目标函数
由于是多标签分类，采用二元交叉熵（Binary Cross-Entropy, BCE）作为损失函数，对每个类别独立计算：
\[
\mathcal{L}(\theta) = -\frac{1}{m} \sum_{j=1}^m \left[ y_j \log p_j + (1 - y_j) \log (1 - p_j) \right]
\]
其中：
- \( p_j = \sigma(z_j) \) 是模型预测的概率
- \( y_j \) 是真实标签
- 优化目标是最小化所有 m 个类别的平均 BCE 损失；本项目在实现中引入类别不平衡权重 \( pos\_weight = \sqrt{\frac{neg}{pos}} \)。

#### (5) 下游分析
完成模型训练后，可进行以下分析：
- 按类别计算：对每个标签单独计算 Precision、Recall、F1、AUC-ROC、AUPRC。
- 整体评估：
  - Micro-F1：所有类别的预测合并后计算 F1（适用于类别不平衡）。
  - Macro-F1：先计算每个类别的 F1，再取平均（平等看待所有类别）。
- 阈值调整：若某些类别样本较少，可调整 \( \tau \) 以优化特定指标；本项目通过在验证集上沿 PR 曲线选择最大 F1 的阈值，并在测试集复用。

### 1.2 句级建模与语义表示

为在保持句级可解释性的同时获得稳健的文档级表征，本工作采用“句级编码 + 面向句向量的下游聚合”的两阶段设计。具体如下。

1) 句子切分与预处理
- 给定一段摘要字符串 s，先进行轻量清洗（去除字符“*”），随后采用基于正则的句界识别
  \[
  \Phi: s \mapsto (\tilde{s}_1, \tilde{s}_2, \dots, \tilde{s}_n), \qquad \text{regex: } [\\.;]\\s
  \]
  得到句子序列。该启发式切分在通用英文摘要上具有较好适用性；对包含缩写（如“e.g.”）的极端情形，未来可引入更强的断句器（如 Punkt/SpaCy）以进一步降低误切分概率。

2) 子词化与张量化
- 每个句子 \( \tilde{s}_i \) 通过本地 BERT tokenizer（路径 `model/bert-base-uncased`）映射为子词序列，并统一填充/截断至固定长度 \( L=\texttt{MAX\_LEN}=256 \)。记
  \[
  X_i = (t_{i1}, \dots, t_{iL}), \quad m_i = (m_{i1}, \dots, m_{iL}) \in \{0,1\}^L,
  \]
  其中 \( m_i \) 为 attention mask，指示有效位置（1）与 padding（0）。将所有句子按“句子为批维”堆叠，得到张量 \( X \in \mathbb{R}^{n \times L} \)、\( M \in \{0,1\}^{n \times L} \)。

3) BERT 编码（句子作为批维并行）
- 采用本地预训练的 `BertModel` \( f_\theta \) 对句子批进行编码：
  \[
  H = f_\theta(X, M) \in \mathbb{R}^{n \times L \times h},
  \]
  其中 \( h \) 为隐藏维度，\( H_{i:\,:} \) 是第 \( i \) 个句子的 token 级隐状态。该设计将“句子数 n”作为 mini-batch 维度以充分利用并行计算资源。

4) 基于 mask 的句向量（句级池化）
- 为消除长度与填充的影响，对每个句子做基于 mask 的均值池化，得到句向量 \( e_i \in \mathbb{R}^{h} \)：
  \[
  e_i = \frac{\sum_{k=1}^{L} m_{ik} H_{ik}}{\sum_{k=1}^{L} m_{ik} + \varepsilon}, \qquad E = (e_1, \dots, e_n)^\top \in \mathbb{R}^{n \times h},
  \]
  其中 \( \varepsilon \) 为数值稳定项。与直接采用 [CLS] 表征相比，掩码均值在跨领域/长句场景下更稳定，且可弱化单一位置表征的偏置。

5) 设计动机与性质（学术角度）
- 长度无关与鲁棒性：掩码均值显式忽略 padding 位，天然适配变长句子；在分布漂移（摘要风格/长度变化）下表现更稳健。
- 结构保真与可解释性：保留句子级序列 \( (e_1, \dots, e_n) \) 进入后续模块（见 1.3），支持在句级粒度上进行注意/门控与可视化解释。
- 复杂度与并行性：以句子为批维，编码复杂度约为 \( \mathcal{O}(n \cdot L \cdot h) \)，能充分利用 GPU 并行；同时避免将整段长文直接拼接导致的超长序列开销。
- 与替代方案对比：相较 CLS 池化与 token 级注意池化，掩码均值引入的可训练参数更少、对小样本更稳定；而后续的门控（1.3 节）在句级-特征级维度上补足了选择性表达能力。

6) 与实现的一致性
- 切分与编码过程对应 `model/text_assessment/text_dataset.py`（`to_literal` 与按句 tokenization）；
- 编码与掩码均值实现于 `model/text_assessment/text_assessor.py` 的 `_forward`：`BertModel` 输出 `last_hidden_state` 后按 `attention_mask` 做加权均值；
- 该阶段输出 \( E \in \mathbb{R}^{n \times h} \) 作为后续门控过滤（1.3 节）的输入。

### 1.3 任务相关要点抽取（门控过滤）

为在句级结构上实现任务相关的精细选择，本节提出一种“句级-特征级门控（feature-wise sentence gating）”机制。相较于标准的句级注意力（对句子打标量权重）或基于 [CLS] 的单向聚合，门控在每个特征维度上学习可独立调节的权重，从而在文档级融合前进行更细粒度的筛选与强调。

1) 形式化定义（任务特定门控）
- 记 1.2 节得到的句向量矩阵为 \(E = (e_1, \dots, e_n)^\top \in \mathbb{R}^{n \times h}\)。对第 \(t\) 个任务（输出头）引入参数化门控映射
  \[
  g_{\phi^{(t)}}: \mathbb{R}^{h} \to (0,1)^{h}, \qquad p^{(t)}_i = g_{\phi^{(t)}}(e_i), \quad i=1,\dots,n,
  \]
  其中 \(g_{\phi^{(t)}}\) 为逐句共享参数的前馈网络，输出与输入同维度的元素级概率。将所有句子的门控向量按行堆叠，得 \(P^{(t)} \in (0,1)^{n \times h}\)。本项目中，\(t\in\{1,2\}\)（两路任务头，随后在 1.4 节拼接得到 5 维 logits）。

2) 参数化实现（行式 MLP + 归一化 + Sigmoid）
- 对每个句向量 \(e_i\) 应用同构 MLP：
  \[
  p^{(t)}_i = \sigma\big(W^{(t)}_2\, \mathrm{Drop}\big(\mathrm{LN}(\mathrm{LReLU}(W^{(t)}_1 e_i + b^{(t)}_1))\big) + b^{(t)}_2\big),
  \]
  其中 \(\mathrm{LN}\) 为 LayerNorm，\(\mathrm{LReLU}\) 为 LeakyReLU，\(\mathrm{Drop}\) 为 Dropout，\(\sigma\) 为 Sigmoid。该结构对应实现中的 `Linear → LayerNorm → LeakyReLU → Dropout → Linear → Sigmoid`，隐藏维按倍率扩展后再还原至维度 \(h\)。

3) 维度化加权与归一融合（文档级表示）
- 对第 \(t\) 个任务，文档级表示 \(v^{(t)} \in \mathbb{R}^{h}\) 通过对句维进行“逐维加权平均”获得：
  \[
  v^{(t)} = \frac{\sum_{i=1}^{n} p^{(t)}_i \odot e_i}{\sum_{i=1}^{n} p^{(t)}_i + \varepsilon}
  \quad \Leftrightarrow \quad
  v^{(t)}_d = \frac{\sum_{i=1}^{n} p^{(t)}_{i,d} e_{i,d}}{\sum_{i=1}^{n} p^{(t)}_{i,d} + \varepsilon},\; d=1,\dots,h,
  \]
  其中 \(\odot\) 表示逐元素乘法，分母为元素级的权重和，\(\varepsilon\) 为数值稳定项（实现中取 \(10^{-8}\)）。矩阵形式可写为 \(v^{(t)} = \frac{(E \odot P^{(t)})^\top \mathbf{1}}{(P^{(t)})^\top \mathbf{1} + \varepsilon}\)。

4) 方法性质与对比（学术分析）
- 细粒度选择性：与标量句级注意力不同，门控在每个特征维度上独立学习重要性，允许“同一句的不同子空间维度对不同任务产生不同贡献”。
- 非竞争性与可稀疏化：Sigmoid 激活不施加全局归一化约束（对句/对维均非必须和为 1），有利于在噪声情形下共同抑制无关维度；Dropout 与 LayerNorm 提升泛化与训练稳定性。
- 可解释性：\(P^{(t)}\) 可视为“句×特征”的显著性热力图；句级 saliency 可定义为 \(s^{(t)}_i = \frac{1}{h}\sum_{d} p^{(t)}_{i,d}\)，维度级 saliency 为 \(r^{(t)}_d = \frac{1}{n}\sum_{i} p^{(t)}_{i,d}\)。
- 复杂度：相较 token 级注意力/加性注意力，门控只在句向量上操作，复杂度约 \(\mathcal{O}(n \cdot h \cdot r)\)（\(r\) 为隐藏倍率），便于在长摘要和多句场景下部署。
- 与 softmax 注意力之别：softmax 往往在句维产生竞争性归一化，可能过度抑制次要但必要的支持证据；门控以逐维独立概率估计替代竞争归一，允许信息的并行保留与抑制。

5) 与实现的一致性
- 对应代码：`model/text_assessment/text_assessor.py` 中的 `_build_filter`（门控 MLP）与 `_forward`（维度化加权均值），且每个任务头拥有独立的门控参数（`self.outputs[k]['filter']`）。
- 数值细节：分母加 \(\varepsilon\) 保证反向传播的数值稳定；门控在每个任务头内独立计算，随后在 1.4 节经各自的评估器映射到任务特定的 logits 维度。

### 1.4 多任务评估与损失设计

本节从映射结构、概率化与决策、损失与不平衡处理以及评估指标四个方面对多任务设置进行形式化描述，并与实现细节保持一致。

1) 评估器结构与映射
- 对于 1.3 节得到的任务特定文档表示 \(v^{(t)} \in \mathbb{R}^{h}\)，为每个任务头 \(t\) 定义一组独立的判别映射（两层前馈）：
  \[
  a_{\psi^{(t)}}: \mathbb{R}^{h} \to \mathbb{R}^{k_t},\qquad
  a_{\psi^{(t)}}(v) = W^{(t)}_4\, \mathrm{Drop}\big(\mathrm{ReLU}(W^{(t)}_3 v + b^{(t)}_3)\big) + b^{(t)}_4,
  \]
  其中 \(k_1=3, k_2=2\)；对应实现为 `Linear → ReLU → Dropout → Linear`（见 `_build_assessor`）。两路头的输出在类别维拼接：
  \[
  z = \big[\, a_{\psi^{(1)}}(v^{(1)})\,\big\Vert\, a_{\psi^{(2)}}(v^{(2)})\,\big] \in \mathbb{R}^{5}.
  \]
  类别顺序在实现中固定为
  \[
  [\texttt{THREAT\_up},\, \texttt{THREAT\_down},\, \texttt{citizen\_impact},\, \texttt{PF\_score},\, \texttt{PF\_US}],
  \]
  即第一路头产生前三维、第二路头产生后两维。该顺序需与训练脚本中的 `LABELS`、损失与精度计算函数严格对齐。

2) 概率化与阈值化决策
- 将 logits \(z_j\) 通过 Sigmoid 得到边缘概率 \(p_j=\sigma(z_j)\in[0,1]\)。
- 训练过程中为监控简便，`caculate_accuracy` 采用固定阈值 \(\tau=0.5\) 的逐维命中率作为近似指标；
- 最终决策采用“验证集阈值调优”：在验证集上对每一维利用精确率-召回率曲线 \((P(t),R(t))\) 计算 \(\mathrm{F1}(t)=\tfrac{2P(t)R(t)}{P(t)+R(t)}\)，取使 F1 最大的阈值 \(\tau_j^*\)，并在测试集复用（见各 `training-3-2-*.py` 脚本的阈值选择与落盘）。

3) 损失函数与类别不平衡
- 采用“带 logits 的二元交叉熵”（BCE-with-logits）独立监督每一维。设批内第 \(i\) 个样本的标签向量为 \(y_i\in\{0,1\}^m\)、logits 为 \(z_i\in\mathbb{R}^m\)，类别权重 \(w \in \mathbb{R}^m\) 由训练集不平衡比 \(w_j=\sqrt{\tfrac{\mathrm{neg}_j}{\mathrm{pos}_j}}\) 给出（`tools.get_pos_weights`）：
  \[
  \ell(y_i,z_i;w)\;=\;\sum_{j=1}^{m} \mathrm{BCE\_logits}(z_{i,j}, y_{i,j};\; \texttt{pos\_weight}=w_j),\qquad
  \mathcal{L}\;=\;\sum_{i\in \mathcal{B}} \ell(y_i,z_i;w).
  \]
  实现中按样本求和后在 epoch 级别按批数取平均，与经验风险最小化等价（尺度因子不影响最优解）。

4) 评估指标与报告
- 单维指标：AUC-ROC、AUPRC、F1（在各自 \(\tau_j^*\) 下）。
- 宏/微聚合：宏平均对各维指标取算术平均（类别均权）；微平均将所有维度、样本展平后计算（对频繁类别更敏感）。
- 实现对验证/测试分别导出 `metrics_val.csv`、`metrics_test.csv`，并在 `preds_test.csv` 中记录逐维真值、概率与二值化预测，便于后续误差分析与可视化。

5) 与实现的一致性
- 评估器结构对应 `text_assessor._build_assessor` 与前向中的拼接；
- 损失与不平衡处理对应 `text_assessor.compute_loss`（逐维 BCE-with-logits 与 `pos_weight`）；
- 决策与度量对应各 `training-3-2-*.py` 中的阈值调优与指标计算；
- 顺序一致性要求：`LABELS`、logits 拼接顺序、`compute_loss`/`caculate_accuracy` 访问索引必须严格一致，否则将造成监督错配。

## 2 网络设计

### 2.1 主干编码器（Backbone）

本工作采用双向 Transformer 语言模型 BERT 作为语义编码骨干，以本地离线权重进行初始化并在下游任务上进行渐进式微调。记 tokenizer 为 WordPiece，编码器为 `BertModel`，其参数以 \(\theta\) 表示，隐藏维度为 \(h\)。

1) 参数化与初始化
- 词表与子词化：采用与权重配套的本地 tokenizer（`model/bert-base-uncased`），确保子词边界与嵌入矩阵一致；
- 模型结构：`BertModel` 包含嵌入层、L 层 Transformer 编码器与池化器（`pooler`），隐藏维度 \(h=\texttt{config.hidden\_size}\)；
- 权重来源：通过 `from_pretrained(PRETRAINED_MODEL_NAME)` 自本地目录载入，避免在线下载导致的版本漂移，保证实验可复现性与离线可用性。

2) 前向计算与输出
- 对于单句子张量 \(X_i\) 与掩码 \(m_i\)（见 1.2 节），编码器返回 token 级隐状态 \(H_{i}\in\mathbb{R}^{L\times h}\)；
- 本工作不使用默认 `pooler\_output` 作为文档表示，而是在 1.2 节采用基于掩码的均值池化获取句向量 \(e_i\)，再在 1.3 节通过门控机制获得任务特定的文档表示；
- 该设计规避了 CLS 单点表示的潜在偏置，并与后续句级门控的结构性需求一致。

3) 微调策略与冻结计划
- 采用“渐进解冻”的微调方案：先仅训练输出头参数，再依次解冻 `pooler`、最后一层、倒数第 3→1 层、倒数第 7→3 层，最终解冻剩余编码层与嵌入层（详见各 `training-3-2-*.py` 的 `PHASES` 配置）；
- 优化器选用 AdamW，并对不同子模块设置差异化学习率（如 `outputs`、`pooler`、`enc_last1`、`enc_last3to1`、`enc_last7to3`、`enc_rest`、`embeddings`），以减小灾难性遗忘并提升收敛稳定性；
- 随机种子在模块与脚本双重设置为 20040508，以降低随机性对实验的影响。

4) 本地化与稳定性考量
- 本地 checkpoint：`PRETRAINED_MODEL_NAME='./model/bert-base-uncased'`，保证 tokenizer、配置与权重的一致性；
- 数值稳定：训练中配合 LayerNorm、Dropout 与小学习率策略；句向量与门控阶段引入 \(\varepsilon\) 以避免除零；
- 资源友好：以句子为批维进行并行编码，相较直接拼接全摘要为超长序列，可显著降低显存占用并提升吞吐。

### 2.2 句级表示与池化

为从 token 级隐状态获得稳定且可解释的句级表示，本文采用基于掩码的均值池化（masked mean pooling）。设 1.2 节中得到的编码张量为 \(H\in\mathbb{R}^{n\times L\times h}\) 与注意掩码 \(M\in\{0,1\}^{n\times L}\)，其中 \(n\) 为句子数，\(L\) 为每句的统一长度（padding/truncation 后），\(h\) 为隐藏维度。记第 \(i\) 句的隐状态与掩码分别为 \(H_i\in\mathbb{R}^{L\times h}\)、\(m_i\in\{0,1\}^{L}\)。

1) 掩码均值池化的定义
- 对每个句子，句向量 \(e_i\in\mathbb{R}^{h}\) 定义为：
  \[
  e_i = \frac{\sum_{k=1}^{L} m_{ik}\, H_{ik}}{\sum_{k=1}^{L} m_{ik} + \varepsilon},\qquad i=1,\dots,n,
  \]
  其中分子为对有效 token 的逐元素求和，分母为有效 token 计数，\(\varepsilon\) 为数值稳定项（实现中取 \(10^{-8}\)）。将所有 \(e_i\) 沿句维堆叠得到句级表示矩阵
  \[
  E = (e_1,\dots,e_n)^{\top} \in \mathbb{R}^{n\times h}.
  \]
  该 \(E\) 将在 1.3 节作为门控过滤的输入。

2) 性质与理论考量
- Padding 不变性：当 \(m_{ik}=0\) 时，对应位置对分子与分母均无贡献，从而对 \(e_i\) 无影响，保证对不同原始句长的一致性；
- 可微与稳定性：均值池化在有效位置上对 \(H_{ik}\) 的梯度为常数因子 \(1/(\sum_k m_{ik}+\varepsilon)\)，有利于稳定反传；\(\varepsilon\) 抑制极短句或空句导致的数值异常；
- 统计稳健性：相较单点表征（如 [CLS]），均值能降低个别 token 噪声的影响；在样本量有限时，较低的可训练自由度通常带来更好的泛化；
- 复杂度：时间复杂度 \(\mathcal{O}(n\,L\,h)\)，与一次线性缩减同量级，适配长摘要与多句并行。

3) 与替代方案的对比
- [CLS] 池化：将信息集中于单个位置，易受预训练域偏置与句式差异影响；
- token 级注意池化：可学习性强但引入额外注意参数与归一化竞争，易在小样本下过拟合；
- 本文取 masked mean 作为“稳健基座”，并在 1.3 节以句级-特征级门控补足选择性表达能力。

4) 实现对齐
- 对应实现位于 `model/text_assessment/text_assessor.py` 的 `_forward`：以 `attention_mask` 为 \(M\)，对 `last_hidden_state` 做掩码加权求和并除以有效长度；
- 输出形状与记号一致：`sentence_embeddings` 即 \(E\in\mathbb{R}^{n\times h}\)，作为后续门控与任务头的输入。

### 2.3 门控过滤（Filter Head）
- 结构：`Linear → LayerNorm → LeakyReLU → Dropout → Linear → Sigmoid`，输出与输入同维度的权重矩阵 `P ∈ (0,1)^{n_sentences × hidden}`。
- 融合：按元素乘法得到加权句向量并在句维求和，随后按权重和归一，形成文档级表示 `v ∈ R^{hidden}`。该过程在每个任务头内独立执行，支持任务特定的注意选择。

### 2.4 评估器（Assessor MLP）
- 结构：`Linear → ReLU → Dropout → Linear`，将 `v` 投射到任务特定的分类维度；当前配置为两路 MLP 分别输出 `[3]` 与 `[2]`，拼接后得到 `[5]` 维 logits。
- 输出映射顺序固定：`[THREAT_up, THREAT_down, citizen_impact, PF_score, PF_US]`。若修改任务维度，需同步更新 `TextAssessor(n2=...)`、`compute_loss(...)`、`caculate_accuracy(...)` 与各训练脚本中的 `LABELS` 顺序与键名。

### 2.5 前向与 Batch 约定（本项目特有）
- Dataloader 的 `collate_fn` 原样返回“样本字典的列表”（不将张量在 batch 维拼接）。
- `TextAssessor.forward` 会遍历该列表，逐样本调用 `_forward`，再在 batch 维堆叠为 `[batch, 5]`。这与常见的 `[batch, seq_len]` 直接送入 encoder 的范式不同，是本项目保持句级结构和解释性的关键约定。

## 3 关键实现与实验流程

### 3.1 数据管道（`model/text_assessment/text_dataset.py`）
- `to_literal()`：`summary` 去 `*` 后用正则 `[\.;]\\s` 分句；
- 编码：按句使用本地 tokenizer，padding/truncation 到 `MAX_LEN`；
- 返回：字典含 `input_ids`、`attention_mask`（形如 `[n_sentences, MAX_LEN]`）及 5 个 float 标签；
- `collate_fn`：原样返回 Python 列表，供模型逐样本处理。

### 3.2 模型与训练循环（`model/text_assessment/text_assessor.py`）
- `TextAssessor`：包含多个输出头 `outputs`，每头含 `filter` 与 `assessor`；
- `_forward`：BERT 编码 → mask 均值句向量 → 门控过滤 → 任务评估；
- `compute_loss`：逐维 BCE-with-logits，并注入 `pos_weight`（对不平衡敏感）；
- `caculate_accuracy`：逐维 0.5 阈值化命中率，平均为样本准确率；
- `save/loads`：保存/加载包含结构参数与权重的包，便于脱离代码结构直接复现；
- `predict_one`：对单段文本返回 5 维概率，键名为 `pred.values.*`。

### 3.3 训练脚本与设置（`training-3-2-*.py`）
- 依赖：Python ≥ 3.11；`pip install torch transformers pandas matplotlib tqdm scikit-learn`（`tools.py` 中的 OpenAI 客户端不用于训练）。
- 分割：各脚本基于 `random_split`，比例略有差异：
  - `training-3-2-10.py`：`[3/15, 2/15, 10/15]`；
  - `training-3-2-15.py`：`[.15, .10, .75]`；
  - `training-3-2-20.py`：`[.12, .08, .80]`；
  - `training-3-2-25.py`：`[.10, 1/15, 5/6]`。
- 训练：CUDA 自动启用（如可用）；按 `PHASES` 渊进解冻，分组学习率（键见 1.5），权重衰减阶段值与最终值区分；以验证集 macro-AUPRC 早停。
- 阈值：在验证集上用 PR 曲线最大化 F1 获取每标签阈值；测试集评估复用该阈值；
- 产物：各脚本输出至对应目录（如 `./3-2-15/`）：`best_model.pth`、`thresholds.json`、`metrics_val.csv`、`metrics_test.csv`、`preds_test.csv`、`curves.svg`。

### 3.4 类别不平衡与稳健性
- 类别权重：`tools.get_pos_weights` 计算负正样本比，损失中取其平方根以缓解梯度不稳定；
- 渐进解冻：从仅训练输出头到逐层解冻 encoder 与 embeddings，降低灾难性遗忘与过拟合风险；
- 正则化：Filter/Assessor 中的 Dropout 与 LayerNorm 提升泛化与数值稳定性。

### 3.5 推理与复现
- 加载：`TextAssessor.loads(path, device)` 加载训练好的权重；
- 单样本：`predict_one(text, model, tokenizer, MAX_LEN, device)` 返回键 `pred.values.THREAT_up` 等 5 维概率；
- 本地依赖：使用本地预训练模型与 tokenizer（`PRETRAINED_MODEL_NAME`），离线环境下可直接运行。

## 4 参考路径与文件
- `model/text_assessment/__init__.py`：`SEED`、`PRETRAINED_MODEL_NAME`、`SAVED_MODEL`、`MAX_LEN`；
- `model/text_assessment/text_dataset.py`：`TextDataset` 与 `collate_fn`；
- `model/text_assessment/text_assessor.py`：模型、训练/评估、推理；
- `training-3-2-*.py`：分阶段训练与评估脚本；
- `tools.py`：类别权重与通用工具（API 示例未用于训练）。

结论
- 该方法以 BERT 为基础，通过句级-特征级门控实现“可解释的要点抽取 + 多任务评估”，结合渐进解冻与类别加权，在类不平衡、长摘要等场景下具备良好的可迁移性与鲁棒性。当前版本以 `training-3-2-*.py` 为标准实验流程，统一在验证集上进行阈值调优并导出完整评估产物，便于复现与对比实验。
