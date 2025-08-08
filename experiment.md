# 仅基于80K数据集的全新实验方案

目的
- 基于 `analysis_result_simple_format_80000_with_summary.xlsx` 的大规模样本，系统评估“句级-特征级门控 + 多头评估”的方法在五个二分类目标上的有效性、可扩展性与可解释性，并与多种强基线进行对比。

1. 研究问题与假设
- RQ1 有效性：逐维门控的句级聚合是否显著优于（a）句级均值/最大池化，（b）句级标量注意力，（c）仅 CLS 池化？
- RQ2 训练策略：渐进解冻是否优于一次性全量微调与仅训练任务头？
- RQ3 数据规模与鲁棒性：在 80K 内部子采样（10K/20K/40K/80K）下，方法是否呈现单调增益与稳定阈值敏感性？
- RQ4 可解释性：门控权重是否与人工关键句一致，支持错误类型归因？

2. 数据集与划分（固定）
- 数据：`data/analysis_result_simple_format_80000_with_summary.xlsx`，字段：`summary` 与五个 0/1 标签（THREAT_up、THREAT_down、citizen_impact、PF_score、PF_US）。
- 预处理：`TextDataset.to_literal()` 按正则 `[.;]` 分句并去除 `*`；逐句使用本地 tokenizer（`model/bert-base-uncased`）到 `MAX_LEN=256`。
- 划分：按 3/2/20（train/val/test）随机划分，固定 SEED=20040508；若条件允许，分层划分以保持各标签比例。训练全过程仅在训练集上拟合，阈值与早停在验证集上选择，测试集仅用于最终一次评估。

3. 模型与对比设置
- Ours（主方法）：`TextAssessor(n1=2, n2=[3,2])`，每个任务头=Filter(逐维 Sigmoid 门控)+Assessor(MLP)。forward 遵循“batch 为样本字典列表”的约定，逐样本 `_forward` 后堆叠为 `[batch,5]`。
- 统一实现：
  - 句向量：BERT token 隐状态经 mask 均值得到 `[n_sentences, hidden]`；
  - 损失：逐维 `BCEWithLogits`，`pos_weight=sqrt(neg/pos)`（见 `tools.get_pos_weights` 与 `compute_loss`）。
- 基线（最小侵入实现）：
  1) MeanPooling：去除 Filter，句维直接均值 → Assessor。
  2) MaxPooling：去除 Filter，句维逐维最大 → Assessor。
  3) SentenceAttention：句向量上做标量注意力（softmax 权重），句维加权求和 → Assessor。
  4) CLSPooling：不做句级聚合，直接用 BERT `pooler_output`（或首句 CLS）→ Assessor。
- 训练策略对比：
  - Progressive（默认）：阶段性解冻（详见 §4）；
  - FullFT：从首轮起解冻全部 encoder+embeddings；
  - HeadsOnly：仅训练 `outputs`（不解冻 encoder）。
- 消融：
  - 共享 vs 独立 Filter：两个任务头共用一个 Filter；
  - 去除 LayerNorm/Dropout；
  - MAX_LEN：128 vs 256；
  - pos_weight 形式：sqrt(neg/pos) vs 直接 (neg/pos)。

4. 训练协议（80K 标准设置）
- 优化器：AdamW；权重衰减 1e-3（Final 阶段 1e-4）。
- Batch：6；设备优先 CUDA。
- 渐进解冻与学习率（与仓库风格一致，统一 epoch）：
  - Phase1：仅 `outputs`（lr=1e-6），epochs=11；
  - Phase2：+ `pooler`（8e-7），epochs=11；
  - Phase3：+ encoder[-1:]（4e-7），epochs=11；
  - Phase4：+ encoder[-3:-1]（4e-7），epochs=11；
  - Phase5：+ encoder[-7:-3]（4e-7），epochs=11；
  - phase6: + encoder[:-7] (4e-7),epochs=11
  - Final：解冻全部 encoder 与 embeddings，分组 lr={1e-6,1e-6,8e-7,4e-7,2e-7}，epochs=11；
- 验证：每 epoch 在 val 上评估；早停（patience=3，监控 macro-AUPRC）；阈值在 val 上按最大 F1 或 Youden 优化；
- 复现：固定 SEED，重复 3 次报告均值±标准差。

5. 评价指标与统计检验
- 指标（逐标签与宏/微平均）：AUC-ROC、AUPRC、F1（val 调阈，test 固定该阈值）、Acc（与仓库实现的 0.5 阈值同步报告）、Brier 分数（可选）。
- 显著性：bootstrap(5k) 95% CI；配对 DeLong（AUC）与 McNemar（F1/Acc）用于与各基线的统计对比。

6. 大规模分析与鲁棒性
- 规模曲线：在 10K/20K/40K/80K 训练子集上复现实验（其余保持不变），绘制 AUPRC/F1 随样本数曲线。
- 阈值敏感性：对每标签绘制 F1-阈值曲线；
- 数据子群：按摘要句数、文本长度、标签稀疏度等分桶报告性能。

7. 可解释性与误差分析
- 可视化：对若干 TP/FP/FN 样本绘制 Filter 权重热力图，标注高权重句；
- 失败类型：否定/反问、长依存、实体歧义、跨句关系缺失等归纳与案例。

8. 资源与落地
- 代码路径：
  - 数据与加载：`model/text_assessment/text_dataset.py`；
  - 模型与训练：`model/text_assessment/text_assessor.py`；
  - 训练脚本：`80000-main.py`（建议改为固定 8/1/1 划分并移除子抽样训练）；
  - 工具：`tools.py`（`get_pos_weights`）。
- 依赖：`torch, transformers, pandas, matplotlib, tqdm`（`openai` 非训练必需）。
- 产物：保存训练/验证曲线、检查点（每阶段与最终）、测试集指标表 `data/test-metrics.csv` 与逐样本预测 `data/test-preds.csv`。

9. 结果呈现（论文模板）
- 主表：Ours 与四个基线的 AUC/AUPRC/F1（宏/微与逐标签）+ 显著性；
- 消融表：Filter/LayerNorm/Dropout/MAX_LEN/pos_weight 变体；
- 曲线图：训练/验证损失与准确率、规模曲线、阈值敏感性曲线；
- 解释图：门控热力图+关键句高亮。

10. 工作
- 固定划分与评估脚本，跑完基线单次；
- 主方法与消融，重复 3 次；
- 规模曲线、误差分析与可视化；
- 统计检验、整理结果与论文写作、复现清单。
