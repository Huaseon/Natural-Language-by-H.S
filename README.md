# TextAssessor: 基于BERT的文本评估模型

- [1. 概述](#1-概述)
  - [1.1 项目背景](#11-项目背景)
  - [1.2 技术特点](#12-技术特点)
- [2. 理论基础](#2-理论基础)
  - [2.1 BERT预训练模型](#21-bert预训练模型)
  - [2.2 双任务学习理论](#22-双任务学习理论)
  - [2.3 注意力过滤机制](#23-注意力过滤机制)
- [3. 模型架构](#3-模型架构)
  - [3.1 整体架构](#31-整体架构)
  - [3.2 组件详解](#32-组件详解)
  - [3.3 数学表示](#33-数学表示)
- [4. 算法原理](#4-算法原理)
  - [4.1 损失函数](#41-损失函数)
  - [4.2 优化策略](#42-优化策略)
  - [4.3 评估指标](#43-评估指标)
- [5. 数据预处理](#5-数据预处理)
  - [5.1 原始数据处理流程](#51-原始数据处理流程)
  - [5.2 目标文本标注](#52-目标文本标注)
  - [5.3 训练数据构建](#53-训练数据构建)
  - [5.4 文本预处理](#54-文本预处理)
  - [5.5 正样本权重计算](#55-正样本权重计算)
  - [5.6 数据处理工具函数](#56-数据处理工具函数)
- [6. 训练策略](#6-训练策略)
  - [6.1 五+F阶段渐进式训练策略](#61-五f阶段渐进式训练策略)
  - [6.2 渐进式学习率调度策略](#62-渐进式学习率调度策略)
  - [6.3 优化器配置](#63-优化器配置)
  - [6.4 训练策略优势](#64-训练策略优势)
- [7. 实验设计](#7-实验设计)
  - [7.1 数据集配置](#71-数据集配置)
  - [7.2 实验环境](#72-实验环境)
  - [7.3 超参数配置](#73-超参数配置)
  - [7.4 实验流程](#74-实验流程)
  - [7.5 评估策略](#75-评估策略)
  - [7.6 实验记录(data/log.log)](#76-实验记录dataloglog)
- [8. 性能评估](#8-性能评估)
  - [8.1 评估指标](#81-评估指标)
  - [8.2 实验结果分析](#82-实验结果分析)
  - [8.3 模型应用和评估](#83-模型应用和评估)
- [9. 算法伪代码](#9-算法伪代码)
  - [9.1 五+F阶段训练算法](#91-五f阶段训练算法)
  - [9.2 前向传播算法](#92-前向传播算法)
  - [9.3 损失计算算法](#93-损失计算算法)
  - [9.4 评估算法](#94-评估算法)
- [10. 代码实现](#10-代码实现)
  - [10.1 核心模型实现](#101-核心模型实现)
  - [10.2 数据处理实现](#102-数据处理实现)
  - [10.3 数据处理完整实现](#103-数据处理完整实现)
- [11. 结论与展望](#11-结论与展望)
  - [11.1 模型优势](#111-模型优势)
  - [11.2 技术创新点](#112-技术创新点)
  - [11.3 性能期望](#113-性能期望)
  - [11.4 完整工作流程](#114-完整工作流程)
- [附录](#附录)
  - [附录A：数据文件说明](#附录a数据文件说明)
  - [附录B：模型文件说明](#附录b模型文件说明)
  - [附录C：可视化文件说明](#附录c可视化文件说明)

---

## 1. 概述

### 1.1 项目背景

TextAssessor是一个基于BERT预训练模型的文本评估模型，专门用于伙伴部队能力评估和上帝抵抗军威胁分析的自动化文本处理。该模型是对RAND公司研究报告[《Leveraging Machine Learning for Operation Assessment》](https://www.rand.org/pubs/research_reports/RR4196.html)在深度学习上的技术拓展和实现。

该模型能够同时处理两类评估任务：
- **PF (Partner Force) 评估**：伙伴部队能力评估，包括PF_score、PF_US、PF_neg三个维度
- **LRA (Lord's Resistance Army) 评估**：上帝抵抗军威胁评估，包括Threat_up、Threat_down、Citizen_impact三个维度

### 1.2 技术特点

- **双任务学习架构**：同时处理PF（伙伴部队）和LRA（上帝抵抗军）两类评估任务
- **注意力过滤机制**：通过可学习的权重过滤机制提升特定任务的特征表示
- **分阶段训练策略**：采用五+F阶段渐进式训练，逐步解冻更多的预训练层
- **不平衡数据处理**：使用正样本权重调整处理标签不平衡问题
- **完整工具链**：集成数据清洗、关键词标注、API调用、权重计算等核心工具函数
- **自动化标注**：基于DeepSeek API的智能文本标注

---

## 2. 理论基础

### 2.1 BERT预训练模型

BERT (Bidirectional Encoder Representations from Transformers) 是一个基于Transformer的双向编码器模型。其核心特点是：

**Self-Attention机制：**

``` math
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

其中：
- $Q$ ：查询矩阵 (Query)
- $K$ ：键矩阵 (Key)  
- $V$ ：值矩阵 (Value)
- $d_k$ ：键向量的维度

**多头注意力：**

``` math
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
```

其中：

``` math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

### 2.2 双任务学习理论

双任务学习通过共享底层表示学习，利用任务间的相关性提高整体性能。设两个任务的损失函数分别为 $L_1$ 和 $L_2$ ，总损失为：

``` math
L_{total} = \alpha L_1 + \beta L_2
```

其中 $\alpha$ 和 $\beta$ 是任务权重参数。

### 2.3 注意力过滤机制

本模型引入可学习的过滤器，对句子级表示进行加权平均：

**句子嵌入计算：**

``` math
\mathbf{s}_i = \frac{\sum_{j=1}^{T} m_{ij} \mathbf{h}_{ij}}{\sum_{j=1}^{T} m_{ij} + \epsilon}
```

其中：
- $\mathbf{h}_{ij}$ ：第 $i$ 个句子第 $j$ 个token的隐状态
- $m_{ij}$ ：注意力掩码
- $\epsilon$ ：防止除零的小常数

**任务特定过滤：**

``` math
p_{task}(\mathbf{s}_i) = \sigma(f_{filter}(\mathbf{s}_i))
```

**加权聚合：**

``` math
\mathbf{c}_{task} = \frac{\sum_{i=1}^{N} p_{task}(\mathbf{s}_i) \mathbf{s}_i}{\sum_{i=1}^{N} p_{task}(\mathbf{s}_i) + \epsilon}
```

---

## 3. 模型架构

### 3.1 整体架构

```
输入文本 → BERT编码器 → 句子嵌入 → [PF过滤器, LRA过滤器] → [PF评估器, LRA评估器] → 输出预测
```

### 3.2 组件详解

#### 3.2.1 文本编码器 (Text Encoder)
- **模型**：BERT-base-uncased
- **维度**：768
- **功能**：将输入文本转换为上下文相关的向量表示

#### 3.2.2 过滤器 (Filter)
- **PF过滤器**：用于识别与伙伴部队相关的句子
- **LRA过滤器**：用于识别与上帝抵抗军相关的句子

过滤器结构：
```
Linear(768 → 192) → LayerNorm → LeakyReLU → Dropout(0.7) → Linear(192 → 1) → Sigmoid
```

#### 3.2.3 评估器 (Assessor)
- **PF评估器**：输出3维预测 (PF_score, PF_US, PF_neg)
- **LRA评估器**：输出3维预测 (Threat_up, Threat_down, Citizen_impact)

评估器结构：
```
Linear(768 → 192) → ReLU → Dropout(0.7) → Linear(192 → 3)
```

### 3.3 数学表示

设输入文本序列为 $X = \{x_1, x_2, ..., x_T\}$ ，模型的前向传播过程为：

1. **BERT编码**：

``` math
\mathbf{H} = \text{BERT}(X)
```

2. **句子嵌入**：

``` math
\mathbf{s} = \frac{\sum_{t=1}^{T} m_t \mathbf{h}_t}{\sum_{t=1}^{T} m_t + \epsilon}
```

3. **任务过滤**：

``` math
p_{PF(LRA)} = \sigma(W_{PF(LRA)} \mathbf{s} + b_{PF(LRA)}) \\
```

4. **加权聚合**：

``` math
\mathbf{c}_{PF(LRA)} = \frac{\sum_{i} p_{PF(LRA)}^{(i)} \mathbf{s}^{(i)}}{\sum_{i} p_{PF(LRA)}^{(i)} + \epsilon}
```

5. **最终预测**：
``` math
\mathbf{y}_{PF(LRA)} = W_{assess\_PF(LRA)} \mathbf{c}_{PF(LRA)} + b_{assess\_PF(LRA)}
```

---

## 4. 算法原理

### 4.1 损失函数

采用加权二元交叉熵损失处理不平衡数据：

``` math
L_{BCE}(y, \hat{y}) = -\sum_{i=1}^{N} w_i [y_i \log(\sigma(\hat{y}_i)) + (1-y_i) \log(1-\sigma(\hat{y}_i))]
```

其中：
- $w_i = \sqrt{\frac{n_{neg}}{n_{pos}}}$ ：正样本权重
- $n_{pos}$ ：正样本数量
- $n_{neg}$ ：负样本数量

**总损失函数**：

``` math
L_{total} = L_{PF\_score} + L_{PF\_US} + L_{PF\_neg} + L_{Threat\_up} + L_{Threat\_down} + L_{Citizen\_impact}
```

### 4.2 优化策略

采用AdamW优化器，具有权重衰减正则化：

``` math
\theta_{t+1} = \theta_t - \eta_t [\frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t]
```

其中：
- $\eta_t$ ：学习率
- $m_t$ 、 $v_t$ ：动量估计
- $\lambda$ ：权重衰减系数

### 4.3 评估指标

**准确率计算**：

``` math
\text{Accuracy} = \frac{\sum_{i=1}^{6} \sum_{j=1}^{N} \mathbb{1}[\sigma(\hat{y}_{ij}) > 0.5] = y_{ij}]}{6N}
```

其中：
- $N$ ：样本数量
- $6$ ：预测维度数量
- $\mathbb{1}[\cdot]$ ：指示函数

---

## 5. 数据预处理

### 5.1 原始数据处理流程

#### 5.1.1 数据来源
原始数据文件为 `Lexis-Nexis_LRA.csv`，包含以下字段：
- **Title**：新闻标题
- **Source**：新闻来源
- **Time**：新闻时间
- **Year**：新闻年份
- **Month**：新闻月份
- **Day**：新闻日期
- **Text**：新闻正文

#### 5.1.2 数据清洗步骤
1. **添加ID列**：为每条记录分配唯一标识符
2. **文本预处理**：
   - 去除 `Text` 列两端空格
   - 删除 `Year`、`Month`、`Day` 列
   - 进行编码与格式检查
   - 去除重复文本
3. **数据验证**：使用 `is_legal_text()` 函数验证文本合法性

清洗后数据保存为 `cleaned.csv`，包含字段：
- **ID**：原数据ID
- **Title**：新闻标题
- **Source**：新闻来源
- **Time**：新闻时间
- **Text**：新闻正文（列表格式）

### 5.2 目标文本标注

#### 5.2.1 PF（伙伴部队）目标标注
筛选包含以下关键词的文本片段，这些地区是伙伴部队活动的主要区域：
- **"Uganda"**（乌干达）
- **"Sudan"**（苏丹） 
- **"Central African Republic", "CAR"**（中非共和国）
- **"Democratic Republic of the Congo", "DRC"**（刚果民主共和国）

使用 `abstract_text()` 函数进行关键词匹配和上下文提取。

#### 5.2.2 LRA（上帝抵抗军）目标标注
筛选包含以下关键词的文本片段：
- **"Lord's Resistance Army", "LRA"**（上帝抵抗军）

### 5.3 训练数据构建

#### 5.3.1 数据筛选
从同时包含PF和LRA目标的文本中选择300个样本：
```python
target_indices = train_df.PF_TARGET.apply(lambda _: any(_)) & train_df.LRA_TARGET.apply(lambda _: any(_))
target_df = train_df.loc[target_indices, ['ID', 'Text']].iloc[np.linspace(0, target_indices.sum(), num=300, endpoint=False, dtype=int)]
```

#### 5.3.2 人工标注流程
使用DeepSeek API进行六个维度的0-1标注：

**API配置**：
```python
from openai import OpenAI

API_KEY = 'sk-7e6e1a9c2ba84b18a09372aff22ff837'  # 已失效
API_MODEL = 'deepseek-chat'
API_URL = 'https://api.deepseek.com/v1'

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)
```

**标注模板**：
```
请根据以下六个基本标准，对上文进行0-1标注
{}
基本标准：
(1)"PF_score"，报告提到伙伴部队的成功
(2)"PF_US"，报告提到美国与伙伴部队合作
(3)"PF_neg"，报告美国伙伴部队的负面情况
(4)"Threat_up"，报告提到上帝抵抗军的威胁增加
(5)"Threat_down"，上帝抵抗军的威胁是否减少
(6)"Citizen_impact"，公民是否受到上帝抵抗军暴力的影响
```

**标注实现**：
```python
def ask_deepseek(ask):
    """调用DeepSeek API进行标注"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides accurate and concise answers to user queries."
        },
        {
            "role": "user",
            "content": ask
        }
    ]
    return client.chat.completions.create(
        model=API_MODEL,
        messages=messages,
        stream=False
    ).choices[0].message.content
```

#### 5.3.3 最终数据集
经过标注和整合，最终训练数据集包含以下字段：
- **Text**：文本列表 (list)
- **PF_TARGET**：伙伴部队相关句子标记 (list)
- **LRA_TARGET**：上帝抵抗军相关句子标记 (list)
- **PF_score, PF_US, PF_neg**：伙伴部队评估标签 (binary)
- **Threat_up, Threat_down, Citizen_impact**：上帝抵抗军威胁评估标签 (binary)

### 5.4 文本预处理

1. **分词和编码**：
   ```python
   tokens = tokenizer(text, 
                     return_tensors='pt',
                     padding='max_length',
                     truncation=True,
                     max_length=256)
   ```

2. **数据类型转换**：
   ```python
   # 字符串列表转换为Python列表
   df['Text'] = df['Text'].apply(ast.literal_eval)
   df['PF_TARGET'] = df['PF_TARGET'].apply(ast.literal_eval)
   df['LRA_TARGET'] = df['LRA_TARGET'].apply(ast.literal_eval)
   ```

### 5.5 正样本权重计算

```python
def get_pos_weights(df, targets):
    """
    计算各任务维度的正样本权重，用于处理类别不平衡问题
    
    Args:
        df: 包含标签的DataFrame
        targets: 目标标签列表
        
    Returns:
        dict: 每个目标的权重字典 {target: weight}
        
    计算公式：
        weight = negative_count / positive_count
        
    使用说明：
        - 权重用于加权二元交叉熵损失函数
        - 较高的权重给予少数类（正样本）更多关注
        - 在实际使用时会对权重取平方根以避免过度加权
    """
    f = lambda x: x[0] / x[1]  # negative_count / positive_count
    weights = {
        key: f(df[key].value_counts()[[0, 1]].to_numpy())
        for key in targets
    }
    return weights
```

### 5.6 数据处理工具函数

#### 5.6.1 文本合法性检查
```python
def is_legal_text(text):
    """
    检查文本是否为合法的ASCII字符串且可被ast.literal_eval解析
    
    Args:
        text (str): 待检查的文本字符串
    
    Returns:
        bool: 文本合法性，True表示合法，False表示不合法
    
    功能说明:
        1. 首先检查文本是否包含非ASCII字符，如包含则返回False
        2. 尝试使用ast.literal_eval解析文本，成功则返回True
        3. 解析失败则返回False
        
    实际实现:
        - 使用正则表达式检查非ASCII字符：r'[^\x00-\x7F]'
        - 使用ast.literal_eval解析文本
    """
    if re.search(r'[^\x00-\x7F]', text):
        return False
    try:
        ast.literal_eval(text)
        return True
    except:
        return False
```

#### 5.6.2 关键词匹配和上下文提取
```python
def abstract_text(text, target):
    """
    在文本列表中匹配目标关键词，并返回包含关键词及其上下文的布尔掩码
    
    Args:
        text (list): 文本句子列表
        target (tuple/list): 目标关键词列表
    
    Returns:
        list: 布尔值列表，指示每个句子是否匹配目标关键词或其上下文
    
    功能说明:
        1. 遍历文本列表，检查每个句子是否包含目标关键词
        2. 使用滑动窗口策略，将匹配范围扩展到相邻句子
        3. 通过zip操作合并当前句子、前一句子、后一句子的匹配状态
        4. 返回扩展后的匹配掩码，用于提取相关上下文
        
    实际实现:
        - 第一步：使用lambda和map的函数式编程风格进行关键词匹配
        - 第二步：通过zip操作实现上下文扩展的滑动窗口
    """
    matches = list(
        map(
            lambda s: any(map(lambda t: t in s, target)),
            text
        )
    )
    return list(
        map(
            lambda _: any(_),
            zip(
                matches, [False] + matches[:-1], matches[1:] + [False]
            )
        )
    )
```

#### 5.6.3 DeepSeek API标注接口
```python
def ask_deepseek(ask):
    """
    调用DeepSeek API进行文本标注
    
    Args:
        ask (str): 标注请求文本
        
    Returns:
        str: API返回的标注结果
        
    配置信息:
        - API_KEY: 'sk-7e6e1a9c2ba84b18a09372aff22ff837' (已失效)
        - API_MODEL: 'deepseek-chat'
        - API_URL: 'https://api.deepseek.com/v1'
        
    实际实现:
        - 使用OpenAI客户端兼容接口调用DeepSeek API
        - 系统提示设置为通用助手角色
        - 采用非流式模式获取完整响应
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides accurate and concise answers to user queries."
        },
        {
            "role": "user",
            "content": ask
        }
    ]
    return client.chat.completions.create(
        model=API_MODEL,
        messages=messages,
        stream=False
    ).choices[0].message.content
```

#### 5.6.4 正样本权重计算
```python
def get_pos_weights(df, targets):
    """
    计算各任务维度的正样本权重，用于处理类别不平衡问题
    
    Args:
        df (DataFrame): 包含标签的DataFrame
        targets (list): 目标标签列表
        
    Returns:
        dict: 每个目标的权重字典 {target: weight}
        
    权重计算公式:
        weight = negative_count / positive_count
        用于在损失函数中平衡正负样本
        
    实际实现:
        - 使用lambda函数简化计算：f = lambda x: x[0] / x[1]
        - 通过value_counts()[[0, 1]]获取负样本和正样本数量
        - 字典推导式批量计算所有目标的权重
    """
    f = lambda x: x[0] / x[1]
    weights = {
        key: f(df[key].value_counts()[[0, 1]].to_numpy())
        for key in targets
    }
    return weights
```

---

## 6. 训练策略

### 6.1 五+F阶段渐进式训练策略

基于实际实验设计，模型采用五+F阶段渐进式训练策略，逐步解冻BERT的不同层次，实现从任务特定组件到深层特征提取的渐进式微调。

#### 第一阶段 (70 epochs)
- **冻结层**：BERT编码器所有参数
- **训练层**：PF过滤器、LRA过滤器、PF评估器、LRA评估器
- **学习率**：1e-6 (所有训练层)
- **权重衰减**：1e-3
- **批次大小**：6
- **目的**：训练任务特定的上层组件，学习任务相关的表示

#### 第二阶段 (50 epochs)
- **解冻层**：BERT pooler层
- **学习率**：
  - 过滤器/评估器：1e-6
  - BERT pooler：8e-7
- **权重衰减**：1e-3
- **批次大小**：6
- **目的**：微调句子级表示，优化序列级特征提取

#### 第三阶段 (70 epochs)
- **解冻层**：BERT最后一层编码器 (layer[-1:])
- **学习率**：
  - 过滤器/评估器：1e-6
  - BERT pooler：8e-7
  - BERT最后一层：4e-7
- **权重衰减**：1e-3
- **批次大小**：6
- **目的**：精细调整高层特征提取

#### 第四阶段 (70 epochs)
- **解冻层**：BERT倒数第2-3层编码器 (layer[-3:-1])
- **学习率**：
  - 过滤器/评估器：1e-6
  - BERT pooler：1e-6
  - BERT最后一层：8e-7
  - BERT倒数2-3层：4e-7
- **权重衰减**：1e-3
- **批次大小**：6
- **目的**：扩展微调范围，优化中高层特征

#### 第五阶段 (90 epochs)
- **解冻层**：BERT倒数第4-7层编码器 (layer[-7:-3])
- **学习率**：
  - 过滤器/评估器：1e-6
  - BERT pooler：1e-6
  - BERT最后一层：1e-6
  - BERT倒数2-3层：8e-7
  - BERT倒数4-7层：4e-7
- **权重衰减**：1e-3
- **批次大小**：6
- **目的**：进一步扩展微调范围

#### 最终阶段 (140 epochs)
- **解冻层**：所有BERT参数（编码器、嵌入层）
- **学习率**：
  - 过滤器/评估器：1e-6
  - BERT pooler：1e-6
  - BERT最后一层：1e-6
  - BERT倒数2-3层：1e-6
  - BERT倒数4-7层：8e-7
  - BERT其余编码器层：4e-7
  - BERT嵌入层：2e-7
- **权重衰减**：1e-4 (降低正则化强度)
- **批次大小**：1 (精细化训练)
- **目的**：全模型微调，达到最优性能

### 6.2 渐进式学习率调度策略

采用层次化学习率策略，遵循以下原则：

1. **任务特定层**：保持较高学习率 (1e-6)，快速适应任务
2. **高层特征**：中等学习率 (8e-7 - 1e-6)，适度调整
3. **中层特征**：较低学习率 (4e-7 - 8e-7)，谨慎微调
4. **底层特征**：最低学习率 (2e-7 - 4e-7)，保持预训练知识

### 6.3 优化器配置

```python
# 示例：第五阶段优化器配置
optimizer = AdamW([
    {'params': model.PF_assessor.parameters(), 'lr': 1e-6},
    {'params': model.LRA_assessor.parameters(), 'lr': 1e-6},
    {'params': model.PF_filter.parameters(), 'lr': 1e-6},
    {'params': model.LRA_filter.parameters(), 'lr': 1e-6},
    {'params': model.text_encoder.pooler.parameters(), 'lr': 1e-6},
    {'params': model.text_encoder.encoder.layer[-1:].parameters(), 'lr': 1e-6},
    {'params': model.text_encoder.encoder.layer[-3:-1].parameters(), 'lr': 8e-7},
    {'params': model.text_encoder.encoder.layer[-7:-3].parameters(), 'lr': 4e-7}
], weight_decay=1e-3)
```

### 6.4 训练策略优势

1. **渐进式解冻**：避免预训练知识的灾难性遗忘
2. **层次化学习率**：不同层采用不同的学习速度
3. **动态批次调整**：最终阶段使用更小批次进行精细调整
4. **自适应正则化**：根据训练阶段调整权重衰减强度

---

## 7. 实验设计

### 7.1 数据集配置

#### 7.1.1 原始数据统计
- **原始文件**：Lexis-Nexis_LRA.csv
- **清洗后数据**：cleaned.csv
- **训练样本选择**：300个同时包含PF和LRA目标的文本
- **最终训练集**：300个标注完成的文档

#### 7.1.2 数据分割策略
- **训练集**：80% (240个样本)
- **测试集**：20% (60个样本)
- **分割方式**：随机分割，固定随机种子20040508
- **数据来源**：train_target_df.csv

#### 7.1.3 数据标注质量
- **标注方式**：基于DeepSeek API的自动标注
- **标注标准**：六个明确的二分类标准
- **质量控制**：通过正则表达式提取和验证标注结果
- **标注文档**：完整记录保存在target_df.md

### 7.2 实验环境

- **硬件**：GPU (CUDA)
- **深度学习框架**：PyTorch + Transformers
- **预训练模型**：BERT-base-uncased (本地路径: ./model/bert-base-uncased)
- **最大序列长度**：256 tokens
- **分词器**：BertTokenizer

### 7.3 超参数配置

#### 7.3.1 基础参数
| 参数 | 值 |
|------|-----|
| 随机种子 | 20040508 |
| Dropout率 | 0.7 |
| 优化器 | AdamW |
| 最大序列长度 | 256 |

#### 7.3.2 各阶段训练参数
| 阶段 | 训练轮数 | 批次大小 | 权重衰减 | 主要学习率范围 |
|------|----------|----------|----------|----------------|
| 第一阶段 | 70 | 6 | 1e-3 | 1e-6 |
| 第二阶段 | 50 | 6 | 1e-3 | 1e-6 ~ 8e-7 |
| 第三阶段 | 70 | 6 | 1e-3 | 1e-6 ~ 4e-7 |
| 第四阶段 | 70 | 6 | 1e-3 | 1e-6 ~ 4e-7 |
| 第五阶段 | 90 | 6 | 1e-3 | 1e-6 ~ 4e-7 |
| 最终阶段 | 140 | 1 | 1e-4 | 1e-6 ~ 2e-7 |

#### 7.3.3 正样本权重
通过 `get_pos_weights()` 函数计算各任务的类别权重：
- **PF_score**: 负样本数 / 正样本数
- **PF_US**: 负样本数 / 正样本数  
- **PF_neg**: 负样本数 / 正样本数
- **Threat_up**: 负样本数 / 正样本数
- **Threat_down**: 负样本数 / 正样本数
- **Citizen_impact**: 负样本数 / 正样本数

### 7.4 实验流程

#### 7.4.1 数据预处理完整流程
1. **原始数据加载**：从Lexis-Nexis_LRA.csv读取新闻数据
2. **数据清洗**：
   - 添加ID列进行唯一标识
   - 文本格式标准化和去重
   - 删除无效和重复记录
3. **关键词标注**：
   - PF目标：标注包含Uganda、Sudan、CAR、DRC的句子
   - LRA目标：标注包含Lord's Resistance Army的句子
4. **样本筛选**：选择同时包含PF和LRA目标的300个文本
5. **人工标注**：使用DeepSeek API进行六维度标注
6. **数据整合**：合并原始文本和标注结果生成train_target_df.csv
7. **训练测试分割**：使用sklearn.train_test_split进行数据分割

#### 7.4.2 模型训练流程
```
1. 初始化模型 → 2. 阶段一训练 → 3. 阶段二训练 → 4. 阶段三训练 
                     ↓
6. 最终阶段训练 ← 5. 阶段五训练 ← 4. 阶段四训练
                     ↓
7. 保存最终模型和训练曲线
```

#### 7.4.3 模型检查点
- **阶段性保存**：每个主要阶段结束后保存模型
- **关键检查点**：
  - `text_assessor_A(260).pth`: 260 epochs后
  - `text_assessor_B(350).pth`: 350 epochs后  
  - `text_assessor_C(490).pth`: 490 epochs后
- **损失(&准确率)图表**：
  - `loss-plot_A(260).svg`
  - `loss-plot_B(350).svg`
  - `loss-plot_C(490).svg`

### 7.5 评估策略

#### 7.5.1 训练时评估
1. **每个epoch评估**：在训练和验证集上计算损失和准确率
2. **实时监控**：使用tqdm显示训练进度
3. **定期可视化**：每10个epoch生成损失和准确率曲线

#### 7.5.2 评估指标
- **损失函数**：加权二元交叉熵损失
- **准确率**：二分类准确率的平均值
- **收敛性**：训练和验证损失的变化趋势

#### 7.5.3 过拟合监控
通过比较训练损失和验证损失的差异来监控过拟合：
- **正常学习**：训练损失和验证损失同步下降
- **轻微过拟合**：验证损失停止下降或轻微上升
- **严重过拟合**：验证损失显著上升

### 7.6 实验记录(data/log.log)

#### 7.6.1 数据统计

训练开始前记录样本数据的统计信息：
```
Sample data - input_ids shape: torch.Size([7, 256])
Sample data - attention_mask shape: torch.Size([7, 256])
Sample data - PF_targets: tensor([0., 0., 1., 1., 1., 0., 0.], device='cuda:0')
Sample data - ...
```

#### 7.6.2 训练日志
每个阶段开始时输出优化器配置和训练参数：
```
=======Training Phase X=======
optimizer: {详细的优化器状态}
```

#### 7.6.3 性能追踪
- **训练损失和准确率**：每个epoch记录
```
- Average training loss: {avg_train_loss:.4f}\t\tAccuracy: {avg_accuracy:.2f}%
```
- **验证损失和准确率**：每个epoch记录
```
+ Average evaluation loss: {avg_loss:.4f}\t\tAccuracy: {avg_accuracy:.2f}%
```
- **可视化输出**：阶段性生成训练曲线图
``` python
# per 10 epochs
plt.savefig('./data/loss-plot.svg')
```
---

## 8. 性能评估

### 8.1 评估指标

#### 8.1.1 二分类准确率
对每个任务维度计算：

``` math
\text{Accuracy}_i = \frac{\text{TP}_i + \text{TN}_i}{\text{TP}_i + \text{TN}_i + \text{FP}_i + \text{FN}_i}
```

#### 8.1.2 整体准确率

``` math
\text{Overall Accuracy} = \frac{1}{6} \sum_{i=1}^{6} \text{Accuracy}_i
```

#### 8.1.3 损失监控
监控训练和验证损失的收敛情况，防止过拟合。

### 8.2 实验结果分析

#### 8.2.1 训练过程监控
通过三个检查点的损失曲线图进行分析：

1. **260 epochs后 (`loss-plot_A(260).svg`)**：
   - 观察初期训练效果
   - 验证渐进式解冻策略的有效性
   - 分析前四个训练阶段的收敛情况

2. **350 epochs后 (`loss-plot_B(350).svg`)**：
   - 中期训练效果评估
   - 检查是否出现过拟合迹象
   - 验证第五阶段训练的必要性

3. **490 epochs后 (`loss-plot_C(490).svg`)**：
   - 最终训练效果
   - 全模型微调的收敛性分析
   - 训练和验证性能的最终对比

#### 8.2.2 性能指标分析
1. **收敛性**：
   - 训练损失和验证损失是否稳定下降
   - 各阶段的收敛速度对比
   - 学习率调整对收敛的影响

2. **过拟合检测**：
   - 训练损失与验证损失的差距变化
   - 批次大小调整（6→1）对过拟合的影响
   - 权重衰减调整（1e-3→1e-4）的效果

3. **学习效果**：
   - 准确率在各阶段的提升趋势
   - 六个任务维度的平衡学习情况
   - 最终达到的性能水平

#### 8.2.3 阶段性评估
- **第一阶段**：任务特定组件的学习效果
- **第二阶段**：句子级表示的优化程度
- **第三-五阶段**：渐进式特征提取的改进
- **最终阶段**：全模型微调的性能提升

### 8.3 模型应用和评估

#### 8.3.1 模型推理流程
模型推理包含以下步骤：

1. **模型加载**：
   ```python
   model = TextAssessor.loads(save_model='./data/text_assessor_B(350).pth', device=device)
   ```

2. **分词器配置**：
   ```python
   tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
   ```

3. **批量预测**：
   ```python
   for _id, *_, _text in tqdm(data_df.itertuples(index=False), total=len(data_df)):
       pred = predict_one(data_text=_text, model=model, tokenizer=tokenizer, 
                         max_len=MAX_LEN, device=device) if _text else default_pred
   ```

#### 8.3.2 预测结果格式
模型输出包含六个维度的预测值：
```python
{
    "pred.values.PF_score": float,
    "pred.values.PF_US": float,
    "pred.values.PF_neg": float,
    "pred.values.Threat_up": float,
    "pred.values.Threat_down": float,
    "pred.values.Citizen_impact": float,
}
```

#### 8.3.3 大规模数据处理
- **数据来源**：cleaned.csv（完整清洗后数据）
- **处理策略**：逐条处理，避免内存溢出
- **异常处理**：对空文本列表进行特殊处理
- **结果保存**：预测结果合并到result_df.csv

#### 8.3.4 模型检查点选择
实验使用了text_assessor_B(350).pth作为推理模型，该检查点代表：
- **训练进度**：350个epochs后的模型状态
- **性能平衡**：在训练充分性和过拟合之间的平衡点
- **实用性**：适合大规模推理应用的稳定模型

---

## 9. 算法伪代码

### 9.1 五+F阶段训练算法

```
算法1: TextAssessor五+F阶段训练算法

输入: 
    训练数据 D_train, 验证数据 D_val
    预训练模型 BERT, 超参数配置 θ
    
输出: 
    训练好的模型 M, 损失记录 losses, 准确率记录 accs

 1: 初始化模型 M ← TextAssessor()
 2: 加载预训练权重 M.text_encoder ← BERT
 3: 设置随机种子 SEED ← 20040508
 4: 
 5: // 第一阶段训练 (70 epochs)
 6: for param in M.text_encoder.parameters():
 7:     param.requires_grad ← False
 8: 
 9: optimizer1 ← AdamW([
10:    {M.PF_assessor.parameters(), lr=1e-6},
11:    {M.LRA_assessor.parameters(), lr=1e-6},
12:    {M.PF_filter.parameters(), lr=1e-6},
13:    {M.LRA_filter.parameters(), lr=1e-6}
14: ], weight_decay=1e-3)
15: 
16: for epoch in range(70):
17:     M, losses, accs ← train_epoch(M, D_train, D_val, optimizer1)
18: M.save()
19:
20: // 第二阶段训练 (50 epochs)
21: for param in M.text_encoder.pooler.parameters():
22:     param.requires_grad ← True
23: 
24: optimizer2 ← AdamW([
25:     {M.[assessors+filters].parameters(), lr=1e-6},
26:     {M.text_encoder.pooler.parameters(), lr=8e-7}
27: ], weight_decay=1e-3)
28: 
29: for epoch in range(50):
30:     M, losses, accs ← train_epoch(M, D_train, D_val, optimizer2)
31: M.save()
32:
33: // 第三阶段训练 (70 epochs)
34: for param in M.text_encoder.encoder.layer[-1:].parameters():
35:     param.requires_grad ← True
36: 
37: optimizer3 ← AdamW([
38:     {M.[assessors+filters].parameters(), lr=1e-6},
39:     {M.text_encoder.pooler.parameters(), lr=8e-7},
40:     {M.text_encoder.encoder.layer[-1:].parameters(), lr=4e-7}
41: ], weight_decay=1e-3)
42: 
43: for epoch in range(70):
44:     M, losses, accs ← train_epoch(M, D_train, D_val, optimizer3)
45: M.save()
46:
47: // 第四阶段训练 (70 epochs)
48: for param in M.text_encoder.encoder.layer[-3:-1].parameters():
49:     param.requires_grad ← True
50: 
51: optimizer4 ← AdamW([
52:     {M.[assessors+filters].parameters(), lr=1e-6},
53:     {M.text_encoder.pooler.parameters(), lr=1e-6},
54:     {M.text_encoder.encoder.layer[-1:].parameters(), lr=8e-7},
55:     {M.text_encoder.encoder.layer[-3:-1].parameters(), lr=4e-7}
56: ], weight_decay=1e-3)
57: 
58: for epoch in range(70):
59:     M, losses, accs ← train_epoch(M, D_train, D_val, optimizer4)
60: M.save('./data/text_assessor_A(260).pth')
61: generate_plot('./data/loss-plot_A(260).svg')
62:
63: // 第五阶段训练 (90 epochs)
64: for param in M.text_encoder.encoder.layer[-7:-3].parameters():
65:     param.requires_grad ← True
66: 
67: optimizer5 ← AdamW([
68:     {M.[assessors+filters+pooler+layer[-1:]].parameters(), lr=1e-6},
69:     {M.text_encoder.encoder.layer[-3:-1].parameters(), lr=8e-7},
70:     {M.text_encoder.encoder.layer[-7:-3].parameters(), lr=4e-7}
71: ], weight_decay=1e-3)
72: 
73: for epoch in range(90):
74:     M, losses, accs ← train_epoch(M, D_train, D_val, optimizer5)
75: M.save('./data/text_assessor_B(350).pth')
76: generate_plot('./data/loss-plot_B(350).svg')
77:
78: // 最终阶段训练 (140 epochs)
79: for param in M.text_encoder.parameters():
80:     param.requires_grad ← True
81: 
82: batch_size ← 1  // 精细化训练
83: D_train, D_val ← rebuild_dataloader(batch_size=1)
84: 
85: optimizer_final ← AdamW([
86:     {M.[top_layers].parameters(), lr=1e-6},
87:     {M.text_encoder.encoder.layer[-7:-3].parameters(), lr=8e-7},
88:     {M.text_encoder.encoder.layer[:-7].parameters(), lr=4e-7},
89:     {M.text_encoder.embeddings.parameters(), lr=2e-7}
90: ], weight_decay=1e-4)
91: 
92: for epoch in range(140):
93:     M, losses, accs ← train_epoch(M, D_train, D_val, optimizer_final)
94: M.save('./data/text_assessor_C(490).pth')
95: generate_plot('./data/loss-plot_C(490).svg')
96: 
97: return M, losses, accs
```

### 9.2 前向传播算法

```
算法2: TextAssessor前向传播

输入: 文本序列列表 X = [x1, x2, ..., xN]
输出: 预测结果 logits = [logit1, logit2, ..., logitN]

 1: logits ← []
 2: for xi in X:  // xi: [n, len]
 3:     // 获取句子嵌入
 4:     // input_ids, attention_mask ← tokenize(xi)
 5:     H ← BERT_encoder(input_ids, attention_mask)
 6:     
 7:     // 加权平均池化
 8:     mask ← attention_mask.unsqueeze(-1).expand(H.size())
 9:     S ← sum(H * mask, dim=1) / (sum(mask, dim=1) + ε)    // [n, hidden_size]
10: 
11:     // 任务特定过滤
12:     P_PF ← PF_filter(S)     // [n]
13:     P_LRA ← LRA_filter(S)   // [n]
14: 
15: // 加权聚合
16:     C_PF ← sum(S * P_PF.unsqueeze(-1), dim=0) / (sum(P_PF) + ε)     // [hidden_size]
17:     C_LRA ← sum(S * P_LRA.unsqueeze(-1), dim=0) / (sum(P_LRA) + ε)  // [hidden_size]
18: 
19:     // 最终预测
20:     y_PF ← PF_assessor(C_PF)     // [3]
21:     y_LRA ← LRA_assessor(C_LRA)  // [3]
22:     
23:     logits.append(concat([y_PF, y_LRA]))  // [6]
24: 
25: return stack(logits, dim=0) // [N, 6]
```

### 9.3 损失计算算法

```
算法3: 加权二元交叉熵损失计算

输入: 
    预测值 y_pred = [y1, y2, ..., y6]
    真实标签 y_true = [t1, t2, ..., t6]  
    正样本权重 pos_weights = [w1, w2, ..., w6]
    
输出: 总损失 L_total

 1: L_total ← 0
 2: targets ← ['PF_score', 'PF_US', 'PF_neg', 
              'Threat_up', 'Threat_down', 'Citizen_impact']
 3: 
 4: for i in range(6):
 5:     wi ← sqrt(pos_weights[targets[i]])
 6:     Li ← binary_cross_entropy_with_logits(y_pred[i], y_true[i], 
                                           pos_weight=wi)
 7:     L_total ← L_total + Li
 8: 
 9: return L_total
```

### 9.4 评估算法

```
算法4: 模型评估

输入: 
    模型 M, 数据加载器 dataloader
    正样本权重 pos_weights
    
输出: 平均损失 avg_loss, 平均准确率 avg_accuracy

 1: M.eval()
 2: total_loss ← 0
 3: total_correct ← 0
 4: 
 5: for batch in dataloader:
 6:     outputs ← M(batch)
 7:     loss ← compute_loss(outputs, batch, pos_weights)
 8:     total_loss ← total_loss + loss.item()
 9:     
10:    // 计算准确率
11:    for output, target in zip(outputs, batch):
12:        predictions ← (sigmoid(output) > 0.5).float()
13:        correct ← count_correct_predictions(predictions, target)
14:        total_correct ← total_correct + correct
15: 
16: avg_loss ← total_loss / len(dataloader)
17: avg_accuracy ← total_correct / len(dataloader)
18: 
19: return avg_loss, avg_accuracy
```

---

## 10. 代码实现

### 10.1 核心模型实现

```python
class TextAssessor(nn.Module):
    def __init__(self, dropout: float = 0.7):
        super().__init__()
        self.dropout = dropout
        
        # BERT编码器
        self.text_encoder = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        encode_dim = self.text_encoder.config.hidden_size
        
        # 任务特定过滤器
        self.PF_filter = self._build_filter(encode_dim)
        self.LRA_filter = self._build_filter(encode_dim)
        
        # 任务特定评估器
        self.PF_assessor = self._build_assessor(encode_dim)
        self.LRA_assessor = self._build_assessor(encode_dim)
```

### 10.2 数据处理实现

```python
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, device='cpu'):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.to_literal_eval()
    
    def __getitem__(self, idx):
        cur = self.df.iloc[idx]
        encodings = self.tokenizer(
            text=cur.Text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        
        return {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device),
            # ... 其他标签字段
        }
```

### 10.3 数据处理完整实现

#### 10.3.1 数据清洗模块
```python
# 数据清洗和预处理
def clean_data(raw_df):
    cleaned_df = raw_df.copy()
    # 添加ID列
    cleaned_df.insert(0, 'ID', range(len(cleaned_df)))
    # 文本预处理
    cleaned_df['Text'] = cleaned_df['Text'].str.strip()
    cleaned_df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
    # 格式验证和去重
    cleaned_df = cleaned_df.loc[cleaned_df.Text.apply(is_legal_text)]
    cleaned_df.drop_duplicates('Text', inplace=True)
    return cleaned_df
```

#### 10.3.2 目标标注模块
```python
import re
import ast

def is_legal_text(text):
    """文本合法性检查"""
    if re.search(r'[^\x00-\x7F]',  text):
        return False
    try:
        ast.literal_eval(text)
        return True
    except:
        return False

def abstract_text(text, target):
    """关键词匹配和上下文提取"""
    matches = list(
        map(
            lambda s: any(map(lambda t: t in s, target)),
            text
        )
    )
    return list(
        map(
            lambda _: any(_),
            zip(
                matches, [False] + matches[:-1], matches[1:] + [False]
            )
        )
    )

# PF目标标注
def annotate_pf_targets(df):
    """标注PF相关的句子"""
    PF_targets = [
        ("Uganda",),
        ("Sudan",),
        ("Central African Republic", "CAR"),
        ("Democratic Republic of the Congo", "DRC",),
    ]
    
    PF_matched = df.Text.apply(lambda s: abstract_text(s, PF_targets[0]))
    for PF_target in PF_targets[1:]:
        PF_matched = map(lambda _: list(map(any, zip(_[0], _[1]))), 
                        zip(PF_matched, df.Text.apply(lambda s: abstract_text(s, PF_target))))
    df['PF_TARGET'] = list(PF_matched)
    return df

# LRA目标标注
def annotate_lra_targets(df):
    """标注LRA相关的句子"""
    LRA_targets = [("Lord's Resistance Army", "LRA",)]
    LRA_matched = df.Text.apply(lambda s: abstract_text(s, LRA_targets[0]))
    df['LRA_TARGET'] = list(LRA_matched)
    return df

# DeepSeek API标注
from openai import OpenAI

API_KEY = 'sk-7e6e1a9c2ba84b18a09372aff22ff837'  # 已失效
API_MODEL = 'deepseek-chat'
API_URL = 'https://api.deepseek.com/v1'

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)

def ask_deepseek(ask):
    """调用DeepSeek API进行文本标注"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides accurate and concise answers to user queries."
        },
        {
            "role": "user",
            "content": ask
        }
    ]
    return client.chat.completions.create(
        model=API_MODEL,
        messages=messages,
        stream=False
    ).choices[0].message.content

# 正样本权重计算
def get_pos_weights(df, targets):
    """计算正样本权重用于处理类别不平衡"""
    f = lambda x: x[0] / x[1]  # negative_count / positive_count
    weights = {
        key: f(df[key].value_counts()[[0, 1]].to_numpy())
        for key in targets
    }
    return weights
```

#### 10.3.3 模型推理实现
```python
def predict_one(data_text, model, tokenizer, max_len, device):
    model.eval()
    
    inputs = tokenizer(
        data_text, return_tensors='pt', padding='max_length', 
        truncation=True, max_length=max_len
    )

    with torch.no_grad():
        outputs = model._forward(
            input_ids=inputs['input_ids'].to(device), 
            attention_mask=inputs['attention_mask'].to(device)
        )
    
    assessments = outputs.sigmoid().cpu().numpy()

    return {
        "pred.values.PF_score": assessments[0],
        "pred.values.PF_US": assessments[1],
        "pred.values.PF_neg": assessments[2],
        "pred.values.Threat_up": assessments[3],
        "pred.values.Threat_down": assessments[4],
        "pred.values.Citizen_impact": assessments[5],
    }
```

#### 10.3.4 批量处理实现
```python
def batch_predict(data_df, model, tokenizer, max_len, device):
    items = {}
    for _id, *_, _text in tqdm(data_df.itertuples(index=False), total=len(data_df)):
        if _text:  # 检查文本是否为空
            pred = predict_one(data_text=_text, model=model, 
                             tokenizer=tokenizer, max_len=max_len, device=device)
        else:
            # 空文本的默认预测
            pred = {
                "pred.values.PF_score": None,
                "pred.values.PF_US": None,
                "pred.values.PF_neg": None,
                "pred.values.Threat_up": None,
                "pred.values.Threat_down": None,
                "pred.values.Citizen_impact": None,
            }
        items[_id] = pred
    return pd.DataFrame.from_dict(items, orient='index')
```

---

## 11. 结论与展望

### 11.1 模型优势

1. **双任务协同学习**：通过共享底层BERT编码器，PF和LRA任务能够相互促进
2. **注意力过滤机制**：有效识别任务相关的关键句子，提高模型解释性
3. **分阶段训练策略**：渐进式微调避免了梯度消失和过拟合问题
4. **不平衡数据处理**：加权损失函数有效处理标签不平衡问题

### 11.2 技术创新点

1. **句子级注意力过滤**：引入可学习的过滤器对句子进行任务特定的重要性评估
2. **五+F阶段渐进式微调策略**：从任务特定组件到全模型的渐进式训练
3. **层次化学习率调度**：不同层采用不同学习率，保持预训练知识的同时优化任务性能
4. **动态批次调整**：最终阶段采用更小批次进行精细化训练
5. **多任务损失平衡**：等权重组合多个二分类损失，处理类别不平衡问题

### 11.3 性能期望

基于当前五+F阶段训练架构和490个epochs的训练策略，预期模型能够达到：
- **整体准确率**：> 80%
- **收敛稳定性**：各阶段均能稳定收敛，总训练时间约490个epochs
- **泛化能力**：通过渐进式微调和动态批次调整，在未见过的文本上保持较好性能
- **训练效率**：通过分阶段训练和层次化学习率，在保证性能的同时提高训练效率

### 11.4 完整工作流程

#### 11.4.1 数据处理流程
1. **原始数据清洗**：从Lexis-Nexis_LRA.csv到cleaned.csv
2. **目标句子标注**：PF和LRA关键词匹配
3. **样本筛选**：选择包含两类目标的300个文本
4. **人工标注**：使用DeepSeek API进行六维标注
5. **训练数据生成**：合并生成train_target_df.csv

#### 11.4.2 模型训练流程  
1. **五+F阶段渐进式训练**：从任务层到全模型的逐步解冻
2. **检查点管理**：关键阶段保存模型状态
3. **性能监控**：实时追踪损失和准确率变化
4. **可视化输出**：生成训练过程的损失曲线图

#### 11.4.3 模型应用流程
1. **模型选择**：使用350 epochs的检查点进行推理
2. **大规模预测**：对全部清洗后数据进行预测
3. **结果整合**：合并预测结果和原始数据
4. **结果保存**：输出完整的result_df.csv

#### 11.4.4 技术亮点总结
- **端到端流程**：从原始数据到最终应用的完整链路
- **自动化标注**：使用DeepSeek API减少人工标注成本
- **渐进式训练**：科学的训练策略确保模型收敛
- **生产就绪**：完整的推理和批量处理能力

## 附录

### 附录A：数据文件说明

| 文件名 | 描述 | 用途 |
|--------|------|------|
| Lexis-Nexis_LRA.csv | 原始新闻数据 | 数据来源 |
| cleaned.csv | 清洗后数据 | 预处理结果 |
| target_df.csv | 标注数据 | 人工标注结果 |
| target_df.md | 标注文档 | 标注过程记录 |
| train_target_df.csv | 训练数据 | 模型训练输入 |
| result_df.csv | 预测结果 | 模型应用输出 |

### 附录B：模型文件说明

| 文件名 | 描述 | 训练轮数 |
|--------|------|----------|
| text_assessor.pth | 最终模型 | 490 epochs |
| text_assessor_A(260).pth | 早期检查点 | 260 epochs |
| text_assessor_B(350).pth | 中期检查点 | 350 epochs |
| text_assessor_C(490).pth | 最终检查点 | 490 epochs |

### 附录C：可视化文件说明

| 文件名 | 描述 | 内容 |
|--------|------|------|
| loss-plot_A(260).svg | 早期训练曲线 | 前260轮损失和准确率 |
| loss-plot_B(350).svg | 中期训练曲线 | 前350轮损失和准确率 |
| loss-plot_C(490).svg | 完整训练曲线 | 全部490轮损失和准确率 |

---

**文档版本**: 1.0


**最后更新**: 2025年7月8日  

**作者**: H.S

**项目代码**: [Natural-Language-by-H.S](https://github.com/Huaseon/Natural-Language-by-H.S)

**联系方式**: dsj34473@163.com

**致谢**: 感谢DeepSeek团队提供的API支持，以及Hugging Face团队提供的预训练模型和工具库。
