# %%
import json
import math
import os
import random
from typing import Dict, List, Tuple


# %%
import matplotlib
matplotlib.rcParams.update({
    'font.size': 10.5,
    'lines.linewidth': .7,
})
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

# %%
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

# %%
from tqdm.auto import tqdm

# %%
from transformers import BertTokenizer

# %%
from model.text_assessment import PRETRAINED_MODEL_NAME, MAX_LEN
from model.text_assessment.text_dataset import TextDataset, collate_fn
from model.text_assessment.text_assessor import TextAssessor, train_epoch
from tools import get_pos_weights

# %%
SEED = 20040508
FILENAME = 'analysis_result_simple_format_80000_with_summary.xlsx'
LABELS = ['THREAT_up', 'THREAT_down', 'citizen_impact', 'PF_score', 'PF_US']
BATCH_SIZE = 6
PATIENCE = 30 # 早停
OUTPUT_DIR = './output/experiment-v2'
WEIGHT_DECAY_PHASE = 1e-3
WEIGHT_DECAY_FINAL = 1e-4

# %% 渐进微调配置 迭代次数和学习率
PHASES = [
    {  # Final: 所有层
        'name': 'final', 'epochs': 210,
        'lrs': {
            'outputs': 1e-6,
            'pooler': 1e-6,
            'enc_last1': 1e-6,
            'enc_last3to1': 1e-6,
            'enc_last7to3': 1e-6,
            'enc_rest': 8e-7,
            'embeddings': 4e-7,
        },
        'weight_decay': WEIGHT_DECAY_FINAL,
    },
]

# %% 设置随机种子 确保可重复性
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.Generator().manual_seed(seed)

# %% 划分数据集 80000 -> 8000/5000+/66000+
def split_train_val_test(df: pd.DataFrame, tokenizer: BertTokenizer) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = TextDataset(df=df, tokenizer=tokenizer, max_len=MAX_LEN, device=get_device())
    return random_split(dataset, [3/30, 2/30, 25/30])


# %%
def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 按配置阶段解冻参数
def enable_params(m: TextAssessor, phase_name: str):
    # 冻结所有参数
    for p in m.text_encoder.parameters():
        p.requires_grad = False
    for p in m.outputs.parameters():
        p.requires_grad = False

    if phase_name in ['phase1']: # 阶段一
        for p in m.outputs.parameters(): # 解冻输出层
            p.requires_grad = True
    elif phase_name in ['phase2']: # 阶段二
        for p in m.outputs.parameters(): # 解冻输出层
            p.requires_grad = True
        for p in m.text_encoder.pooler.parameters(): # 解冻[CLS]池化层
            p.requires_grad = True
    elif phase_name in ['phase3']: # 阶段三
        for p in m.outputs.parameters(): # 解冻输出层
            p.requires_grad = True
        for p in m.text_encoder.pooler.parameters(): # 解冻[CLS]池化层
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-1:].parameters(): # 解冻最后一层Transformer
            p.requires_grad = True
    elif phase_name in ['phase4']: # 阶段四
        for p in m.outputs.parameters(): # 解冻输出层
            p.requires_grad = True
        for p in m.text_encoder.pooler.parameters(): # 解冻[CLS]池化层
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-1:].parameters(): # 解冻最后一层Transformer
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-3:-1].parameters(): # 解冻倒数第3到倒数第2层Transformer
            p.requires_grad = True
    elif phase_name in ['phase5']: # 阶段五
        for p in m.outputs.parameters(): # 解冻输出层
            p.requires_grad = True
        for p in m.text_encoder.pooler.parameters(): # 解冻[CLS]池化层
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-1:].parameters(): # 解冻最后一层Transformer
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-3:-1].parameters(): # 解冻倒数第3到倒数第2层Transformer
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-7:-3].parameters(): # 解冻倒数第7到倒数第4层Transformer
            p.requires_grad = True
    elif phase_name in ['phase6']: # 阶段六
        for p in m.outputs.parameters(): # 解冻输出层
            p.requires_grad = True
        for p in m.text_encoder.pooler.parameters(): # 解冻[CLS]池化层
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-1:].parameters(): # 解冻最后一层Transformer
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-3:-1].parameters(): # 解冻倒数第3到倒数第2层Transformer
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[-7:-3].parameters(): # 解冻倒数第7到倒数第4层Transformer
            p.requires_grad = True
        for p in m.text_encoder.encoder.layer[:-7].parameters(): # 解冻剩余的Transformer层
            p.requires_grad = True
    elif phase_name in ['final']: # 最终阶段
        for p in m.outputs.parameters(): # 解冻输出层
            p.requires_grad = True
        for p in m.text_encoder.parameters(): # 解冻整个文本编码器（包括嵌入层和所有Transformer层）
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown phase: {phase_name}")

# %% 按配置的学习率和权重衰减构建AdamW优化器 
def build_optimizer(m: TextAssessor, lrs: Dict[str, float], weight_decay: float) -> AdamW:
    param_groups = []
    if 'outputs' in lrs:
        param_groups.append({'params': m.outputs.parameters(), 'lr': lrs['outputs']})
    if 'pooler' in lrs:
        param_groups.append({'params': m.text_encoder.pooler.parameters(), 'lr': lrs['pooler']})
    if 'enc_last1' in lrs:
        param_groups.append({'params': m.text_encoder.encoder.layer[-1:].parameters(), 'lr': lrs['enc_last1']})
    if 'enc_last3to1' in lrs:
        param_groups.append({'params': m.text_encoder.encoder.layer[-3:-1].parameters(), 'lr': lrs['enc_last3to1']})
    if 'enc_last7to3' in lrs:
        param_groups.append({'params': m.text_encoder.encoder.layer[-7:-3].parameters(), 'lr': lrs['enc_last7to3']})
    if 'enc_rest' in lrs:
        param_groups.append({'params': m.text_encoder.encoder.layer[:-7].parameters(), 'lr': lrs['enc_rest']})
    if 'embeddings' in lrs:
        param_groups.append({'params': m.text_encoder.embeddings.parameters(), 'lr': lrs['embeddings']})
    return AdamW(param_groups, weight_decay=weight_decay)

# %% 在数据集上收集模型的probs和真实标签
def collect_probs_targets(model: TextAssessor, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Eval collect', leave=False):
            logits = model(batch)  # [batch, 5]
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            # targets
            for sample in batch:
                all_targets.append([float(sample[l]) for l in LABELS])
    return np.vstack(all_probs), np.asarray(all_targets)

# %% 查找各标签上的概率阈值
def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    thresholds = []
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        pj = y_prob[:, j]
        # 单标签使用默认阈值0.5
        if len(np.unique(yj)) < 2:
            thresholds.append(0.5)
            continue
        # 精确率，召回率，概率阈值
        precision, recall, th = precision_recall_curve(yj, pj)
        # 计算各概率阈值上的F1分数
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1s = []
        for k, t in enumerate(th):
            p = precision[k]
            r = recall[k]
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            f1s.append(f1)
        if len(f1s) == 0:
            thresholds.append(0.5)
        else:
            best_idx = int(np.argmax(f1s))
            thresholds.append(float(th[best_idx]))
    return thresholds

# %% 计算各标签的AUC, AUPRC, F1分数
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # 各标签的AUC, AUPRC, F1
    aucs = []
    aprs = []
    f1s = []
    for j, label in enumerate(LABELS):
        yj = y_true[:, j]
        pj = y_prob[:, j]
        # AUC / AUPRC
        try:
            auc = roc_auc_score(yj, pj) if len(np.unique(yj)) > 1 else math.nan
        except Exception:
            auc = math.nan
        try:
            apr = average_precision_score(yj, pj) if len(np.unique(yj)) > 1 else math.nan
        except Exception:
            apr = math.nan
        # 概率阈值上的F1分数
        y_pred = (pj >= thresholds[j]).astype(np.float32)
        try:
            f1 = f1_score(yj, y_pred)
        except Exception:
            f1 = math.nan
        metrics[f'AUC_{label}'] = auc
        metrics[f'AUPRC_{label}'] = apr
        metrics[f'F1_{label}'] = f1
        if not math.isnan(auc):
            aucs.append(auc)
        if not math.isnan(apr):
            aprs.append(apr)
        if not math.isnan(f1):
            f1s.append(f1)
    # 宏平均计算 AUC, AUPRC, F1
    metrics['AUC_macro'] = float(np.mean(aucs)) if len(aucs) else math.nan
    metrics['AUPRC_macro'] = float(np.mean(aprs)) if len(aprs) else math.nan
    metrics['F1_macro'] = float(np.mean(f1s)) if len(f1s) else math.nan
    # 微平均计算 F1
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = (y_prob >= np.array(thresholds)[None, :]).astype(np.float32).reshape(-1)
    try:
        metrics['F1_micro'] = f1_score(y_true_flat, y_pred_flat)
    except Exception:
        metrics['F1_micro'] = math.nan
    return metrics

# %% 保存指标到CSV文件
def save_metrics_csv(path: str, metrics: Dict[str, float]):
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)

# %%
def main():
    # 设置随机种子
    set_seed(SEED)

    # 设备
    device = get_device()
    print(f"Using device: {device}\n")

    # 数据文件
    print(f"source: {FILENAME}")
    df = pd.read_excel('./data/' + FILENAME)
    print(f"Data shape: {df.shape}\n{df.head()}\n")

    # BERT分词器
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    print(f"BertTokenizer: {tokenizer}\n")

    # 3/2/20 split
    dataset_train, dataset_val, dataset_test = split_train_val_test(df, tokenizer)
    print(f"train={len(dataset_train)}\tval={len(dataset_val)}\ttest={len(dataset_test)}\n")

    # 训练集、验证集和测试集
    dl_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    dl_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Train-DataLoader.size: {len(dl_train)}\nVal-DataLoader.size: {len(dl_val)}\nTest-DataLoader.size: {len(dl_test)}\n")

    # 正样本加权
    pos_weights_train = get_pos_weights(df.dropna(subset=df.columns), LABELS)

    # 模型初始化
    print("Initializing model...")
    model = TextAssessor(n1=2, n2=[3, 2]).to(device)
    print(f"Model: {model}\n")

    # 逐阶段训练 + 早停
    best_val_ap = -1.0  # 最佳验证集宏AUPRC
    best_state = None   # 最佳模型状态
    train_losses = []   # 训练损失
    val_accs = []       # 验证准确率

    for phase in PHASES:
        print("=" * 20 + f"\t{phase['name']}\t" + "=" * 20)
        # 解冻参数
        enable_params(model, phase['name'])
        # optimizer
        weight_decay = phase.get('weight_decay', WEIGHT_DECAY_PHASE) # 1-6阶段为1e-3, 最终阶段为1e-4
        optimizer = build_optimizer(model, phase['lrs'], weight_decay=weight_decay)

        no_improve = 0 # 连续未改进的epoch计数
        for epoch in range(1, phase['epochs'] + 1):
            print(f"Epoch {epoch}/{phase['epochs']}")
            # 单次训练
            tr_loss, _ = train_epoch(model=model, dataloader=dl_train, optimizer=optimizer, device=device, pos_weights=pos_weights_train)
            
            # 进行验证
            val_probs, val_true = collect_probs_targets(model, dl_val)
            # 以0.5为阈值进行验证集上的平均f1分数计算度量准确率
            val_pred05 = (val_probs >= 0.5).astype(np.float32)
            try:
                f1_macro_05 = float(np.mean([f1_score(val_true[:, j], val_pred05[:, j]) for j in range(len(LABELS))]))
            except Exception:
                f1_macro_05 = float('nan')
            # 宏平均 AUPRC
            aprs = []
            for j in range(len(LABELS)):
                yj = val_true[:, j]
                pj = val_probs[:, j]
                if len(np.unique(yj)) > 1:
                    aprs.append(average_precision_score(yj, pj))
            val_macro_ap = float(np.mean(aprs)) if len(aprs) else -1.0

            # 记录训练损失和验证准确率
            train_losses.append(tr_loss)
            val_accs.append(f1_macro_05)

            print(f"Val macro-AUPRC: {val_macro_ap:.4f}\tVal F1@0.5 (macro): {f1_macro_05:.4f}")

            improved = val_macro_ap > best_val_ap
            if improved:
                best_val_ap = val_macro_ap
                best_state = {
                    'model': model.state_dict(),
                    'phase': phase['name'] + '_' + f"{epoch}",
                }
                torch.save(best_state, OUTPUT_DIR + '/' + 'best_model.pth')
                print(f"保存新的最优模型: {phase['name']} (epoch {epoch})")
                # 重置未改进计数
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"触发早停！stop: {phase['name']}\t(no_improve: {PATIENCE} epochs).")
                break

        # 保存每个阶段的最后模型
        torch.save({'model': model.state_dict()}, OUTPUT_DIR + '/' + f'{phase["name"]}_last.pth')
        print(f"保存阶段模型: {phase['name']}_last.pth\n")

    # 加载最佳模型状态
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        model.eval()
        print(f"加载最优模型: {best_state['phase']}\n")

    # 确定验证集上的阈值
    val_probs, val_true = collect_probs_targets(model, dl_val)
    thresholds = find_best_thresholds(val_true, val_probs)
    with open(OUTPUT_DIR + '/' + 'thresholds.json', 'w') as f:
        json.dump({label: thr for label, thr in zip(LABELS, thresholds)}, f, indent=2)

    # 记录最优指标
    metrics_val = compute_metrics(val_true, val_probs, thresholds)
    save_metrics_csv(OUTPUT_DIR + '/' + 'metrics_val.csv', metrics_val)

    # 测试集评估
    test_probs, test_true = collect_probs_targets(model, dl_test)
    metrics_test = compute_metrics(test_true, test_probs, thresholds)
    save_metrics_csv(OUTPUT_DIR + '/' + 'metrics_test.csv', metrics_test)

    # 保存测试结果
    cols = []
    data = {}
    for j, label in enumerate(LABELS):
        yj = test_true[:, j]
        pj = test_probs[:, j]
        yhat = (pj >= thresholds[j]).astype(np.float32)
        data[f'{label}.y'] = yj
        data[f'{label}.p'] = pj
        data[f'{label}.yhat'] = yhat
        cols += [f'{label}.y', f'{label}.p', f'{label}.yhat']
    pd.DataFrame(data, columns=cols).to_csv(OUTPUT_DIR + '/' + 'preds_test.csv', index=False)

    # 保存训练损失和验证准确率
    pd.DataFrame({
        'train_loss': train_losses,
        'val_f1_macro_05': val_accs,
    }).to_csv(OUTPUT_DIR + '/' + 'train_val_metrics.csv', index=False)

    # 作图 训练损失与验证准确率
    try:
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_accs, label='Val F1@0.5 (macro)')
        plt.title('Val F1@0.5')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + '/' + 'curves.svg')
        plt.close()
    except Exception as e:
        print(f"{e}")

if __name__ == '__main__':
    main()
