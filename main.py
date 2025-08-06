SEED = 20040508
FILENAME = 'analysis_result_simple_format_80000_with_summary.xlsx'

# %%
import matplotlib
matplotlib.RcParams.update({
    'font.size': 10.5
})

# %%
import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# %%
from model.text_assessment import *
import pandas as pd
print(f"source: {FILENAME}")
df = pd.read_excel(f'./data/{FILENAME}')
print(df.head())
print()

# %%
from sklearn.model_selection import train_test_split as tts
train_data, test_data = tts(df, test_size=1/3, random_state=SEED)
print(f"train_data shape: {train_data.shape}\t\ttest_data shape: {test_data.shape}\n")

# %%
from model.text_assessment import MAX_LEN
max_len = MAX_LEN

# %%
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
train_dataset = TextDataset(df=train_data, tokenizer=tokenizer, max_len=max_len, device=device)
test_dataset = TextDataset(df=test_data, tokenizer=tokenizer, max_len=max_len, device=device)
print(f"train_dataset.size: {train_dataset.__len__()}\t\ttest_dataset.size: {test_dataset.__len__()}\n")

# %%
from tools import get_pos_weights
targets = ['THREAT_up', 'THREAT_down', 'citizen_impact', 'PF_score', 'PF_US']
pos_weights = {
    "train": get_pos_weights(df, targets),
    "test": get_pos_weights(df, targets)
}
import json
print(f"Positive weights for targets: {json.dumps(pos_weights, indent=2)}\n")

# %%
print("test data loading...")
sample_data = next(iter(train_dataset))
print(f"Sample data - input_ids shape: {sample_data['input_ids'].shape}")
print(f"Sample data - attention_mask shape: {sample_data['attention_mask'].shape}")
print(f"Sample data - THREAT_up: {sample_data['THREAT_up']}")
print(f"Sample data - THREAT_down: {sample_data['THREAT_down']}")
print(f"Sample data - citizen_impact: {sample_data['citizen_impact']}")
print(f"Sample data - PF_score: {sample_data['PF_score']}")
print(f"Sample data - PF_US: {sample_data['PF_US']}\n")

# %%
from torch.utils.data import DataLoader
from model.text_assessment.text_dataset import collate_fn
batch_size = 6
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print(f"train_dataloader length: {len(train_dataloader)}\ttest_dataloader length: {len(test_dataloader)}\n")

# %%
print("source TextAssessor:")
model = TextAssessor(n1=2, n2=[3, 2]).to(device)
for param in model.text_encoder.parameters():
    param.requires_grad = False
print(model)
print()

# %% 第一阶段
print("=" * 30 + "\tTraining Phase 1\t" + "=" * 30)
from torch.optim import AdamW
optimizer = AdamW(
    [
        {
            'params': model.outputs.parameters(),
            'lr': 1e-6,
        },
    ], weight_decay=1e-3
)
print(f"oprimizer: {json.dumps(optimizer.state_dict(), indent=2)}\n")

from model.text_assessment.text_assessor import train_model
model, losses, accs = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    device=device,
    epochs=1,
    pos_weights=pos_weights
)

model.save()
print("Model training completed and saved.\n")

# %% 第二阶段
print("=" * 30 + "\tTraining Phase 2\t" + "=" * 30)
for param in model.text_encoder.pooler.parameters():
    param.requires_grad = True

optimizer = AdamW(
    [
        {
            'params': model.outputs.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.pooler.parameters(),
            'lr': 8e-7,
        }
    ], weight_decay=1e-3
)
print(f"oprimizer: {json.dumps(optimizer.state_dict(), indent=2)}\n")

model, losses, accs = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    device=device,
    epochs=1,
    pos_weights=pos_weights,
    losses=losses,
    accs=accs
)

model.save()
print("Model training completed and saved.\n")

# %% 第三阶段
print("=" * 30 + "\tTraining Phase 3\t" + "=" * 30)
for param in model.text_encoder.encoder.layer[-1:].parameters():
    param.requires_grad = True

optimizer = AdamW(
    [
        {
            'params': model.outputs.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.pooler.parameters(),
            'lr': 8e-7,
        }, {
            'params': model.text_encoder.encoder.layer[-1:].parameters(),
            'lr': 4e-7,
        }
    ], weight_decay=1e-3
)
print(f"oprimizer: {json.dumps(optimizer.state_dict(), indent=2)}\n")

model, losses, accs = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    device=device,
    epochs=1,
    pos_weights=pos_weights,
    losses=losses,
    accs=accs
)

model.save()
print("Model training completed and saved.\n")

# %% 第四阶段
print("=" * 30 + "\tTraining Phase 4\t" + "=" * 30)
for param in model.text_encoder.encoder.layer[-3:-1].parameters():
    param.requires_grad = True

optimizer = AdamW(
    [
        {
            'params': model.outputs.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.pooler.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.encoder.layer[-1:].parameters(),
            'lr': 8e-7,
        }, {
            'params': model.text_encoder.encoder.layer[-3:-1].parameters(),
            'lr': 4e-7,
        }
    ], weight_decay=1e-3
)
print(f"oprimizer: {json.dumps(optimizer.state_dict(), indent=2)}\n")

model, losses, accs = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    device=device,
    epochs=1,
    pos_weights=pos_weights,
    losses=losses,
    accs=accs
)

model.save()
print("Model training completed and saved.\n")

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses.get('train'), label='Train Loss')
plt.plot(losses.get('test'), label='Test Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1, 2, 2)
plt.plot(accs.get('train'), label='Train Accuracy')
plt.plot(accs.get('test'), label='Test Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.savefig('./data/loss-plot_A(4).svg')
plt.close()

model.save(save_model='./data/text_assessor_A(4).pth')

# %% 第五阶段
print("=" * 30 + "\tTraining phase 5\t" + "=" * 30)
for param in model.text_encoder.encoder.layer[-7:-3].parameters():
    param.requires_grad = True

optimizer = AdamW(
    [
        {
            'params': model.outputs.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.pooler.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.encoder.layer[-1:].parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.encoder.layer[-3:-1].parameters(),
            'lr': 8e-7,
        }, {
            'params': model.text_encoder.encoder.layer[-7:-3].parameters(),
            'lr': 4e-7,
        }
    ], weight_decay=1e-3
)
print(f"oprimizer: {json.dumps(optimizer.state_dict(), indent=2)}\n")

model, losses, accs = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    device=device,
    epochs=1,
    pos_weights=pos_weights,
    losses=losses,
    accs=accs
)

model.save()
print("Model training completed and saved.\n")

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses.get('train'), label='Train Loss')
plt.plot(losses.get('test'), label='Test Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1, 2, 2)
plt.plot(accs.get('train'), label='Train Accuracy')
plt.plot(accs.get('test'), label='Test Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.savefig('./data/loss-plot_B(5).svg')
plt.close()

model.save(save_model='./data/text_assessor_B(5).pth')

# %% 最后阶段
print("=" * 30 + "\tFinal Model State\t" + "=" * 30)
for param in model.text_encoder.encoder.layer.parameters():
    param.requires_grad = True
for param in model.text_encoder.embeddings.parameters():
    param.requires_grad = True

optimizer = AdamW(
    [
        {
            'params': model.outputs.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.pooler.parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.encoder.layer[-1:].parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.encoder.layer[-3:-1].parameters(),
            'lr': 1e-6,
        }, {
            'params': model.text_encoder.encoder.layer[-7:-3].parameters(),
            'lr': 8e-7,
        }, {
            'params': model.text_encoder.encoder.layer[:-7].parameters(),
            'lr': 4e-7,
        }, {
            'params': model.text_encoder.embeddings.parameters(),
            'lr': 2e-7,
        }
    ], weight_decay=1e-4
)
print(f"oprimizer: {json.dumps(optimizer.state_dict(), indent=2)}\n")

batch_size = 6
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print(f"train_dataloader length: {len(train_dataloader)}\ttest_dataloader length: {len(test_dataloader)}\n")

model, losses, accs = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    device=device,
    epochs=2,
    pos_weights=pos_weights,
    losses={'train': [], 'test': []},
    accs={'train': [], 'test': []}
)

model.save()
print("Model training completed and saved.\n")

# %%
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses.get('train'), label='Train Loss')
plt.plot(losses.get('test'), label='Test Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1, 2, 2)
plt.plot(accs.get('train'), label='Train Accuracy')
plt.plot(accs.get('test'), label='Test Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.savefig('./data/loss-plot_C(7).svg')
plt.close()

model.save(save_model='./data/text_assessor_C(7).pth')

