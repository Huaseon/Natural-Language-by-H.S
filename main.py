SEED = 20040508

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# %%
from model.text_assessment import *
import pandas as pd
print("source: train_target_df.csv")
df = pd.read_csv('./data/train_target_df.csv')
print(df.head())
print()

# %%
from sklearn.model_selection import train_test_split as tts
train_data, test_data = tts(df, test_size=0.2, random_state=SEED)
print(f"train_data shape: {train_data.shape}\t\ttest_data shape: {test_data.shape}\n")

# %%
max_len = 256

# %%
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
train_dataset = TextDataset(df=train_data, tokenizer=tokenizer, max_len=max_len)
test_dataset = TextDataset(df=test_data, tokenizer=tokenizer, max_len=max_len)

# %%
print("test data loading...")
sample_data = next(iter(train_dataset))
print(f"Sample data - input_ids shape: {sample_data['input_ids'].shape}")
print(f"Sample data - attention_mask shape: {sample_data['attention_mask'].shape}")
print(f"Sample data - PF_targets: {sample_data['PF_target']}")
print(f"Sample data - LRA_targets: {sample_data['LRA_target']}")
print(f"Sample data - PF_score: {sample_data['PF_score']}")
print(f"Sample data - PF_US: {sample_data['PF_US']}")
print(f"Sample data - PF_neg: {sample_data['PF_neg']}")
print(f"Sample data - Threat_up: {sample_data['Threat_up']}")
print(f"Sample data - Threat_down: {sample_data['Threat_down']}")
print(f"Sample data - Citizen_impact: {sample_data['Citizen_impact']}\n")

# %%
print("source TextAssessor:")
model = TextAssessor().to(device)
print(model)
print()

# %%
print("test model running...")
with torch.no_grad():
    outputs = model(input_ids=sample_data['input_ids'].to(device), attention_mask=sample_data['attention_mask'].to(device))
print(outputs)
print()

# %%
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=2e-5)

# %%
from model.text_assessment.text_assessor import train_model

model = train_model(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    optimizer=optimizer,
    device=device,
    epochs=1
)
model.save()

print("Model training completed and saved.\n")

