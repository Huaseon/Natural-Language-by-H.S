from . import SEED

from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from ast import literal_eval
import torch

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int):
        self.df = df
        self.tokenizer = tokenizer
        self.df.reset_index(drop=True, inplace=True)
        self.to_literal_eval()
        self.max_len = max_len
    
    def to_literal_eval(self):
        for col in ['Text', 'PF_TARGET', 'LRA_TARGET']:
            if isinstance(self.df[col][0], str):
                self.df[col] = self.df[col].apply(literal_eval)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cur = self.df.iloc[idx]
        
        encodings = self.tokenizer(
            text=cur.Text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len
        )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'PF_target': torch.tensor(cur.PF_TARGET, dtype=torch.float),
            'LRA_target': torch.tensor(cur.LRA_TARGET, dtype=torch.float),
            'PF_score': torch.tensor(cur.PF_score, dtype=torch.float),
            'PF_US': torch.tensor(cur.PF_US, dtype=torch.float),
            'PF_neg': torch.tensor(cur.PF_neg, dtype=torch.float),
            'Threat_up': torch.tensor(cur.Threat_up, dtype=torch.float),
            'Threat_down': torch.tensor(cur.Threat_down, dtype=torch.float),
            'Citizen_impact': torch.tensor(cur.Citizen_impact, dtype=torch.float),
        }
