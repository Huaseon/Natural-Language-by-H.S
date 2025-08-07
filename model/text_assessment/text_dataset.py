from . import SEED
from . import MAX_LEN

from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import torch

from re import split

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int=MAX_LEN, device: str='cpu'):
        self.df = df.dropna(subset=df.columns).copy()
        self.tokenizer = tokenizer
        self.df.reset_index(drop=True, inplace=True)
        self.to_literal()
        self.max_len = max_len
        self.device = device
        self.buff = {}
    
    def to_literal(self):
        self.df['summary'] = self.df.summary.apply(lambda x: split('[\.\;]\s', x.replace('*', '')))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        cur = self.df.iloc[idx]
        
        encodings = self.tokenizer(
            text=cur.summary, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len
        )

        return {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device),
            'THREAT_up': torch.tensor(cur.THREAT_up, device=self.device, dtype=torch.float),
            'THREAT_down': torch.tensor(cur.THREAT_down, device=self.device, dtype=torch.float),
            'citizen_impact': torch.tensor(cur.citizen_impact, device=self.device, dtype=torch.float),
            'PF_score': torch.tensor(cur.PF_score, device=self.device, dtype=torch.float),
            'PF_US': torch.tensor(cur.PF_US, device=self.device, dtype=torch.float),
        }
        
def collate_fn(batch):
    return batch
