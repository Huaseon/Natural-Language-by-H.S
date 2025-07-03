from . import SEED
from . import PRETRAINED_MODEL_NAME    
from . import SAVED_MODEL

import torch
from torch import nn
from transformers import BertModel

SEED = 20040508
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class TextAssessor(nn.Module):
    def __init__(self, dropout: float=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.text_encoder = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        
        encode_dim = self.text_encoder.config.hidden_size
        
        self.PF_filter = self._build_filter(in_features=encode_dim)
        self.LRA_filter = self._build_filter(in_features=encode_dim)

        self.PF_assessor = self._build_assessor(in_features=encode_dim)
        self.LRA_assessor = self._build_assessor(in_features=encode_dim)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sentence_embeddings = encoder_outputs.last_hidden_state[:, 0, :]

        PF_probs = self.PF_filter(sentence_embeddings).squeeze(-1)
        LRA_probs = self.LRA_filter(sentence_embeddings).squeeze(-1)

        PF_avg = torch.sum(
            sentence_embeddings * PF_probs.unsqueeze(-1), dim=0
        ) / (PF_probs.sum(dim=0, keepdim=True) + 1e-8)
        LRA_avg = torch.sum(
            sentence_embeddings * LRA_probs.unsqueeze(-1), dim=0
        ) / (LRA_probs.sum(dim=0, keepdim=True) + 1e-8)

        PF_assessments = self.PF_assessor(PF_avg)
        LRA_assessments = self.LRA_assessor(LRA_avg)

        return PF_probs, LRA_probs, PF_assessments, LRA_assessments

    def _build_filter(self, in_features: int):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=in_features // 2, out_features=1),
            nn.Sigmoid()
        )
    
    def _build_assessor(self, in_features: int, n_classes: int=3):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=in_features // 2, out_features=n_classes)
        )

    def save(self):
        torch.save(self.state_dict(), SAVED_MODEL)
        print(f"saved model to {SAVED_MODEL}\n")

    @classmethod
    def loads(cls, device):
        model = cls().to(device)
        model.load_state_dict(torch.load(SAVED_MODEL, map_location=device))
        model.eval()
        return model

import torch.nn.functional as F

def compute_loss(outputs, targets):
    PF_probs, LRA_probs, PF_assessments, LRA_assessments = outputs
    PF_target, LRA_target = targets["PF_target"], targets["LRA_target"]

    loss_PF = F.binary_cross_entropy(PF_probs, PF_target)
    loss_LRA = F.binary_cross_entropy(LRA_probs, LRA_target)

    loss_PF_score = F.binary_cross_entropy_with_logits(PF_assessments[0], targets['PF_score'])
    loss_PF_US = F.binary_cross_entropy_with_logits(PF_assessments[1], targets['PF_US'])
    loss_PF_neg = F.binary_cross_entropy_with_logits(PF_assessments[2], targets['PF_neg'])
    loss_Threat_up = F.binary_cross_entropy_with_logits(LRA_assessments[0], targets['Threat_up'])
    loss_Threat_down = F.binary_cross_entropy_with_logits(LRA_assessments[1], targets['Threat_down'])
    loss_Citizen_impace = F.binary_cross_entropy_with_logits(LRA_assessments[2], targets['Citizen_impact'])

    total_loss = loss_PF + loss_LRA + loss_PF_score + loss_PF_US + loss_PF_neg + loss_Threat_up + loss_Threat_down + loss_Citizen_impace

    return total_loss

from tqdm.auto import tqdm
def train_epoch(model, dataset, optimizer, device, criterion=compute_loss):
    model.train()
    total_loss = .0

    for idx, data in tqdm(enumerate(dataset), desc="Epoch", total=len(dataset), leave=False):
        inputs = {key: value.to(device) for key, value in data.items()}

        optimizer.zero_grad()
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        loss = criterion(outputs, inputs)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(dataset)

    return avg_train_loss

def evaluate(model, dataset, device, criterion=compute_loss):
    model.eval()
    total_loss = .0

    with torch.no_grad():
        for idx, data in tqdm(enumerate(dataset), desc="Evaluation", total=len(dataset), leave=False):
            inputs = {key: value.to(device) for key, value in data.items()}

            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = criterion(outputs, inputs)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataset)

    return avg_loss

import matplotlib.pyplot as plt
def train_model(model, train_dataset, test_dataset, optimizer, device, epochs, criterion=compute_loss):
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", position=0):
        train_loss = train_epoch(model=model, dataset=train_dataset, optimizer=optimizer, device=device, criterion=criterion)
        test_loss = evaluate(model=model, dataset=test_dataset, criterion=criterion, device=device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.savefig('./data/loss-plot.svg')
    
    return model

def predict_one(data_text, model, tokenizer, max_len, device):
    model.eval()
    
    inputs = tokenizer(
        data_text, return_tensors='pt', padding='max_length', truncation=True, max_length = max_len
    )

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
    
    PF_probs, LRA_probs, PF_assessments, LRA_assessments = outputs
    PF_probs, LRA_probs = PF_probs.cpu().numpy(), LRA_probs.cpu().numpy()
    PF_assessments = PF_assessments.sigmoid().cpu().numpy()
    LRA_assessments = LRA_assessments.sigmoid().cpu().numpy()

    return {
        "pred.values.PF_TARGET": PF_probs,
        "pred.values.LRA_probs": LRA_probs,
        "pred.values.PF_score": PF_assessments[0],
        "pred.values.PF_US": PF_assessments[1],
        "pred.values.PF_neg": PF_assessments[2],
        "pred.values.Threat_up": LRA_assessments[0],
        "pred.values.Threat_down": LRA_assessments[1],
        "pred.values.Citizen_impact": LRA_assessments[2],
    }
