from . import SEED
from . import PRETRAINED_MODEL_NAME    
from . import SAVED_MODEL
from . import MAX_LEN
import torch
from torch import nn
from transformers import BertModel

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class TextAssessor(nn.Module):
    def __init__(self, dropout: float=0.7):
        super().__init__()
        self.dropout = dropout
        
        self.text_encoder = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

        encode_dim = self.text_encoder.config.hidden_size
        
        self.PF_filter = self._build_filter(in_features=encode_dim)
        self.LRA_filter = self._build_filter(in_features=encode_dim)

        self.PF_assessor = self._build_assessor(in_features=encode_dim)
        self.LRA_assessor = self._build_assessor(in_features=encode_dim)

    def forward(self, inputs):
        logitss = []
        for input in inputs:
            input_ids, attention_mask = input['input_ids'], input['attention_mask']
            logits = self._forward(input_ids=input_ids, attention_mask=attention_mask)
            logitss.append(logits)
        logitss = torch.stack(logitss, dim=0) # [n, n_classes]
        return logitss
    
    def _forward(self, input_ids, attention_mask):
        # inputs_ids: [n, seq_len]
        # attention_mask: [n, seq_len]
        encoder_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ) # [last_hidden_state, pooler_output]
        
        mask = attention_mask.unsqueeze(-1).expand(encoder_outputs.last_hidden_state.size()).float() # [n, seq_len, encode_dim]
         # [n, encode_dim]
        sentence_embeddings = torch.sum(
            encoder_outputs.last_hidden_state * mask, dim=1
        ) / (torch.sum(mask, dim=1) + 1e-8) # [n, encode_dim]

        PF_probs = self.PF_filter(sentence_embeddings).squeeze(-1) # [n]
        LRA_probs = self.LRA_filter(sentence_embeddings).squeeze(-1) # [n]

        PF_avg = torch.sum(
            sentence_embeddings * PF_probs.unsqueeze(-1), dim=0
        ) / (PF_probs.sum(dim=0, keepdim=True) + 1e-8) # [encode_dim]
        LRA_avg = torch.sum(
            sentence_embeddings * LRA_probs.unsqueeze(-1), dim=0
        ) / (LRA_probs.sum(dim=0, keepdim=True) + 1e-8) # [encode_dim]

        PF_assessments = self.PF_assessor(PF_avg) # [n_classes]
        LRA_assessments = self.LRA_assessor(LRA_avg) # [n_classes]

        return torch.hstack([PF_assessments, LRA_assessments])

    def _build_filter(self, in_features: int, hidden_size_rate: int=4):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // hidden_size_rate),
            nn.LayerNorm(in_features // hidden_size_rate),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=in_features // hidden_size_rate, out_features=1),
            nn.Sigmoid()
        )
    
    def _build_assessor(self, in_features: int, hidden_size_rate: int=4, n_classes: int=3):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // hidden_size_rate),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=in_features // hidden_size_rate, out_features=n_classes)
        )

    def save(self, save_model: str=None):
        torch.save(self.state_dict(), save_model or SAVED_MODEL)
        print(f"saved model to {save_model or SAVED_MODEL}\n")

    @classmethod
    def loads(cls, save_model, device):
        model = cls().to(device)
        model.load_state_dict(torch.load(save_model, map_location=device))
        model.eval()
        print(f"loaded model from {save_model}\n")
        return model

import torch.nn.functional as F

def compute_loss(outputs, targets, pos_weights, device):
    total_losses = .0
    for output, target in zip(outputs, targets):
        total_loss = _compute_loss(output, target, pos_weights, device)
        total_losses += total_loss
    return total_losses

def _compute_loss(output, target, pos_weights, device):

    loss_PF_score = F.binary_cross_entropy_with_logits(output[0], target['PF_score'], pos_weight=torch.tensor(pos_weights.get('PF_score', 1.)).sqrt().to(device))
    loss_PF_US = F.binary_cross_entropy_with_logits(output[1], target['PF_US'], pos_weight=torch.tensor(pos_weights.get('PF_US', 1.)).sqrt().to(device))
    loss_PF_neg = F.binary_cross_entropy_with_logits(output[2], target['PF_neg'], pos_weight=torch.tensor(pos_weights.get('PF_neg', 1.)).sqrt().to(device))
    loss_Threat_up = F.binary_cross_entropy_with_logits(output[3], target['Threat_up'], pos_weight=torch.tensor(pos_weights.get('Threat_up', 1.)).sqrt().to(device))
    loss_Threat_down = F.binary_cross_entropy_with_logits(output[4], target['Threat_down'], pos_weight=torch.tensor(pos_weights.get('Threat_down', 1.)).sqrt().to(device))
    loss_Citizen_impace = F.binary_cross_entropy_with_logits(output[5], target['Citizen_impact'], pos_weight=torch.tensor(pos_weights.get('Citizen_impact', 1.)).sqrt().to(device))

    total_loss = loss_PF_score + loss_PF_US + loss_PF_neg + loss_Threat_up + loss_Threat_down + loss_Citizen_impace

    return total_loss

from tqdm.auto import tqdm
def train_epoch(model, dataloader, optimizer, device, pos_weights, criterion=compute_loss):
    model.train()
    total_loss = .0
    correct = 0

    for idx, batch in tqdm(enumerate(dataloader), desc="Epoch", total=len(dataloader), leave=False):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch, pos_weights=pos_weights, device=device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        correct += acculate_accuracy(outputs, batch)

    avg_train_loss = total_loss / len(dataloader)
    avg_accuracy = 100 * correct / len(dataloader)
    print(f"- Average training loss: {avg_train_loss:.4f}\t\tAccuracy: {avg_accuracy:.2f}%")

    return avg_train_loss, avg_accuracy

def evaluate(model, dataloader, device, pos_weights, criterion=compute_loss):
    model.eval()
    total_loss = .0
    correct = 0

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), desc="Evaluation", total=len(dataloader), leave=False):
            outputs = model(batch)
            loss = criterion(outputs, batch, pos_weights=pos_weights, device=device)

            total_loss += loss.item()

            correct += acculate_accuracy(outputs, batch)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = 100 * correct / len(dataloader)
    print(f"+ Average evaluation loss: {avg_loss:.4f}\t\tAccuracy: {avg_accuracy:.2f}%")

    return avg_loss, avg_accuracy

import matplotlib.pyplot as plt
def train_model(model, train_dataloader, test_dataloader, optimizer, device, epochs, pos_weights={}, criterion=compute_loss, losses={'train': [], 'test': []}, accs={'train': [], 'test': []}) -> tuple[TextAssessor, dict, dict]:
    train_losses = losses.get('train')
    train_accs = accs.get('train')
    test_losses = losses.get('test')
    test_accs = accs.get('test')

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", position=0):
        train_loss, train_acc = train_epoch(model=model, dataloader=train_dataloader, optimizer=optimizer, device=device, criterion=criterion, pos_weights=pos_weights.get('train', {}))
        test_loss, test_acc = evaluate(model=model, dataloader=test_dataloader, criterion=criterion, device=device, pos_weights=pos_weights.get('test', {}))

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if epoch % 10 == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.legend()
            plt.title('Loss over epochs')

            plt.subplot(1, 2, 2)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(test_accs, label='Test Accuracy')
            plt.legend()
            plt.title('Accuracy over epochs')

            plt.savefig('./data/loss-plot.svg')
            plt.close()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.savefig('./data/loss-plot.svg')
    plt.close()
    
    return model, losses, accs

def predict_one(data_text, model, tokenizer, max_len, device):
    model.eval()
    
    inputs = tokenizer(
        data_text, return_tensors='pt', padding='max_length', truncation=True, max_length = max_len
    )

    with torch.no_grad():
        outputs = model._forward(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device))
    
    assessments = outputs.sigmoid().cpu().numpy()

    return {
        "pred.values.PF_score": assessments[0],
        "pred.values.PF_US": assessments[1],
        "pred.values.PF_neg": assessments[2],
        "pred.values.Threat_up": assessments[3],
        "pred.values.Threat_down": assessments[4],
        "pred.values.Citizen_impact": assessments[5],
    }

def acculate_accuracy(outputs, targets):
    accuracy = .0
    for output, target in zip(outputs, targets):
        accuracy += _acculate_accuracy(output, target)

    return accuracy / len(outputs)

def _acculate_accuracy(output, target):
    correct = 0
    total = 0

    assessments = (output.sigmoid() > 0.5).float()

    correct += assessments[0].eq(target['PF_score']).sum().item() + \
                assessments[1].eq(target['PF_US']).sum().item() + \
                assessments[2].eq(target['PF_neg']).sum().item() + \
                assessments[3].eq(target['Threat_up']).sum().item() + \
                assessments[4].eq(target['Threat_down']).sum().item() + \
                assessments[5].eq(target['Citizen_impact']).sum().item()
    total += assessments.numel()

    return correct / (total + 1e-8)