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
    def __init__(self, n1: int, n2: list, dropout: float=0.7):
        super().__init__()
        assert len(n2) == n1
        self.dropout = dropout
        
        self.text_encoder = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False

        encode_dim = self.text_encoder.config.hidden_size

        self.outputs = nn.ModuleList([
            nn.ModuleDict({
                'filter': self._build_filter(in_features=encode_dim),
                'assessor': self._build_assessor(in_features=encode_dim, n_classes=_n2)
            }) for _n2 in n2
        ])

    def forward(self, inputs):
        logitss = []
        for input in inputs:
            input_ids, attention_mask = input['input_ids'], input['attention_mask']
            logits = self._forward(input_ids=input_ids, attention_mask=attention_mask)
            logitss.append(logits)
        logitss = torch.stack(logitss, dim=0) # [n, n2]
        return logitss
    
    def _forward(self, input_ids, attention_mask):
        # inputs_ids: [n, seq_len]
        # attention_mask: [n, seq_len]
        encoder_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ) # [last_hidden_state, pooler_output]
        
        mask = attention_mask.unsqueeze(-1).expand(encoder_outputs.last_hidden_state.size()).float() # [n, seq_len, encode_dim]
        sentence_embeddings = torch.sum(
            encoder_outputs.last_hidden_state * mask, dim=1
        ) / (torch.sum(mask, dim=1) + 1e-8) # [n, encode_dim]

        logits = []
        
        for output in self.outputs:
            filter = output['filter']
            assessor = output['assessor']
            
            probs = filter(sentence_embeddings) # [n, encode_dim]
            avg = torch.sum(
                sentence_embeddings * probs, dim=0
            ) / (probs.sum(dim=0) + 1e-8) # [encode_dim]
            assessments = assessor(avg) # [_n2]
            logits.append(assessments)

        return torch.hstack(logits) # [n, n2]

    def _build_filter(self, in_features: int, hidden_size_rate: int=4):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features * hidden_size_rate),
            nn.LayerNorm(in_features * hidden_size_rate),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=in_features * hidden_size_rate, out_features=in_features),
            nn.Sigmoid()
        )
    
    def _build_assessor(self, in_features: int, hidden_size_rate: int=4, n_classes: int=3):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features * hidden_size_rate),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=in_features * hidden_size_rate, out_features=n_classes)
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

    loss_Threat_up = F.binary_cross_entropy_with_logits(output[0], target['THREAT_up'], pos_weight=torch.tensor(pos_weights.get('THREAT_up', 1.)).sqrt().to(device))
    loss_Threat_down = F.binary_cross_entropy_with_logits(output[1], target['THREAT_down'], pos_weight=torch.tensor(pos_weights.get('THREAT_down', 1.)).sqrt().to(device))
    loss_Citizen_impace = F.binary_cross_entropy_with_logits(output[2], target['citizen_impact'], pos_weight=torch.tensor(pos_weights.get('citizen_impact', 1.)).sqrt().to(device))
    loss_PF_score = F.binary_cross_entropy_with_logits(output[3], target['PF_score'], pos_weight=torch.tensor(pos_weights.get('PF_score', 1.)).sqrt().to(device))
    loss_PF_US = F.binary_cross_entropy_with_logits(output[4], target['PF_US'], pos_weight=torch.tensor(pos_weights.get('PF_US', 1.)).sqrt().to(device))
    
    total_loss = loss_PF_score + loss_PF_US + loss_Threat_up + loss_Threat_down + loss_Citizen_impace

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

        correct += caculate_accuracy(outputs, batch)

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

            correct += caculate_accuracy(outputs, batch)

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
        if epoch % 1 == 0:
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
        "pred.values.THREAT_up": assessments[0],
        "pred.values.THREAT_down": assessments[1],
        "pred.values.citizen_impact": assessments[2],
        "pred.values.PF_score": assessments[3],
        "pred.values.PF_US": assessments[4],
    }

def caculate_accuracy(outputs, targets):
    accuracy = .0
    for output, target in zip(outputs, targets):
        accuracy += _caculate_accuracy(output, target)

    return accuracy / len(outputs)

def _caculate_accuracy(output, target):
    correct = 0
    total = 0

    assessments = (output.sigmoid() > 0.5).float()

    correct +=  assessments[0].eq(target['THREAT_up']).sum().item() + \
                assessments[1].eq(target['THREAT_down']).sum().item() + \
                assessments[2].eq(target['citizen_impact']).sum().item() + \
                assessments[3].eq(target['PF_score']).sum().item() + \
                assessments[4].eq(target['PF_US']).sum().item()

    total += assessments.numel()

    return correct / (total + 1e-8)