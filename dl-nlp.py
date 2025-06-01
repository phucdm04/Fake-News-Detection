# %% import packages
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Tuple
from tqdm import tqdm
import numpy as np
import random
import pickle

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %% create class
def load_data(X_path: str, y_path: str, batch_size: int) -> torch.utils.data.DataLoader:
    with open(X_path, 'rb') as f:
        text_vector = pickle.load(f)
    with open(y_path, 'rb') as f:
        labels = pickle.load(f)
        
    return DataLoader(TensorDataset(torch.tensor(text_vector, dtype=torch.long), torch.tensor(labels)),
                      batch_size = batch_size, shuffle=True)
    

train_loader = load_data(X_path = '/kaggle/input/pps-data/dl/train_text.pkl',
                        y_path = '/kaggle/input/pps-data/dl/train_labels.pkl',
                        batch_size = 64)
test_loader = load_data(X_path = '/kaggle/input/pps-data/dl/test_text.pkl',
                        y_path = '/kaggle/input/pps-data/dl/test_labels.pkl',
                        batch_size = 64)
val_loader = load_data(X_path = '/kaggle/input/pps-data/dl/val_text.pkl',
                        y_path = '/kaggle/input/pps-data/dl/val_labels.pkl',
                        batch_size = 64)

class GRUTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1821,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 1,
        device: str = 'cpu'
    ) -> None:
        super(GRUTextClassifier, self).__init__()
        
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru: nn.GRU = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc: nn.Linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()
        self.device: str = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, sequence_length), type torch.LongTensor
        
        Returns:
            Tensor: Output probabilities of shape (batch_size, 1), values in [0,1]
        """
        x = x.long().to(self.device)
        embedded = self.embedding(x)                         # (B, T, E)
        gru_out, _ = self.gru(embedded)                      # (B, T, H)
        last_hidden = gru_out[:, -1, :]                      # (B, H)
        output = self.fc(last_hidden)                        # (B, 1)
        return self.sigmoid(output)

class LSTMTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1821,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 1,
        device: str = 'cpu'
    ) -> None:
        super(LSTMTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]
        out = self.fc(hidden)
        return self.sigmoid(out)

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 device: str ='cpu', 
                 lr: float = 1e-3,
                 epochs: int = 10,
                 patience: int = 3) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.9)
        self.patience = patience
        self.epochs = epochs
    
    def evaluate(self, 
                 data_loader: torch.utils.data.DataLoader, 
                 print_confusion_matrix: bool = False
                )-> Tuple[float]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

                preds = (outputs > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        if print_confusion_matrix:
            cm = confusion_matrix(all_labels, all_preds)
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            print(f"Confusion Matrix: {cm_normalized}")

        return avg_loss, accuracy, precision, recall, f1

    def train(self, 
              train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader
             ) -> None:
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)

            for inputs, labels in loop:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                loop.set_postfix(loss=loss.item())

            # Đánh giá trên tập huấn luyện và validation
            avg_train_loss, train_acc, train_prec, train_rec, train_f1 = self.evaluate(train_loader)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.evaluate(val_loader, print_confusion_matrix = True)

            print(f"  Train: loss={avg_train_loss:.4f}, acc={train_acc:.4f}, precision={train_prec:.4f}, recall={train_rec:.4f}, f1={train_f1:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, precision={val_prec:.4f}, recall={val_rec:.4f}, f1={val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

    # def inference(self, inputs):
    #     """
    #     Dự đoán đầu ra cho một batch tensor input.
    #     Đầu vào: `inputs` là tensor có shape [batch_size, seq_len]
    #     Đầu ra: xác suất dự đoán, nhị phân (0 hoặc 1)
    #     """
    #     self.model.eval()
    #     inputs = inputs.to(self.device)

    #     with torch.no_grad():
    #         outputs = self.model(inputs)
    #         probs = torch.sigmoid(outputs)
    #         preds = (probs > 0.5).int()

    #     return preds.cpu().numpy(), probs.cpu().numpy()

# %% define main
def main():
    set_seed()
    gru_config = {
        "embedding_dim": 128, 
        "hidden_dim": 256, 
        "output_dim": 1, 
        "device":'cuda' if torch.cuda.is_available() else 'cpu', 
    }
    gru = GRUTextClassifier(**gru_config)
    gru_trainer_config = {
        "model": gru,
        "lr": 5e-4,
        "epochs": 50,
        "device":'cuda' if torch.cuda.is_available() else 'cpu'
    }
    gru_trainer = Trainer(**gru_trainer_config)
    gru_trainer.train(train_loader, val_loader)
    gru_trainer.evaluate(test_loader, print_confusion_matrix=True)


    
    lstm_config = {
        "embedding_dim": 128, 
        "hidden_dim": 512, 
        "output_dim": 1, 
        "device":'cuda' if torch.cuda.is_available() else 'cpu', 
    }
    lstm = LSTMTextClassifier(**lstm_config)
    
    lstm_trainer_config = {
        "model": lstm,
        "lr": 5e-4,
        "epochs": 50,
        "device":'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    lstm_trainer = Trainer(**lstm_trainer_config, patience=4)

    lstm_trainer.train(train_loader, val_loader)
    lstm_trainer.evaluate(test_loader, print_confusion_matrix=True)
    

if __name__ == "__main__":
    main()

