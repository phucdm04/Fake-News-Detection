# -*- coding: utf-8 -*-
"""(Final) BERT-XLNet-model

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1P0CllVY8KyzxCIFhzmrZxPGAmLwKI5qo
"""

#Global Imports
import os
import random
import numpy as np
import torch
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    get_scheduler, PreTrainedModel, PreTrainedTokenizer
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import kagglehub
from typing import Type, Optional, Dict

#Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Load & prepare data
def load_and_prepare_data():
    path = kagglehub.dataset_download("tisdang/pps-data")
    llm_path = os.path.join(path, 'llm')

    with open(os.path.join(llm_path, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)

    with open(os.path.join(llm_path, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)

    with open(os.path.join(llm_path, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    train_data = train_data.dropna(subset=['text'])
    val_data = val_data.dropna(subset=['text'])
    test_data = test_data.dropna(subset=['text'])

    label_map = {0: "Fake", 1: "True"}

    return train_data, val_data, test_data, label_map

#Tokenize
def tokenize(df, tokenizer, max_len):
    encodings = tokenizer(
        list(df["text"]),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    labels = torch.tensor(df["labels"].values)
    return encodings, labels

#Dataset & DataLoader
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }

def create_loader(encodings, labels, batch_size=16, shuffle=False, num_workers=0, seed=42):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset = FakeNewsDataset(encodings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=worker_init_fn)

#Trainer
class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs=3, lr=2e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        self.scheduler = get_scheduler("linear", self.optimizer, 0, total_steps)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            all_preds, all_labels = [], []
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)

            for batch in loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            acc = accuracy_score(all_labels, all_preds)
            tqdm.write(f"\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
            self.evaluate(self.val_loader, name="Validation")

    def evaluate(self, loader, name="Validation"):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {name}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        tqdm.write(f"{name} Loss: {avg_loss:.4f}")
        tqdm.write(f"{name} Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
        tqdm.write("\n" + classification_report(all_labels, all_preds, digits=2))

        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        tqdm.write("Confusion Matrix (normalized):")
        for i, row in enumerate(cm_normalized):
            tqdm.write(f"{i}: [{', '.join(f'{val:.2f}' for val in row)}]")

#Inference
class Inference:
    def __init__(self, model, tokenizer, max_len):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                truncation=True, max_length=self.max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data, val_data, test_data, label_map = load_and_prepare_data()

def llm_pipeline(
    model_name: str,
    model_class: Type[PreTrainedModel],
    tokenizer_class: Type[PreTrainedTokenizer],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_len: int = 64,
    batch_size: int = 8,
    epochs: int = 3,
    lr: float = 2e-5,
    num_workers: int = 2,
    label_map: Optional[Dict[int, str]] = None
) -> None:

    if label_map is None:
        label_map = {0: "Fake", 1: "True"}

    print(f"\n======== Running {model_name.upper()} ========\n")

    # Tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name)
    if "xlnet-base-cased" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    train_enc, train_labels = tokenize(train_data, tokenizer, max_len)
    val_enc, val_labels = tokenize(val_data, tokenizer, max_len)
    test_enc, test_labels = tokenize(test_data, tokenizer, max_len)

    # DataLoader
    train_loader = create_loader(train_enc, train_labels, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_loader(val_enc, val_labels, batch_size=batch_size, num_workers=num_workers)
    test_loader = create_loader(test_enc, test_labels, batch_size=batch_size, num_workers=num_workers)

    # Model
    model = model_class.from_pretrained(model_name, num_labels=2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nModel architecture:\n")
    print(model)

    # Training
    trainer = Trainer(model, train_loader, val_loader, epochs, lr)
    trainer.train()

    # Evaluation
    print(f"\n----- {model_name.upper()} Evaluation on Test Set -----")
    trainer.evaluate(test_loader, name="Test")

    # Inference
    predictor = Inference(model, tokenizer, max_len)
    examples = [
        "Covid-19 has been completely eradicated without a vaccine.",
        "President Donald Trump again raised the possibility of a U.S. government shutdown on Wednesday.",
        "The Trump administration would find it easier to take apart the exchanges following this move.",
        "Aliens have landed on Earth and are living among us in secret."
    ]
    print(f"\n----- Inference with {model_name.upper()} -----")
    for i, text in enumerate(examples, 1):
        pred_label = predictor.predict(text)
        print(f"- {i}. \"{text}\" → {label_map[pred_label]} (Label {pred_label})")

llm_pipeline("bert-base-uncased", BertForSequenceClassification, BertTokenizer, train_data, val_data, test_data, label_map=label_map)

llm_pipeline("xlnet-base-cased", XLNetForSequenceClassification, XLNetTokenizer, train_data, val_data, test_data)