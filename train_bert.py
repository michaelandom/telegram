import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG = {
    "csv_path": "huhu_output.csv",
    "text_column": "combined_description_text",
    "label_column": "title",
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 5,
    "test_size": 0.2,
    "random_state": 42,
    "output_dir": "bert_output"
}

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_data():
    df = pd.read_csv(CONFIG["csv_path"])
    logger.info(f"Data loaded with {len(df)} samples")
    
    if CONFIG["text_column"] not in df.columns:
        raise ValueError(f"Text column '{CONFIG['text_column']}' not found in CSV")
    if CONFIG["label_column"] not in df.columns:
        raise ValueError(f"Label column '{CONFIG['label_column']}' not found in CSV")
    
    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df[CONFIG["label_column"]])
    
    X_train, X_val, y_train, y_val = train_test_split(
        df[CONFIG["text_column"]],
        df['encoded_labels'],
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )
    
    return X_train, X_val, y_train, y_val, label_encoder

def train_model():
    X_train, X_val, y_train, y_val, label_encoder = prepare_data()
    
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=len(label_encoder.classes_)
    )
    
    train_dataset = BertDataset(X_train, y_train, tokenizer, CONFIG["max_length"])
    val_dataset = BertDataset(X_val, y_val, tokenizer, CONFIG["max_length"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=0,
        pin_memory=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Avg training loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                _, preds = torch.max(outputs.logits, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate all metrics
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision_micro = precision_score(all_labels, all_preds, average='micro')
        precision_macro = precision_score(all_labels, all_preds, average='macro')
        recall_micro = recall_score(all_labels, all_preds, average='micro')
        recall_macro = recall_score(all_labels, all_preds, average='macro')
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        logger.info(f"\nValidation Metrics - Epoch {epoch + 1}:")
        logger.info(f"Loss: {avg_val_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision (Micro): {precision_micro:.4f}")
        logger.info(f"Precision (Macro): {precision_macro:.4f}")
        logger.info(f"Recall (Micro): {recall_micro:.4f}")
        logger.info(f"Recall (Macro): {recall_macro:.4f}")
        logger.info(f"F1 Score (Micro): {f1_micro:.4f}")
        logger.info(f"F1 Score (Macro): {f1_macro:.4f}")
    
    # Save model
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    logger.info(f"Model saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    train_model()