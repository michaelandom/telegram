import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "csv_path": "huhu_output.csv",
    "text_column": "combined_description_text",  # Column containing text data
    "label_column": "title",  # Column containing labels
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "test_size": 0.2,
    "random_state": 42,
    "output_dir": "bert_output"
}

# Custom Dataset Class
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

# Data Preparation
def prepare_data():
    # Load data
    df = pd.read_csv(CONFIG["csv_path"])
    logger.info(f"Data loaded with {len(df)} samples")
    
    # Check columns
    if CONFIG["text_column"] not in df.columns:
        raise ValueError(f"Text column '{CONFIG['text_column']}' not found in CSV")
    if CONFIG["label_column"] not in df.columns:
        raise ValueError(f"Label column '{CONFIG['label_column']}' not found in CSV")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df[CONFIG["label_column"]])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df[CONFIG["text_column"]],
        df['encoded_labels'],
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )
    
    return X_train, X_val, y_train, y_val, label_encoder

# Training Function
def train_model():
    # Prepare data
    X_train, X_val, y_train, y_val, label_encoder = prepare_data()
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=len(label_encoder.classes_)
    )
    
    # Create datasets
    train_dataset = BertDataset(X_train, y_train, tokenizer, CONFIG["max_length"])
    val_dataset = BertDataset(X_val, y_val, tokenizer, CONFIG["max_length"])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=0
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    total_steps = len(train_loader) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
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
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
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
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        logger.info(f"Validation loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    logger.info(f"Model saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    train_model()