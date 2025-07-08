import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class JobDataset(Dataset):
    """Custom Dataset for job descriptions"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(nn.Module):
    """BERT-based classifier for job title prediction"""
    
    def __init__(self, num_classes, model_name='bert-base-uncased', dropout_rate=0.3):
        super(BertClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained BERT
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        logger.info(f"BERT model initialized with {num_classes} classes on {self.device}")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=3, batch_size=16, learning_rate=2e-5, max_length=512):
        """Train the BERT model"""
        
        logger.info(f"Starting BERT training for {epochs} epochs...")
        
        # Prepare label encoder if needed
        if not hasattr(self, 'label_encoder'):
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train
            
        if X_val is not None and y_val is not None:
            if hasattr(self, 'label_encoder'):
                y_val_encoded = self.label_encoder.transform(y_val)
            else:
                y_val_encoded = y_val
        
        # Create datasets
        train_dataset = JobDataset(X_train, y_train_encoded, self.tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = JobDataset(X_val, y_val_encoded, self.tokenizer, max_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.parameters(), lr=learning_rate, eps=1e-8)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.train()
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            total_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if X_val is not None:
                val_accuracy = self._evaluate(val_loader)
                val_accuracies.append(val_accuracy)
                logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        logger.info("BERT training completed!")
        
        # Store training history
        self.training_history = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies if X_val is not None else []
        }
        
        return self
    
    def _evaluate(self, data_loader):
        """Evaluate model on validation set"""
        self.eval()
        total_accuracy = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy = (predictions == labels).float().mean()
                
                total_accuracy += accuracy.item() * len(labels)
                total_samples += len(labels)
        
        self.train()
        return total_accuracy / total_samples
    
    def predict(self, X):
        """Make predictions on new data"""
        self.eval()
        predictions = []
        
        # Create dataset
        # Use dummy labels for prediction
        dummy_labels = [0] * len(X)
        dataset = JobDataset(X, dummy_labels, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        # Convert back to original labels if label encoder exists
        if hasattr(self, 'label_encoder'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        self.eval()
        all_probabilities = []
        
        # Create dataset
        dummy_labels = [0] * len(X)
        dataset = JobDataset(X, dummy_labels, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Getting probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(outputs.logits, dim=-1)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_probabilities)
    
    @property
    def classes_(self):
        """Get class labels (for sklearn compatibility)"""
        if hasattr(self, 'label_encoder'):
            return self.label_encoder.classes_
        else:
            return np.arange(self.num_classes)


def create_bert_classifier(num_classes, model_name='bert-base-uncased'):
    """Factory function to create BERT classifier"""
    try:
        model = BertClassifier(num_classes, model_name)
        logger.info(f"BERT classifier created successfully with {num_classes} classes")
        return model
    except Exception as e:
        logger.error(f"Failed to create BERT classifier: {str(e)}")
        raise


def train_bert_model(X_train, y_train, X_val, y_val, num_classes, 
                     epochs=3, batch_size=16, learning_rate=2e-5):
    """
    Convenient function to train BERT model

    """
    
    model = create_bert_classifier(num_classes)
    
    model.fit(
        X_train, y_train, 
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    return model


def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")


def get_model_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
        logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Cached: {memory_cached:.1f}MB")
        return memory_allocated, memory_cached
    return 0, 0