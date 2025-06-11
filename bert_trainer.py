import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import logging
import numpy as np
import os
import json
import warnings
from typing import Tuple, Dict, Any
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
# CRITICAL: Set multiprocessing start method to avoid segmentation faults
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Suppress the expected BERT warning
warnings.filterwarnings(
    "ignore", message="Some weights of BertForSequenceClassification were not initialized")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class BertTrainer:
    def __init__(self,
                 csv_path: str = "hahu_output.csv",
                 text_column: str = "combined_description_text",
                 label_column: str = "title",
                 model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 epochs: int = 4,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 output_dir: str = "bert_output",
                 early_stopping_patience: int = 3,
                 warmup_ratio: float = 0.1):

        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir
        self.early_stopping_patience = early_stopping_patience
        self.warmup_ratio = warmup_ratio

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_data(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, LabelEncoder]:
        """Load and prepare the data for training."""
        df = pd.read_csv(self.csv_path)
        logger.info(f"Data loaded with {len(df)} samples")

        # Validate columns exist
        if self.text_column not in df.columns:
            raise ValueError(
                f"Text column '{self.text_column}' not found in CSV")
        if self.label_column not in df.columns:
            raise ValueError(
                f"Label column '{self.label_column}' not found in CSV")

        # Remove rows with missing text or labels
        initial_len = len(df)
        df = df.dropna(subset=[self.text_column, self.label_column])
        if len(df) < initial_len:
            logger.info(
                f"Removed {initial_len - len(df)} rows with missing data")

        # Encode labels
        label_encoder = LabelEncoder()
        df['encoded_labels'] = label_encoder.fit_transform(
            df[self.label_column])

        logger.info(f"Number of unique labels: {len(label_encoder.classes_)}")
        logger.info(
            f"Label distribution:\n{df[self.label_column].value_counts()}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            df[self.text_column],
            df['encoded_labels'],
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df['encoded_labels']  # Ensure balanced splits
        )

        logger.info(
            f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Save label encoder
        label_mapping = {i: label for i,
                         label in enumerate(label_encoder.classes_)}
        with open(os.path.join(self.output_dir, 'label_mapping.json'), 'w') as f:
            json.dump(label_mapping, f, indent=2)

        return X_train, X_val, y_train, y_val, label_encoder

    def evaluate_model(self, model, data_loader, device, epoch, label_encoder) -> Dict[str, float]:
        """Evaluate the model and return metrics."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                _, preds = torch.max(outputs.logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision_micro = precision_score(
            all_labels, all_preds, average='micro')
        precision_macro = precision_score(
            all_labels, all_preds, average='macro', zero_division=0)
        recall_micro = recall_score(all_labels, all_preds, average='micro')
        recall_macro = recall_score(
            all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_macro = f1_score(all_labels, all_preds,
                            average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
        self._plot_confusion_matrix(
            conf_matrix,
            class_names=list(label_encoder.classes_),
            epoch=epoch + 1,
            output_dir=self.output_dir
        )
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision_micro': precision_micro,
            'precision_macro': precision_macro,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'predictions': all_preds,
            'labels': all_labels
        }

    def _plot_confusion_matrix(self, cm, class_names, epoch, output_dir):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Epoch {epoch}')

        file_path = os.path.join(
            output_dir, f'confusion_matrix_bert_epoch_{epoch}.png')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def train_model(self):
        """Train the BERT model with improvements."""
        logger.info("Starting BERT model training...")

        # Prepare data
        X_train, X_val, y_train, y_val, label_encoder = self.prepare_data()

        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(label_encoder.classes_)
        )

        # Create datasets
        train_dataset = BertDataset(
            X_train, y_train, tokenizer, self.max_length)
        val_dataset = BertDataset(X_val, y_val, tokenizer, self.max_length)

        # CRITICAL: Set num_workers=0 to avoid multiprocessing issues
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Important: No multiprocessing
            pin_memory=False  # Disable pin_memory for stability
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=0,  # Important: No multiprocessing
            pin_memory=False  # Disable pin_memory for stability
        )

        # Setup device and move model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model = model.to(device)

        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, eps=1e-8)
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training tracking
        training_history = []
        best_f1 = 0
        patience_counter = 0

        logger.info(
            f"Training for {self.epochs} epochs with {total_steps} total steps")
        logger.info(f"Warmup steps: {warmup_steps}")

        for epoch in range(self.epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            logger.info(f"{'='*50}")

            # Training phase
            model.train()
            total_train_loss = 0
            train_progress = tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}")

            for batch in train_progress:
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
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            logger.info("\nEvaluating on validation set...")
            val_metrics = self.evaluate_model(
                model, val_loader, device, epoch, label_encoder)

            # Log results
            logger.info(f"\nEpoch {epoch + 1} Results:")
            logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Validation Loss: {val_metrics['loss']:.4f}")
            logger.info(
                f"Precision F1 (Macro): {val_metrics['precision_macro']:.4f}")
            logger.info(
                f"Precision F1 (Micro): {val_metrics['precision_micro']:.4f}")
            logger.info(
                f"Recall F1 (Macro): {val_metrics['recall_macro']:.4f}")
            logger.info(
                f"Recall F1 (Micro): {val_metrics['recall_micro']:.4f}")
            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(
                f"Validation F1 (Macro): {val_metrics['f1_macro']:.4f}")
            logger.info(
                f"Validation F1 (Micro): {val_metrics['f1_micro']:.4f}")
            # Save training history - CONVERT TO PYTHON NATIVE TYPES
            epoch_data = {
                'epoch': int(epoch + 1),  # Ensure it's a Python int
                # Ensure it's a Python float
                'train_loss': float(avg_train_loss),
            }

            # Convert all val_metrics to native Python types
            for key, value in val_metrics.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    epoch_data[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    epoch_data[key] = float(value)
                else:
                    epoch_data[key] = value

            training_history.append(epoch_data)

            # Early stopping check
            # Ensure it's a Python float
            current_f1 = float(val_metrics['f1_macro'])
            if current_f1 > best_f1:
                best_f1 = current_f1
                patience_counter = 0

                # Save best model
                model.save_pretrained(os.path.join(
                    self.output_dir, 'best_model'))
                tokenizer.save_pretrained(
                    os.path.join(self.output_dir, 'best_model'))
                logger.info(
                    f"New best model saved with F1 score: {best_f1:.4f}")
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")

            if patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Save final model and training history
        model.save_pretrained(os.path.join(self.output_dir, 'final_model'))
        tokenizer.save_pretrained(os.path.join(self.output_dir, 'final_model'))

        # JSON serialization with proper error handling
        try:
            with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
                json.dump(training_history, f, indent=2,
                          default=convert_to_serializable)
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # Fallback: save with pickle if JSON fails
            import pickle
            with open(os.path.join(self.output_dir, 'training_history.pkl'), 'wb') as f:
                pickle.dump(training_history, f)
            logger.info(
                "Training history saved as pickle file due to JSON serialization issues")

        # Generate final classification report
        best_epoch_idx = max(range(len(training_history)),
                             key=lambda i: training_history[i]['f1_macro'])
        best_metrics = training_history[best_epoch_idx]

        logger.info(f"\n{'='*60}")
        logger.info("TRAINING COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Best epoch: {best_metrics['epoch']}")
        logger.info(
            f"Best validation F1 (macro): {best_metrics['f1_macro']:.4f}")
        logger.info(
            f"Best validation accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"Models saved to: {self.output_dir}")

        return training_history


def convert_to_serializable(obj):
    """Helper function to convert non-serializable objects to serializable ones."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    """Main function to be called from other files."""
    try:
        trainer = BertTrainer(
            csv_path="hahu_output.csv",
            epochs=5,
            batch_size=16,
            learning_rate=2e-5,
            early_stopping_patience=3
        )
        history = trainer.train_model()
        return history

    except FileNotFoundError:
        logger.error("CSV file not found. Please check the file path.")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
