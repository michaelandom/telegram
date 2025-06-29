import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class JobTitleClassifier:
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.eval()
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.xgb_model = None

    def load_data(self, path):
        df = pd.read_csv(path)
        df = df[['title', 'combined_description_text']].dropna()
        df = df[df['combined_description_text'].str.strip().astype(bool)]
        self.df = df
        return df

    def encode_labels(self):
        self.df['label'] = self.label_encoder.fit_transform(self.df['title'])
        return self.df

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

    # def embed_all_texts(self):
    #     print("🔁 Generating BERT embeddings...")
    #     self.embeddings = np.vstack([
    #         self.get_bert_embedding(text)
    #         for text in tqdm(self.df['combined_description_text'], desc="Embedding texts")
    #     ])
    #     return self.embeddings
    def embed_all_texts(self, save_path='embeddings.npy', batch_size=128):
        print("🔁 Generating BERT embeddings and writing to disk...")

        embeddings_list = []
        for i in tqdm(range(0, len(self.df), batch_size), desc="Embedding texts"):
            batch_texts = self.df['combined_description_text'].iloc[i:i+batch_size].tolist()
            batch_embeddings = []

            for text in batch_texts:
                emb = self.get_bert_embedding(text)
                batch_embeddings.append(emb)

            embeddings_list.extend(batch_embeddings)

            # Optional: save periodically to avoid data loss on crash
            if i % (10_000) == 0 and i > 0:
                np.save(save_path, np.array(embeddings_list))
                print(f"✅ Saved partial embeddings at index {i}")

        embeddings_array = np.array(embeddings_list)
        np.save(save_path, embeddings_array)
        self.embeddings = embeddings_array
        return self.embeddings

    def train_model(self, test_size=0.2, random_state=42):
        X = self.embeddings
        y = self.df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=len(np.unique(y)),
            use_label_encoder=False,
            eval_metric="mlogloss"
        )

        print("🚀 Training XGBoost model...")
        self.xgb_model.fit(X_train, y_train)

        self.X_test, self.y_test = X_test, y_test
        return self.xgb_model

    def evaluate_model(self):
        y_pred = self.xgb_model.predict(self.X_test)

        print("\n📊 Evaluation Metrics:")
        metrics = {
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'precision_micro': precision_score(self.y_test, y_pred, average='micro'),
            'precision_macro': precision_score(self.y_test, y_pred, average='macro'),
            'recall_micro': recall_score(self.y_test, y_pred, average='micro'),
            'recall_macro': recall_score(self.y_test, y_pred, average='macro'),
            'f1_micro': f1_score(self.y_test, y_pred, average='micro'),
            'f1_macro': f1_score(self.y_test, y_pred, average='macro')
        }

        for k, v in metrics.items():
            if 'matrix' not in k:
                print(f"{k}: {v:.4f}")
        return metrics

    def plot_confusion_matrix(self, figsize=(10, 8), save_path=None):
        y_pred = self.xgb_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = self.label_encoder.classes_

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save(self, model_path='xgb_model.joblib', encoder_path='label_encoder.joblib'):
        joblib.dump(self.xgb_model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Model saved to {model_path}, encoder saved to {encoder_path}")

    def load(self, model_path='xgb_model.joblib', encoder_path='label_encoder.joblib'):
        self.xgb_model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        print("Model and encoder loaded")


if __name__ == "__main__":
    clf = JobTitleClassifier()

    clf.load_data("job_output.csv")
    clf.encode_labels()

    clf.embed_all_texts()

    clf.train_model()
    clf.evaluate_model()
    clf.plot_confusion_matrix()

    clf.save()
