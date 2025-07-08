import os
import time
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pymongo import MongoClient
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class RoBERTaJobClassifier:
    def __init__(self,
                 model_dir: str="roberta_output/best_model",
                 label_map_file: str="roberta_output/label_mapping.json",
                 mongo_uri: str="mongodb://localhost:27017",
                 db_name: str="telegram",
                 collection_name: str="jobs",
                 text_field: str = "description",
                 prediction_field: str = "predicted_job_title_roberta_model",
                 confidence_field: str = "prediction_confidence_roberta_model",
                 max_length: int = 128,
                 batch_size: int = 1000):
        
        self.model_dir = model_dir
        self.label_map_file = label_map_file
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.text_field = text_field
        self.prediction_field = prediction_field
        self.confidence_field = confidence_field
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

        self.label_map = self._load_label_map()

    def _load_label_map(self) -> dict:
        with open(self.label_map_file, "r") as f:
            label_map = json.load(f)
        return {int(k): v for k, v in label_map.items()}

    def classify_texts(self, texts: list[str]) -> tuple[list[str], list[float]]:
        predictions, confidences = [], []
        start_time = time.time()

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Classifying"):
                batch = texts[i:i + self.batch_size]
                encoding = self.tokenizer(batch,
                                          padding=True,
                                          truncation=True,
                                          max_length=self.max_length,
                                          return_tensors="pt")
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)

                top_probs, top_indices = torch.max(probs, dim=1)
                predictions.extend([self.label_map[idx.item()] for idx in top_indices])
                confidences.extend([prob.item() for prob in top_probs])

        print(f"Classification time for {len(texts)} documents: {time.time() - start_time:.2f} seconds")
        return predictions, confidences

    def predict_and_update_mongodb(self):
        start_time = time.time()

        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[self.collection_name]

        query = {
            self.text_field: {"$exists": True, "$ne": ""},
            self.prediction_field: None
        }
        docs = list(collection.find(query))

        if not docs:
            print("No documents found with valid text.")
            return

        texts = [doc[self.text_field] for doc in docs]
        print(f"Found {len(texts)} documents to classify...")

        predictions, confidences = self.classify_texts(texts)

        for doc, pred, conf in zip(docs, predictions, confidences):
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    self.prediction_field: pred,
                    self.confidence_field: round(conf, 4)
                }}
            )

        print(f"Updated {len(docs)} documents.")
        print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    classifier = RoBERTaJobClassifier()
    classifier.predict_and_update_mongodb()
