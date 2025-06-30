from tqdm import tqdm
import time
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from pymongo import MongoClient
import json
import os
from tqdm import tqdm
import torch.nn.functional as F

MODEL_DIR = "bert_output/best_model"
LABEL_MAP_FILE = "bert_output/label_mapping.json"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "telegram"
COLLECTION_NAME = "jobs"
TEXT_FIELD = "description"
PREDICTION_FIELD = "predicted_job_title_bert_model"
CONFIDENCE_FIELD = "prediction_confidence_bert_model"
MAX_LENGTH = 128
BATCH_SIZE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model = model.to(DEVICE)
model.eval()

with open(LABEL_MAP_FILE, "r") as f:
    label_mapping = json.load(f)

id_to_label = {int(k): v for k, v in label_mapping.items()}


def classify_texts(texts):
    """Classifies a list of texts and returns predictions and confidences."""
    predictions = []
    confidences = []

    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Classifying"):
            batch = texts[i:i + BATCH_SIZE]
            encoding = tokenizer(batch,
                                 padding=True,
                                 truncation=True,
                                 max_length=MAX_LENGTH,
                                 return_tensors="pt")
            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

            top_probs, top_indices = torch.max(probs, dim=1)
            preds = [id_to_label[idx.item()] for idx in top_indices]
            confs = [prob.item() for prob in top_probs]

            predictions.extend(preds)
            confidences.extend(confs)
            

    end_time = time.time()
    print(f"Classification time for {len(texts)} documents: {end_time - start_time:.2f} seconds")

    return predictions, confidences


def predict_and_update_mongodb():
    """Reads data from MongoDB, predicts labels, and updates documents."""
    start_total = time.time()

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    cursor = collection.find({TEXT_FIELD: {"$exists": True, "$ne": ""}})
    docs = list(cursor)

    if not docs:
        print("No documents found with valid text.")
        return

    texts = [f"{doc["title"]} [SEP] {doc[TEXT_FIELD]}" for doc in docs]

    print(f"Found {len(texts)} documents to classify...")

    predictions, confidences = classify_texts(texts)

    for doc, pred, conf in zip(docs, predictions, confidences):
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {PREDICTION_FIELD: pred, CONFIDENCE_FIELD: round(conf, 4)}}
        )

    end_total = time.time()
    print(f"Updated {len(docs)} documents.")
    print(f"Total time for prediction and update: {end_total - start_total:.2f} seconds")


if __name__ == "__main__":
    predict_and_update_mongodb()
