import time
from pymongo import MongoClient
from tqdm import tqdm
import joblib
import os

MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"  
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "telegram"
COLLECTION_NAME = "jobs"
TEXT_FIELD = "description"
PREDICTION_FIELD = "predicted_job_title_xgb_model"
CONFIDENCE_FIELD = "prediction_confidence_xgb_model"
BATCH_SIZE = 1000

# Load model, vectorizer, and label encoder
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)  # Usually sklearn.preprocessing.LabelEncoder

def classify_texts(texts):
    """Classifies a list of texts and returns predictions and confidences."""
    predictions = []
    confidences = []

    start_time = time.time()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Classifying"):
        batch = texts[i:i + BATCH_SIZE]
        X = vectorizer.transform(batch)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        confs = probs.max(axis=1)

        decoded_preds = label_encoder.inverse_transform(preds)
        predictions.extend(decoded_preds)
        confidences.extend(confs)

    end_time = time.time()
    print(f"Classification time for {len(texts)} documents: {end_time - start_time:.2f} seconds")

    return predictions, confidences


def predict_and_update_mongodb():
    """Reads data from MongoDB, predicts labels, and updates documents."""
    start_total = time.time()

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Fetch documents with non-empty text field
    cursor = collection.find({TEXT_FIELD: {"$exists": True, "$ne": ""}})
    docs = list(cursor)

    if not docs:
        print("No documents found with valid text.")
        return

    texts = [f"{doc["title"]}  {doc[TEXT_FIELD]}" for doc in docs]

    print(f"Found {len(texts)} documents to classify...")

    predictions, confidences = classify_texts(texts)

    # Update predictions back to MongoDB
    for doc, pred, conf in zip(docs, predictions, confidences):
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                PREDICTION_FIELD: pred,
                CONFIDENCE_FIELD: round(float(conf), 4)
            }}
        )

    end_total = time.time()
    print(f"Updated {len(docs)} documents.")
    print(f"Total time for prediction and update: {end_total - start_total:.2f} seconds")


if __name__ == "__main__":
    predict_and_update_mongodb()
