import os
import time
import joblib
from pymongo import MongoClient
from tqdm import tqdm


class XGBJobClassifier:
    def __init__(self,
                 model_path: str="models/best_model.pkl",
                 vectorizer_path: str="models/tfidf_vectorizer.pkl",
                 label_encoder_path: str="models/label_encoder.pkl",
                 mongo_uri: str="mongodb://localhost:27017",
                 db_name: str="telegram",
                 collection_name: str="jobs",
                 text_field: str = "description",
                 prediction_field: str = "predicted_job_title_xgb_model",
                 confidence_field: str = "prediction_confidence_xgb_model",
                 batch_size: int = 1000):

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(label_encoder_path)

        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.text_field = text_field
        self.prediction_field = prediction_field
        self.confidence_field = confidence_field
        self.batch_size = batch_size

    def classify_texts(self, texts: list[str]) -> tuple[list[str], list[float]]:
        predictions, confidences = [], []
        start_time = time.time()

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Classifying"):
            batch = texts[i:i + self.batch_size]
            X = self.vectorizer.transform(batch)
            preds = self.model.predict(X)
            probs = self.model.predict_proba(X)
            confs = probs.max(axis=1)

            decoded_preds = self.label_encoder.inverse_transform(preds)
            predictions.extend(decoded_preds)
            confidences.extend(confs)

        print(f"Classification time for {len(texts)} documents: {time.time() - start_time:.2f} seconds")
        return predictions, confidences

    def predict_and_update_mongodb(self):
        start_total = time.time()

        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[self.collection_name]

        query = {
            self.text_field: {"$exists": True, "$ne": ""},
            self.prediction_field: {"$exists": False}
        }
        docs = list(collection.find(query))

        if not docs:
            print("No documents found with valid text.")
            return

        texts = [f"{doc.get('title', '')} {doc[self.text_field]}" for doc in docs]
        print(f"Found {len(texts)} documents to classify...")

        predictions, confidences = self.classify_texts(texts)

        for doc, pred, conf in zip(docs, predictions, confidences):
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    self.prediction_field: pred,
                    self.confidence_field: round(float(conf), 4)
                }}
            )

        print(f"Updated {len(docs)} documents.")
        print(f"Total time: {time.time() - start_total:.2f} seconds")


if __name__ == "__main__":
    classifier = XGBJobClassifier()
    classifier.predict_and_update_mongodb()
