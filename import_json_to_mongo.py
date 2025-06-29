import json
from pymongo import MongoClient

# === CONFIGURATION ===
JSON_FILE = "data.json"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "telegram"
COLLECTION_NAME = "jobs"

# === CONNECT TO MONGO ===
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# === READ AND INSERT JSON ===
with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    cleaned_data = [{k: v for k, v in doc.items() if k != "_id"} for doc in data]
    result = collection.insert_many(cleaned_data)
    print(f"Inserted {len(result.inserted_ids)} documents.")
else:
    data.pop("_id", None)  # Remove _id if present
    result = collection.insert_one(data)
    print(f"Inserted 1 document with ID: {result.inserted_id}")
