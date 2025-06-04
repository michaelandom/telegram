from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import pandas as pd
import tf_keras as keras
mongo_uri = "mongodb://localhost:27017/"  # Update if needed
db_name = "telegram"  # Replace with your DB name
collection_name = "huhu"  # Replace with your collection name
client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# Fetch unique titles
titles = collection.distinct("title")
print(f"Fetched {len(titles)} unique titles.")

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(titles, convert_to_tensor=False)

# Cluster embeddings
clustering_model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.8,
    affinity='cosine',
    linkage='average'
)
labels = clustering_model.fit_predict(embeddings)

# Group titles by cluster
clustered_titles = defaultdict(list)
for label, title in zip(labels, titles):
    clustered_titles[label].append(title)

df = pd.DataFrame(clustered_titles)
df.to_csv("output_file.csv", index=False)

print("Categorized titles stored successfully.")

