from pymongo import MongoClient
from datetime import datetime

class CategoryAggregator:
    def __init__(self, mongo_uri, db_name, source_collection):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.source_collection = self.db[source_collection]
        self.output_collection = self.db["categories"]
        self.output_collection.create_index("name", unique=True)

    def aggregate_by_category(self):
        pipeline = [
            {
                "$group": {
                    "_id": "$category",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            }
        ]
        return list(self.source_collection.aggregate(pipeline))
 
    def save_results(self, results):
        collection_name = "categories"
        for doc in results:
            self.output_collection.update_one(
                {"name": doc["_id"]},
                {"$set": {"total_count": doc["count"]}},
                upsert=True
            )

        return collection_name

    def run(self):
        results = self.aggregate_by_category()
        return self.save_results(results)
