
from pymongo import MongoClient
import pandas as pd
import re
import html

class AfriworkToCsv:
    def __init__(self, mongo_uri, db_name, collection_name, output_file="telegram_output.csv"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.output_file = output_file

    def export_jobs_to_csv(self, get_all_jobs= False):
        print(f"get_all_jobs {get_all_jobs}")
        query = {"job_detail_for_markup_url.data": {"$exists": True}}
        if get_all_jobs:
            query = {"title": {"$ne": ""},"category":{"$ne": ""}}
        data = []

        for doc in self.collection.find(query):
            try:
                category = doc.get("category", "")
                description = doc.get("combined_description_text", "") or doc.get("text", "")
                description = self.clean_description(description)
                data.append({
                    "title": category,
                    "combined_description_text": description
                })
            except Exception as e:
                print(f"Skipping document due to error: {e}")

        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        print(f"CSV created: {self.output_file}")

    def clean_description(self,description):
        if description:
            description = html.unescape(description)
            quote_chars = [ '"', "'", '“', '”', '‘', '’', '«', '»', '`', '´', '❝', '❞']
            for q in quote_chars:
                description = description.replace(q, '')
            description = re.sub(r'<.*?>', '', description)  
            description = description.strip()
            description = f"{description}"
        return description