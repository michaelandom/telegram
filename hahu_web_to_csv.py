
from pymongo import MongoClient
import pandas as pd
import re
import html

class HahuToCsv:
    def __init__(self, mongo_uri, db_name, collection_name, output_file="hahu_output.csv"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.output_file = output_file

    def export_jobs_to_csv(self):
        query = {}
        data = []

        for doc in self.collection.find(query):
            try:
                category = doc.get("category", "")
                description = doc.get("description", "")
                description = self.clean_description(description)
                data.append({
                    "title": category,
                    "combined_description_text": description
                })
            except Exception as e:
                print(f"Skipping document due to error: {e}")

        df = pd.DataFrame(data)
        unique_df = df.drop_duplicates(subset=['combined_description_text'], keep='first')
        unique_df.to_csv(self.output_file, index=False)
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