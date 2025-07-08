from pymongo import MongoClient
from bs4 import BeautifulSoup
import re
import json


class ExtractAndAddCategory:
    def __init__(self, mongo_uri, db_name, collection_name, cs_categories_json="cs_job_categories.json"):
        self.path_cs_categories_json = cs_categories_json
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def load_categories(self):
        with open(self.path_cs_categories_json, "r") as f:
            return json.load(f)

    def categorize_job_title(self, title: str, description: str, summary: str, cs_categorize):
        def contains_whole_word(text, word):
            pattern = r'\b{}\b'.format(re.escape(word))
            return re.search(pattern, text, re.IGNORECASE) is not None

        title_lower = title.lower()
        for category, subcategories in cs_categorize.items():
            for sub in subcategories:
                if contains_whole_word(title_lower, sub):
                    return category

        description_lower = description.lower()
        summary_lower = summary.lower()
        for category, subcategories in cs_categorize.items():
            for sub in subcategories:
                if contains_whole_word(description_lower, sub) or contains_whole_word(summary_lower, sub):
                    return category

        return "other"

    def run(self):
        query = {
            "$or": [
                {"category": {"$eq": ""}},
                {"category": {"$exists": False}}
            ]
        }
        documents = self.collection.find(query)
        cs_categorize = self.load_categories()
        for doc in documents:
            title = doc.get("title", "")
            description = doc.get("description", "") or ""
            summary = doc.get("summary", "") or ""
            category = None
            if title:
                category = self.categorize_job_title(
                    title, description, summary, cs_categorize)
            self.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"title": title, "category": category, }})
        print("Combined job title saved")
