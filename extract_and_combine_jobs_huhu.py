from pymongo import MongoClient
from bs4 import BeautifulSoup
import re
import json

def load_categories(path = "cs_job_categories.json"):
    with open(path, "r") as f:
        return json.load(f)
    

def categorize_job_title(title:str, cs_categorize):
    title_lower= title.lower()
    for category, subcategories in cs_categorize.items():
        for sub in subcategories:
            if sub.lower() in title_lower:
                return category
    return "other"

def main():
    client = MongoClient("mongodb://localhost:27017")  
    db = client["telegram"]  
    collection = db["huhu"]  

    query = {}
    documents = collection.find(query)
    cs_categorize = load_categories()
    for doc in documents:
        title = doc.get("title", "")
        category = None
        if title:
            category = categorize_job_title(title,cs_categorize)
        collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"title": title,"category":category,}})
    print("Combined job descriptions saved to combined.txt")

if __name__ == "__main__":
    main()
