from pymongo import MongoClient
from bs4 import BeautifulSoup
import re
import json
def clean_and_combine(text, description_html):
    text_cleaned = re.sub(r'\\n', '\n', text)
    text_cleaned = re.sub(r'[*_✅]', '', text_cleaned)
    text_cleaned = re.sub(r'_+', '', text_cleaned)
    text_cleaned = re.sub(r'Description:.*?\[view details.*?\]', '', text_cleaned, flags=re.IGNORECASE | re.DOTALL)
    text_cleaned = re.sub(r'From:.*?(?=\n|$)', '', text_cleaned)

    soup = BeautifulSoup(description_html, "html.parser")
    description_text = soup.get_text(separator='\n')
    description_text = re.sub(r'\s+\n', '\n', description_text)
    description_text = re.sub(r'\n\s+', '\n', description_text)
    description_text = re.sub(r'\n{2,}', '\n\n', description_text)

    combined = f"{text_cleaned.strip()}\n\nDescription:\n{description_text.strip()}"
    return [combined, description_text.strip()]
def extract_job_title(text):
    """
    Extracts the job title from a block of text.
    Looks for the pattern 'Job Title: **XYZ**'
    """
    # match = re.search(r'Job Title:\s*\*\*(.*?)\*\*', text)
    match = re.search(r'Job Title:\s*(\*\*)?(?P<title>.+?)(\*\*)?[\n\r]', text)
    if match:
        return match.group("title").strip()
    return None

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
    collection = db["messages"]  

    query = {"channel_username": {"$eq": "freelance_ethio"}}
    documents = collection.find(query)
    cs_categorize = load_categories()
    for doc in documents:
        text = doc.get("text", "")
        title = extract_job_title(text)
        category = None
        if title:
            category = categorize_job_title(title,cs_categorize)
        description = doc.get("job_detail_for_markup_url", {}).get("data", {}).get("view_job_details", {}).get("description", "")
        if text and description:
            combined, update_description_text = clean_and_combine(text, description)
            collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"title": title,"category":category, "combined_description_text": combined,"update_description_text": update_description_text}})
        else:
            collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"title": title,"category":category,}})
    print("Combined job descriptions saved to combined.txt")

if __name__ == "__main__":
    main()
