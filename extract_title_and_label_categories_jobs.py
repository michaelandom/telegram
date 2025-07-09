
from pymongo import MongoClient
from bs4 import BeautifulSoup
import re
import json


class ExtractTitleAndLabelCategoryAfriwork:

    def __init__(self, mongo_uri, db_name, collection_name, cs_categories_json="cs_job_categories.json"):
        self.path_cs_categories_json = cs_categories_json
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def clean_and_combine_freelance(self, text, description_html):
        text_cleaned = re.sub(r'\\n', '\n', text)
        text_cleaned = re.sub(r'[*_✅]', '', text_cleaned)
        text_cleaned = re.sub(r'_+', '', text_cleaned)
        text_cleaned = re.sub(
            r'Description:.*?\[view details.*?\]', '', text_cleaned, flags=re.IGNORECASE | re.DOTALL)
        text_cleaned = re.sub(r'From:.*?(?=\n|$)', '', text_cleaned)

        soup = BeautifulSoup(description_html, "html.parser")
        description_text = soup.get_text(separator='\n')
        description_text = re.sub(r'\s+\n', '\n', description_text)
        description_text = re.sub(r'\n\s+', '\n', description_text)
        description_text = re.sub(r'\n{2,}', '\n\n', description_text)

        combined = f"{text_cleaned.strip()}\n\nDescription:\n{description_text.strip()}"
        return [combined, description_text.strip()]

    def clean_and_combine_geezjob(self, text):
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'[*_✅]', '', text)
        text = text.replace('#', '')
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # def extract_job_title(self, text):
    #     # match = re.search(r'Job Title:\s*\*\*(.*?)\*\*', text)
    #     match = re.search(
    #         r'Job Title:\s*(\*\*)?(?P<title>.+?)(\*\*)?[\n\r]', text)
    #     if match:
    #         return match.group("title").strip()
    #     return None
    def extract_job_title(self, text):
        pattern_with_label = r'Job Title:\s*(\*\*)?(?P<title>.+?)(\*\*)?(\r?\n|$)'
        match = re.search(pattern_with_label, text, re.IGNORECASE)

        if match:
            return match.group("title").strip()
        text = remove_expired_line(text)
        position_match = re.search(
            r'\*\*Position:\*\*\s*(.+)', text, re.IGNORECASE)
        if position_match:
            return position_match.group(1).strip()

        position_match_alt = re.search(
            r'Position:\s*(.+)', text, re.IGNORECASE)
        if position_match_alt:
            return position_match_alt.group(1).strip()
        pattern_bold_title = r'^\s*(\*\*|\*)(?P<title>.+?)(\*\*|\*)\s*$'
        lines = text.strip().splitlines()
        if lines:
            first_line = lines[0]
            match_bold = re.search(pattern_bold_title, first_line)
            if match_bold:
                return match_bold.group("title").strip()

            first_line = lines[0].replace(":", "")
            return first_line.strip()
        return None

    def load_categories(self):
        with open(self.path_cs_categories_json, "r") as f:
            return json.load(f)

    def categorize_job_title(self, title: str, text: str, cs_categorize):
        def contains_whole_word(text, word):
            pattern = r'\b{}\b'.format(re.escape(word))
            return re.search(pattern, text, re.IGNORECASE) is not None
        title_lower = title.lower()
        for category, subcategories in cs_categorize.items():
            for sub in subcategories:
                if contains_whole_word(title_lower, sub):
                    return category
        text_lower = text.lower()
        for category, subcategories in cs_categorize.items():
            for sub in subcategories:
                if contains_whole_word(text_lower, sub):
                    return category
        return "other"

    def run(self, channel_username="freelance_ethio"):
        if channel_username == "freelance_ethio":
            self.freelance_ethio()
        elif channel_username == "geezjob":
            self.geezjob()
        elif channel_username == "hahujobs":
            self.hahujobs()
        elif channel_username == "linkedin_jobs":
            self.linkedinJobs()

        print("Combined job descriptions saved to combined.txt")

    def freelance_ethio(self):
        query = {"channel_username": {"$eq": "freelance_ethio"},
                 "$or": [
            {"category": {"$eq": ""}},
            {"category": {"$exists": False}}
        ]
        }
        documents = self.collection.find(query)
        cs_categorize = self.load_categories()
        for doc in documents:
            text = doc.get("text", "")
            title = self.extract_job_title(text)
            category = None
            if title:
                category = self.categorize_job_title(
                    title, text, cs_categorize)
            description = doc.get("job_detail_for_markup_url", {}).get(
                "data", {}).get("view_job_details", {}).get("description", "")
            if text and description:
                combined, update_description_text = self.clean_and_combine_freelance(
                    text, description)
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"title": title, "category": category, "combined_description_text": combined, "update_description_text": update_description_text}})
            else:
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"title": title, "category": category, }})

    def geezjob(self):
        query = {"channel_username": {"$eq": "geezjob"},
                 "$or": [
            {"category": {"$eq": ""}},
            {"category": {"$exists": False}}
        ]
        }
        documents = self.collection.find(query)
        cs_categorize = self.load_categories()
        for doc in documents:
            text = doc.get("text", "")
            title = self.extract_job_title(text)
            category = None
            if title:
                category = self.categorize_job_title(
                    title, text, cs_categorize)
            if text:
                combined = self.clean_and_combine_geezjob(text)
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"title": title, "category": category, "combined_description_text": combined}})
            else:
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"title": title, "category": category, }})

    def hahujobs(self):
        query = {"channel_username": {"$eq": "hahujobs"},
                 "$or": [
            {"category": {"$eq": ""}},
            {"category": {"$exists": False}}
        ]
        }
        documents = self.collection.find(query)
        cs_categorize = self.load_categories()
        for doc in documents:
            text = doc.get("text", "")
            title = self.extract_job_title(text)
            category = None
            if title:
                category = self.categorize_job_title(
                    title, text, cs_categorize)
            if text:
                text = remove_expired_line(text)
                combined = self.clean_and_combine_geezjob(text)
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"title": title, "category": category, "combined_description_text": combined}})
            else:
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"title": title, "category": category, }})

    def linkedinJobs(self):
        query = {
            "$or": [
                {"category": {"$eq": ""}},
                {"category": {"$exists": False}}
            ]
        }

        documents = self.collection.find(query)
        cs_categorize = self.load_categories()
        for doc in documents:
            text = doc.get("job_description", "")
            title = doc.get("job_title", "")
            category = None
            if title:
                category = self.categorize_job_title(
                    title, text, cs_categorize)
            if text:
                text = remove_text_line(text)
            self.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"title": title, "category": category, "combined_description_text":  text}})


def remove_expired_line(text):
    text = text.replace('- - EXPIRED - -', '')
    text = text.replace('⚠️⚠️⚠️ THIS JOB IS EXPIRED ⚠️⚠️⚠️', '')
    return text

def remove_text_line(text):
    text = text.replace('Show more', '')
    text = text.replace('Show less', '')
    return text
