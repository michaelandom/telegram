import pymongo
from datetime import datetime, timedelta
import re
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
import time
def get_country_from_location(location_str: str) -> str:
    geolocator = Nominatim(user_agent="location_parser")

    try:
        location = geolocator.geocode(location_str, language='en')
        if location and location.raw.get("display_name"):
            address = location.raw.get("display_name")
            return address.split(",")[-1].strip()
        else:
            return "Unknown"
    except (GeocoderUnavailable, GeocoderTimedOut):
        return "Unknown"


def calculate_posted_date(time_posted: str, created_at: str) -> str:
    created_at_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    match = re.match(r"(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago", time_posted.lower())
    if not match:
        raise ValueError("Invalid time_posted format")
    number = int(match.group(1))
    unit = match.group(2)

    delta_args = {
        "second": timedelta(seconds=number),
        "minute": timedelta(minutes=number),
        "hour": timedelta(hours=number),
        "day": timedelta(days=number),
        "week": timedelta(weeks=number),
        "month": timedelta(days=30 * number),  
        "year": timedelta(days=365 * number)      
        }

    posted_date = created_at_dt - delta_args[unit]
    return posted_date.isoformat()

class JobDataSet:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        if db_name != 'telegram':
            self.db_job = self.client["telegram"]
        else:
            self.db_job = self.db
        self.collection_jobs =self.db_job["jobs"]

        
    def move_job_from_linkedin(self):
        linkedin_jobs = self.collection.find({"job_moved": {"$exists": False},"combined_description_text":{"$exists": True, "$ne": ""}})
        
        existing_job_ids = set(doc["job_id"] for doc in self.collection_jobs.find({}, {"job_id": 1}))
        insert_operations = []
        update_operations = []
        count = 0
        batch_size = 1000

        batch_start_time = time.time()  # Start timing batch creation

        for linkedin_job in linkedin_jobs:
            job_id = linkedin_job.get("job_id")
            if not job_id:
                continue

            if job_id in existing_job_ids:
                update_operations.append(pymongo.UpdateOne(
                    {"_id": linkedin_job["_id"]},
                    {"$set": {"job_moved": True}}
                ))
                continue

            job = {
                "job_id": job_id,
                "source": "linkedin",
                "date": calculate_posted_date(linkedin_job.get("time_posted"), str(linkedin_job.get("created_at")))
                    if linkedin_job.get("time_posted") and linkedin_job.get("created_at") else None,
                "description": linkedin_job.get("combined_description_text"),
                "title": linkedin_job.get("job_title"),
                "category": linkedin_job.get("category"),
                "data_from": "linkedin",
                "extracted_skills": None,
                "max_years_of_experience": None,
                "maximum_education_level": None,
                "minimum_education_level": None,
                "type": linkedin_job.get("employment_type"),
                "company_name": linkedin_job.get("company_name"),
                "seniority_level": linkedin_job.get("seniority_level"),
                "job_location": linkedin_job.get("job_location") or None,
                "country": None,
                "new_job": True,
            }

            insert_operations.append(pymongo.InsertOne(job))
            update_operations.append(pymongo.UpdateOne(
                {"_id": linkedin_job["_id"]},
                {"$set": {"job_moved": True}}
            ))

            count += 1

            if count % batch_size == 0:
                batch_end_time = time.time()
                elapsed_batch_time = batch_end_time - batch_start_time
                print(f"Batch of {batch_size} collected in {elapsed_batch_time:.2f} seconds.")

                # Bulk write
                if insert_operations:
                    self.collection_jobs.bulk_write(insert_operations)
                    insert_operations.clear()
                if update_operations:
                    self.collection.bulk_write(update_operations)
                    update_operations.clear()

                batch_start_time = time.time()  # Restart timing for the next batch

        if insert_operations or update_operations:
            print(f"Final batch (less than {batch_size}) collected in {time.time() - batch_start_time:.2f} seconds.")
            if insert_operations:
                self.collection_jobs.bulk_write(insert_operations)
            if update_operations:
                self.collection.bulk_write(update_operations)

    def move_job_from_message(self):
        messages_jobs = self.collection.find({"job_moved": {"$exists": False},"combined_description_text":{"$exists": True, "$ne": ""}})
        existing_job_ids = set(doc["job_id"] for doc in self.collection_jobs.find({}, {"job_id": 1}))
        insert_operations = []
        update_operations = []
        count = 0
        batch_size = 10
        batch_start_time = time.time()  
        for messages_job in messages_jobs:
            job_id = f"{messages_job.get("channel_id")}_{messages_job.get("message_id")}"
            if not job_id:
                continue
            if job_id in existing_job_ids:
                update_operations.append(pymongo.UpdateOne(
                    {"_id": messages_job["_id"]},
                    {"$set": {"job_moved": True}}
                ))
                continue
            job = {
                "job_id": job_id,
                "source": messages_job.get("channel_username"),
                "date": messages_job.get("date"),
                "description": messages_job.get("combined_description_text"),
                "title": messages_job.get("title"),
                "category": messages_job.get("category"),
                "predicted_job_title_roberta_model": messages_job.get("predicted_job_title"),
                "prediction_confidence_roberta_model": messages_job.get("prediction_confidence"),
                "data_from": "telegram",
                "extracted_skills": messages_job.get("extracted_skills"),
                "max_years_of_experience": None,
                "maximum_education_level": None,
                "minimum_education_level": None,
                "type": None,
                "company_name": None,
                "seniority_level": None,
                "job_location": "Addis Ababa",
                "country": "Ethiopia",
                "new_job": True,
            }

            insert_operations.append(pymongo.InsertOne(job))
            update_operations.append(pymongo.UpdateOne(
                {"_id": messages_job["_id"]},
                {"$set": {"job_moved": True}}
            ))

            count += 1

            if count % batch_size == 0:
                batch_end_time = time.time()
                elapsed_batch_time = batch_end_time - batch_start_time
                print(f"Batch of {batch_size} collected in {elapsed_batch_time:.2f} seconds.")

                # Bulk write
                if insert_operations:
                    self.collection_jobs.bulk_write(insert_operations)
                    insert_operations.clear()
                if update_operations:
                    self.collection.bulk_write(update_operations)
                    update_operations.clear()

                batch_start_time = time.time()  # Restart timing for the next batch

        if insert_operations or update_operations:
            print(f"Final batch (less than {batch_size}) collected in {time.time() - batch_start_time:.2f} seconds.")
            if insert_operations:
                self.collection_jobs.bulk_write(insert_operations)
            if update_operations:
                self.collection.bulk_write(update_operations)

    def move_job_from_hahu_web(self):
        messages_jobs = self.collection.find({"job_moved": {"$exists": False},"combined_description_text":{"$exists": True, "$ne": ""}})
        
        existing_job_ids = set(doc["job_id"] for doc in self.collection_jobs.find({}, {"job_id": 1}))
        insert_operations = []
        update_operations = []
        count = 0
        batch_size = 1000

        batch_start_time = time.time()  
        for messages_job in messages_jobs:
            job_id = f"{messages_job.get("id")}"
            if not job_id:
                continue

            if job_id in existing_job_ids:
                update_operations.append(pymongo.UpdateOne(
                    {"_id": messages_job["_id"]},
                    {"$set": {"job_moved": True}}
                ))
                continue

            job = {
                "job_id": job_id,
                "source": messages_job.get("source"),
                "date": messages_job.get("created_at"),
                "description": messages_job.get("combined_description_text"),
                "title": messages_job.get("title"),
                "category": messages_job.get("category"),
                "predicted_job_title_roberta_model": messages_job.get("predicted_job_title"),
                "prediction_confidence_roberta_model": messages_job.get("prediction_confidence"),
                "data_from": "hahu",
                "extracted_skills": messages_job.get("extracted_skills"),
                "max_years_of_experience":  messages_job.get("max_years_of_experience"),
                "maximum_education_level": messages_job.get("maximum_education_level"),
                "minimum_education_level":  messages_job.get("minimum_education_level"),
                "type":  messages_job.get("type"),
                "company_name": None,
                "seniority_level": None,
                "job_location": "Addis Ababa",
                "country": "Ethiopia",
                "new_job": True
            }

            insert_operations.append(pymongo.InsertOne(job))
            update_operations.append(pymongo.UpdateOne(
                {"_id": messages_job["_id"]},
                {"$set": {"job_moved": True}}
            ))

            count += 1

            if count % batch_size == 0:
                batch_end_time = time.time()
                elapsed_batch_time = batch_end_time - batch_start_time
                print(f"Batch of {batch_size} collected in {elapsed_batch_time:.2f} seconds.")

                if insert_operations:
                    self.collection_jobs.bulk_write(insert_operations)
                    insert_operations.clear()
                if update_operations:
                    self.collection.bulk_write(update_operations)
                    update_operations.clear()

                batch_start_time = time.time()  

        if insert_operations or update_operations:
            print(f"Final batch (less than {batch_size}) collected in {time.time() - batch_start_time:.2f} seconds.")
            if insert_operations:
                self.collection_jobs.bulk_write(insert_operations)
            if update_operations:
                self.collection.bulk_write(update_operations)

