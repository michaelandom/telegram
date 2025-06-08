import requests
import json
from pymongo import MongoClient, UpdateOne
import random
import time
import requests


class HahuWebSiteScraper:
    def __init__(self, mongo_uri, db_name, collection_name, limit=1000, offset=0, url="https://graph.aggregator.hahu.jobs/v1/graphql"):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.limit = limit
        self.offset = offset
        self.url = url
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        print(f"collection_name {self.collection}")

    def fetch_jobs(self):
        url = self.url
        query = f"""
        query MyQuery {{
        jobs(limit: {self.limit},offset: {self.offset}, order_by: {{created_at: desc}}) {{
            id
            title
            telegram_id
            deadline
            description
            description_type
            coordinates
            source
            company {{
            name
            }}
            created_at
            url
            summary
            salary
            position {{
            name
            description
            }}
            gender_priority
            general_skill_level
            job_skills {{
            proficency
            }}
            is_online
            max_years_of_experience
            maximum_education_level {{
            name
            }}
            minimum_education_level {{
            name
            }}
            number_of_applicants
            organization_maybe
            payment_period
            posted_on
            print_classification_level
            priority
            years_of_experience
            web_view_count
            view_count
            total_web_view_count
            total_view_count
            type
        }}
        }}
        """

        payload = {
            "query": query,
            "variables": {},
            "operationName": "MyQuery"
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                print("GraphQL Errors:", data["errors"])
                return None
            else:
                return data["data"]["jobs"]
        else:
            raise Exception(
                f"GraphQL query failed with status code {response.status_code}: {response.text}")

    def save_job_reply_markup_with_details(self, details):
        operations = []
        for item in details:
            filter_query = {"id": item["id"]}
            update_query = {"$set": item}
            operations.append(
                UpdateOne(filter_query, update_query, upsert=True))

        if operations:
            self.collection.bulk_write(operations)

    def run(self,loop=5):
        has_data = True
        self.collection.create_index("id", unique=True)
        offset = self.offset
        limit = self.limit
        while has_data:
            try:
                print("Fetching job details...")
                result = self.fetch_jobs()
                if result:
                    print("Updating document in MongoDB...")
                    self.save_job_reply_markup_with_details(
                        result
                    )
                    offset += limit
                else:
                    has_data = False
                has_data = (offset / limit) <=loop 
            except Exception as e:
                print(e)
                has_data = False
            sleep_duration = random.uniform(0, 1)
            print(f"⏱ Sleeping for {sleep_duration:.2f} seconds...\n")
            time.sleep(sleep_duration)
        self.client.close()
        print("Document updated successfully.")
