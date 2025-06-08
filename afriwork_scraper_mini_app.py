import requests
import json
from pymongo import MongoClient
import random
import time


class AfriworketMiniAppScraper:

    def __init__(self, mongo_uri, db_name, collection_name, token="", url="https://api.afriworket.com:9000/v1/graphql"):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.token = token
        self.url = url

    def get_token(self) -> bool:
        url = "https://api.afriworket.com:9010/mini-app/validate-request"
        payload = {"telegram_id": "7283293326"}

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PostmanRuntime/7.43.4",
            "Accept": "*/*",
            "x-bot-type":"APPLICANT",
            "x-telegram-init-data":"user=%7B%22id%22%3A7283293326%2C%22first_name%22%3A%22C%22%2C%22last_name%22%3A%22B%22%2C%22username%22%3A%22cb23459%22%2C%22language_code%22%3A%22en%22%2C%22photo_url%22%3A%22https%3A%5C%2F%5C%2Ft.me%5C%2Fi%5C%2Fuserpic%5C%2F320%5C%2FXCcppoP3L10G-9fk-XI9uLuwv-0YDJBaKnzegyDI3kFB7XkYU1zBmsUS5xv3bLy6.svg%22%7D&chat_instance=2912981003447109990&chat_type=private&start_param=1c212ad4-f3fa-4d42-9d85-0f66ac3a8ae0&auth_date=1749127640&signature=wj_9i6ORJ9HEc-3ZZrL_g52I9tPPIJAAf5cRLN7MVZztSHEP_0noFdZlXiskWsJOu2s3p8KZxLYQ4P0ESKH_Cg&hash=06726d72fd70213a358dc510e8b11d631a0b3183ea2abe2e1a3a20e7b3465ae6",
            "Cache-Control": "no-cache",
            "Host": "api.afriworket.com:9000",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }

        response = requests.post(url, headers=headers,
                                 data=json.dumps(payload))

        if response.status_code == 200 and response.json()["token"]:
            self.token = response.json()["token"] if response.json()[
                "token"] else self.token
            print("Token is set")
            return True
        else:
            raise Exception(
                f"Request failed ({response.status_code}): {response.text}")

    def get_job_details(self, job_id: str, share_id: str = None) -> dict:
        url = self.url
        payload = {
            "operationName": "viewDetails",
            "query": "query viewDetails($id: uuid!, $share_id: uuid) {\n  view_job_details(obj: {job_id: $id, share_id: $share_id}) {\n    id\n    title\n    approval_status\n    closed_at\n    job_type\n    job_site\n    location\n    created_at\n    entity {\n      name\n      type\n      jobs_aggregate {\n        aggregate {\n          count\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    sectors {\n      sector {\n        name\n        id\n        __typename\n      }\n      __typename\n    }\n    description\n    city {\n      id\n      name\n      en\n      country {\n        name\n        id\n        en\n        __typename\n      }\n      __typename\n    }\n    platform {\n      name\n      id\n      __typename\n    }\n    skill_requirements {\n      skill {\n        name\n        __typename\n        id\n      }\n      __typename\n    }\n    deadline\n    vacancy_count\n    gender_preference\n    compensation_amount_cents\n    job_education_level {\n      education_level\n      __typename\n    }\n    experience_level\n    compensation_type\n    compensation_currency\n    __typename\n  }\n}",
            "variables": {
                "id": job_id
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "PostmanRuntime/7.43.4",
            "Accept": "*/*",
            "Cache-Control": "no-cache",
            "Host": "api.afriworket.com:9000",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }

        response = requests.post(url, headers=headers,
                                 data=json.dumps(payload))

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Request failed ({response.status_code}): {response.text}")

    def get_job_reply_markup_ids(self):
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[self.collection_name]

        cursor = collection.find(
            {"job_detail_for_markup_url": {"$eq": None},
             "reply_markup": {"$regex": "startapp="}
             },
            {"job_reply_markup_id": 1, "_id": 1}
        )

        results = [{"_id": doc["_id"], "job_reply_markup_id": doc["job_reply_markup_id"]}
                   for doc in cursor]

        client.close()
        return results

    def update_job_reply_markup_with_details(self, doc_id, details):
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[self.collection_name]

        collection.update_one(
            {"_id": doc_id},
            {"$set": {"job_detail_for_markup_url": details}}
        )
        client.close()

    def run(self):
        is_token_set = self.get_token()
        if is_token_set:
            docs = self.get_job_reply_markup_ids()

            for doc in docs:
                job_reply_markup_id = doc["job_reply_markup_id"]
                job_reply_markup_id = job_reply_markup_id.replace(
                    "APPLICANT", "")

                print(
                    f"\nProcessing job_reply_markup_id: {job_reply_markup_id}")
                try:
                    print("Fetching job details...")
                    result = self.get_job_details(job_reply_markup_id)
                    print("Updating document in MongoDB... ")
                    self.update_job_reply_markup_with_details(
                        doc["_id"], result)
                except Exception as e:
                    print("Error processing job_reply_markup_id",
                          job_reply_markup_id, ":", e)
                sleep_duration = random.uniform(0, 1)
                print(f"⏱ Sleeping for {sleep_duration:.2f} seconds...\n")
                time.sleep(sleep_duration)
            print("Document updated successfully.")

        else:
            print("Error issue on Token")
