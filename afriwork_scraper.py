import requests
import json
from pymongo import MongoClient
import random
import time


def get_job_details(job_id: str, share_id: str = None) -> dict:
    """
    Fetch job details from the Afriwork GraphQL API.

    Args:
        job_id (str): UUID of the job.
        share_id (str, optional): UUID of the shared job link, if any.

    Returns:
        dict: Parsed JSON response containing job details.
    """
    url = "https://api.afriworket.com:9000/v1/graphql"
    payload = {
        "operationName": "viewDetails",
        "query": "query viewDetails($id: uuid!, $share_id: uuid) {\n  view_job_details(obj: {job_id: $id, share_id: $share_id}) {\n    id\n    title\n    approval_status\n    closed_at\n    job_type\n    job_site\n    location\n    created_at\n    entity {\n      name\n      type\n      jobs_aggregate {\n        aggregate {\n          count\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    sectors {\n      sector {\n        name\n        id\n        __typename\n      }\n      __typename\n    }\n    description\n    city {\n      id\n      name\n      en\n      country {\n        name\n        id\n        en\n        __typename\n      }\n      __typename\n    }\n    platform {\n      name\n      id\n      __typename\n    }\n    skill_requirements {\n      skill {\n        name\n        __typename\n        id\n      }\n      __typename\n    }\n    deadline\n    vacancy_count\n    gender_preference\n    compensation_amount_cents\n    job_education_level {\n      education_level\n      __typename\n    }\n    experience_level\n    compensation_type\n    compensation_currency\n    __typename\n  }\n}",
        "variables": {
            "id": job_id
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3MjgzMjkzMzI2IiwiZnJlZWxhbmNlIjp7IngtaGFzdXJhLXRlbGVncmFtLWlkIjoiNzI4MzI5MzMyNiIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6Imluc2VydF90ZW1wb3JhcnlfdXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsiYW5vbnltb3VzIiwiaW5zZXJ0X3RlbXBvcmFyeV91c2VyIl19LCJpYXQiOjE3NDkwNDE5NzMsImV4cCI6MTc0OTEyODM3MywiYXVkIjoidGVtcG9yYXJ5In0.9053XhLRF2XBuNiBJAPH0CaZC3a-DPE9A3q8Fgj6wG0",
        "User-Agent": "PostmanRuntime/7.43.4",
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Host": "api.afriworket.com:9000",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Request failed ({response.status_code}): {response.text}")


def get_job_reply_markup_ids(mongo_uri="mongodb://localhost:27017/", db_name="telegram", collection_name="messages"):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

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


def update_job_reply_markup_with_details(mongo_uri, db_name, collection_name, doc_id, details):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    collection.update_one(
        {"_id": doc_id},
        {"$set": {"job_detail_for_markup_url": details}}
    )
    client.close()


if __name__ == "__main__":
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "telegram"
    collection_name = "messages"

    docs = get_job_reply_markup_ids(mongo_uri, db_name, collection_name)

    for doc in docs:
        job_reply_markup_id = doc["job_reply_markup_id"]
        job_reply_markup_id = job_reply_markup_id.replace("APPLICANT", "")

        print(f"\nProcessing job_reply_markup_id: {job_reply_markup_id}")
        try:
            print("Fetching job details...")
            result = get_job_details(job_reply_markup_id)
            print("Updating document in MongoDB... ")
            update_job_reply_markup_with_details(
                mongo_uri, db_name, collection_name, doc["_id"], result
            )
        except Exception as e:
            print("Error processing job_reply_markup_id",
                  job_reply_markup_id, ":", e)
        sleep_duration = random.uniform(0, 1)
        print(f"⏱ Sleeping for {sleep_duration:.2f} seconds...\n")
        time.sleep(sleep_duration)
    print("Document updated successfully.")
