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
        { "job_detail_for_markup_url": { "$eq": None } , 
         "reply_markup": { "$regex": "startapp=" }
         },
#         {
#           "job_reply_markup_id": {
#               "$in": ["eaf3a944-3dbd-49bb-abda-961f86f9473e",
# "2fb3ccec-15e3-4ba6-9e36-7d98ea5b606e",
# "93069026-cf45-475c-b6b2-6da002ae9ccd",
# "face8c55-fba3-4e72-aac0-e2312ed9770e",
# "b3e2b95b-9eba-4bd8-ba86-8cc394e3dc8b",
# "23bc3296-949a-4882-a7a8-226b1e81d06b",
# "3b610df4-b9b0-4167-8649-fdee5ac14f1e",
# "bb719681-f277-4a62-acf7-9417b58878aa",
# "e74f7542-aa6d-44f5-8a25-89dee12ba70f",
# "57bc0d30-4f55-47fa-82a2-3ad9b724b8ed",
# "b28a5b63-5acf-4674-8c3d-ce28156115e5",
# "8f5dabcb-513a-4f53-a033-56f2fa3f93ea",
# "395ebdbd-d988-43f3-8171-3891e7640546",
# "9f7cb8da-24d0-42bd-a5a8-04303a600b48",
# "69aa3b09-dc3d-4024-bb9a-dcf568f86bbb",
# "a33229e6-be29-4f82-8473-d729c0cc05c5",
# "824a5200-fb4d-4b7f-8542-06e59e8f4bf8",
# "c555bf67-f72c-404f-9d44-0e34fc7686b0",
# "9bc76738-3e22-4855-85d8-718ebb54ae4a",
# "48e4b357-bf6a-4bde-8561-c01d03f832f3",
# "56182f62-5bd3-48e0-88e6-ffc738a42445",
# "a5d05e8a-bce4-49c5-9142-be73f61ab01c",
# "15359d76-7f85-4487-bca5-3664d037516a",
# "279452cd-74ee-4ca4-a3cd-1abdfb273eb0",
# "2bcca56b-c1c7-43e8-b160-904df31809c3",
# "acd70423-22d4-4d6a-a2fe-bd8c8dd455c1",
# "6a6e8ae7-d7c6-4c3a-97f2-af88449aac11",
# "84535987-77c1-45ad-b7de-fa0e3dc0af7b",
# "8f7a6b9e-b9dc-4047-b11e-ccde794a4b6d",
# "d17f0d58-2f29-4ed8-9184-b4b6f2a93aa6",
# "af95a867-531c-4c1c-a467-6d8c1585bd33"]
#           }  
#         },
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
