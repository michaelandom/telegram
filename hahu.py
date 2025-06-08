import requests
import json
from pymongo import MongoClient, UpdateOne
import random
import time
import requests


def fetch_jobs(limit=1000, offset=0):
    url = "https://graph.aggregator.hahu.jobs/v1/graphql"
    query = f"""
    query MyQuery {{
      jobs(limit: {limit},offset: {offset}) {{
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


def save_job_reply_markup_with_details(mongo_uri, db_name, collection_name, details):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    operations = []
    for item in details:
        filter_query = {"id": item["id"]}
        update_query = {"$set": item}
        operations.append(UpdateOne(filter_query, update_query, upsert=True))

    if operations:
        collection.bulk_write(operations)
    client.close()


if __name__ == "__main__":
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "telegram"
    collection_name = "huhu"
    has_data = True
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.create_index("id", unique=True)
    offset = 0
    limit = 1000
    while has_data:
        try:
            print("Fetching job details...")
            result = fetch_jobs(limit, offset)
            if result:
                print("Updating document in MongoDB...")
                save_job_reply_markup_with_details(
                    mongo_uri, db_name, collection_name, result
                )
                offset += limit
            else:
                has_data = False
        except Exception as e:
            print(e)
            has_data = False
        sleep_duration = random.uniform(0, 1)
        print(f"⏱ Sleeping for {sleep_duration:.2f} seconds...\n")
        time.sleep(sleep_duration)
    print("Document updated successfully.")
