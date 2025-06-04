from pymongo import MongoClient
import pandas as pd

def export_jobs_to_csv(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    output_file: str = "exported_jobs.csv"
):
    """
    Export job data from MongoDB to CSV.

    Fields exported:
    - sector_name (one row per sector)
    - title
    - combined_description_text
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    query = {"job_detail_for_markup_url.data": {"$exists": True}}
    data = []

    for doc in collection.find(query):
        try:
            view_details = doc["job_detail_for_markup_url"]["data"]["view_job_details"]
            title = view_details.get("title", "")
            category = doc.get("category", "")
            description = doc.get("combined_description_text", "")
            sectors = view_details.get("sectors", [])

            # Handle multiple sectors per job
            data.append({
                    # "sector_name": [s["sector"]["name"] for s in sectors if "sector" in s and "name" in s["sector"]],
                    "title": category,
                    "combined_description_text": description
                })
        except Exception as e:
            print(f"Skipping document due to error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"CSV created: {output_file}")

def main():
    mongo_uri = "mongodb://localhost:27017/"  # Update if needed
    db_name = "telegram"  # Replace with your DB name
    collection_name = "messages"  # Replace with your collection name
    output_file = "afriworket_output.csv"

    export_jobs_to_csv(mongo_uri, db_name, collection_name, output_file)

if __name__ == "__main__":
    main()