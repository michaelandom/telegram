from pymongo import MongoClient, UpdateOne


class VotingJobLabeler:
    def __init__(self,
                 mongo_uri="mongodb://localhost:27017",
                 database_name="telegram",
                 source_collection="jobs",
                 target_collection="jobs",
                 override_xgb_labels=None):

        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.source_collection_name = source_collection
        self.target_collection_name = target_collection

        self.override_xgb_labels = override_xgb_labels or [
            "other",
            "Product / Project Management (Technical)"
        ]

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.source_collection_name]

    def build_pipeline(self):
        return [
            {
                "$match": {
                    "predicted_job_title_xgb_model": {"$exists": True},
                    "predicted_job_title_bert_model": {"$exists": True}
                }
            },
            {
                "$addFields": {
                    "final_prediction_label": {
                        "$cond": {
                            "if": {
                                "$in": ["$predicted_job_title_xgb_model", self.override_xgb_labels]
                            },
                            "then": "$predicted_job_title_xgb_model",
                            "else": "$predicted_job_title_bert_model"
                        }
                    }
                }
            },
            # Optional accuracy field
            # {
            #     "$addFields": {
            #         "is_prediction_wrong": {
            #             "$cond": {
            #                 "if": {"$eq": ["$category", "$final_prediction_label"]},
            #                 "then": 0,
            #                 "else": 1
            #             }
            #         }
            #     }
            # },
            {
                "$merge": {
                    "into": self.target_collection_name,
                    "whenMatched": "merge"
                }
            }
        ]
    
    def update_final_prediction_labels(self):
        xgb_categories = ["other", "Product / Project Management (Technical)"]
        bulk_updates = []

        cursor = self.collection.find({
            "approved": {"$exists": False},
            "predicted_job_title_xgb_model": {"$exists": True},
            "predicted_job_title_bert_model": {"$exists": True}
        })

        for doc in cursor:
            job_id = doc["_id"]
            xgb_prediction = doc["predicted_job_title_xgb_model"]
            bert_prediction = doc["predicted_job_title_bert_model"]

            final_label = (
                xgb_prediction if xgb_prediction in xgb_categories else bert_prediction
            )

            bulk_updates.append(
                UpdateOne(
                {"_id": job_id},
                {"$set": {"final_prediction_label": final_label}}
                )
            )

        if bulk_updates:
            result = self.collection.bulk_write(bulk_updates)
            print(f"Updated {result.modified_count} job(s).")
        else:
            print("No updates needed.")

    def run(self):
        # pipeline = self.build_pipeline()
        # result = list(self.collection.aggregate(pipeline))
        self.update_final_prediction_labels()
        print(f"Aggregation pipeline executed. documents processed (if aggregation returns output).")


if __name__ == "__main__":
    labeler = VotingJobLabeler()
    labeler.run()
