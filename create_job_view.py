from pymongo import MongoClient


class VotingJobLabeler:
    def __init__(self,
                 mongo_uri="mongodb://localhost:27017",
                 database_name="telegram",
                 source_collection="jobs",
                 target_collection="job_with_voting",
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

    def run(self):
        pipeline = self.build_pipeline()
        result = list(self.collection.aggregate(pipeline))
        print(f"Aggregation pipeline executed. {len(result)} documents processed (if aggregation returns output).")


if __name__ == "__main__":
    labeler = VotingJobLabeler()
    labeler.run()
