from pymongo import MongoClient, UpdateOne
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# MongoDB config
MONGO_URI = "mongodb://localhost:27017"
DATABASE_NAME = "telegram"
COLLECTION_NAME = "jobs"

model_mapping = {}
xgb_categories = []


def update_view(collection):
    pipeline = [
        {
            "$match": {
                "predicted_job_title_xgb_model": {"$exists": True},
                "predicted_job_title_bert_model": {"$exists": True}
            }
        },
        {
            "$addFields": 
                {
                "final_prediction_label": {
                    "$cond": [
                        { "$eq": ["$approved", True] },
                        "$final_prediction_label", 
                        {
                            "$cond": [
                                {
                                    "$in": [
                                        "$predicted_job_title_xgb_model",
                                        [
                                            "other",
                                            "Product / Project Management (Technical)"
                                        ]
                                    ]
                                },
                                "$predicted_job_title_xgb_model",
                                "$predicted_job_title_bert_model"
                            ]
                        }
                    ]
                }
            },
        },
        #  {
        #     "$addFields": {
        #         "is_prediction_wrong": {
        #             "$cond": {
        #                 "if": {
        #                     "$eq": ["$category", "$final_prediction_label"]
        #                 },
        #                 "then": 0,
        #                 "else": 1
        #             }
        #         }
        #     }
        # },
        {
            "$merge": {
                "into": "job_with_voting",
                "whenMatched": "merge"
            }
        }
    ]
    return list(collection.aggregate(pipeline))


def update_final_prediction_labels(collection):
    xgb_categories = ["other", "Product / Project Management (Technical)"]
    bulk_updates = []

    cursor = collection.find({
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
        result = collection.bulk_write(bulk_updates)
        print(f"Updated {result.modified_count} job(s).")
    else:
        print("No updates needed.")

def per_category_accuracy_with_voting(db):
    collection = db["jobs"]
    pipeline = [
        {
            "$match": {
                "new_job": {
                    "$exists": True
                }
            }
        },
        {
            "$project": {
                "category": 1,
                "correct": {"$eq": ["$category", "$final_prediction_label"]}
            }
        },
        {
            "$group": {
                "_id": "$category",
                "total": {"$sum": 1},
                "correct_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$correct", True]}, 1, 0]
                    }
                }
            }
        },
        {
            "$project": {
                "category": "$_id",
                "accuracy": {
                    "$cond": [
                            {"$gt": ["$total", 0]},
                            {"$divide": ["$correct_count", "$total"]},
                            0
                    ]
                },
                "total_jobs": "$total",
                "_id": 0
            }
        },
        {"$sort": {"total_jobs": -1}}
    ]
    results = list(collection.aggregate(pipeline))
    total_correct = 0
    total_jobs = 0

    for entry in results:
        correct = entry["accuracy"] * entry["total_jobs"]
        total_correct += correct
        total_jobs += entry["total_jobs"]

    overall_accuracy = total_correct / total_jobs if total_jobs > 0 else 0
    print(
        f"\n Overall Accuracy (Voting): {overall_accuracy:.4f} ({total_correct:.0f}/{total_jobs})")

    return results


def per_category_accuracy(collection):
    pipeline = [
        {
            "$project": {
                "category": 1,
                "roberta_correct": {"$eq": ["$category", "$predicted_job_title_roberta_model"]},
                "xgb_correct": {"$eq": ["$category", "$predicted_job_title_xgb_model"]},
                "bert_correct": {"$eq": ["$category", "$predicted_job_title_bert_model"]}
            }
        },
        {
            "$group": {
                "_id": "$category",
                "total": {"$sum": 1},
                "roberta_correct_count": {"$sum": {"$cond": ["$roberta_correct", 1, 0]}},
                "xgb_correct_count": {"$sum": {"$cond": ["$xgb_correct", 1, 0]}},
                "bert_correct_count": {"$sum": {"$cond": ["$bert_correct", 1, 0]}}
            }
        },
        {
            "$project": {
                "category": "$_id",
                "roberta_accuracy": {"$divide": ["$roberta_correct_count", "$total"]},
                "xgb_accuracy": {"$divide": ["$xgb_correct_count", "$total"]},
                "bert_accuracy": {"$divide": ["$bert_correct_count", "$total"]},
                "total_jobs": "$total",
                "_id": 0
            }
        },
        {"$sort": {"total_jobs": -1}}
    ]
    results = list(collection.aggregate(pipeline))
    total_xgb_correct = 0
    total_bert_correct = 0
    total_RoBERTa_correct = 0
    total_jobs = 0

    for entry in results:
        model_mapping[entry["category"]] = "bert" if max(entry["xgb_accuracy"], entry["roberta_accuracy"]) < entry["bert_accuracy"] else "xgb" if max(
            entry["xgb_accuracy"], entry["roberta_accuracy"]) == entry["xgb_accuracy"] else "roberta"
        correct = entry["xgb_accuracy"] * entry["total_jobs"]
        total_xgb_correct += correct
        correct = entry["bert_accuracy"] * entry["total_jobs"]
        total_bert_correct += correct
        correct = entry["roberta_accuracy"] * entry["total_jobs"]
        total_RoBERTa_correct += correct
        total_jobs += entry["total_jobs"]
    xgb_categories = [category for category,
                      model in model_mapping.items() if model == 'xgb']
    overall_accuracy = total_xgb_correct / total_jobs if total_jobs > 0 else 0
    print(
        f"\n Overall Accuracy of xgb: {overall_accuracy:.4f} ({total_xgb_correct:.0f}/{total_jobs})")
    overall_accuracy = total_bert_correct / total_jobs if total_jobs > 0 else 0
    print(
        f"\n Overall Accuracy of bert: {overall_accuracy:.4f} ({total_bert_correct:.0f}/{total_jobs})")
    overall_accuracy = total_RoBERTa_correct / total_jobs if total_jobs > 0 else 0
    print(
        f"\n Overall Accuracy of RoBERTa: {overall_accuracy:.4f} ({total_RoBERTa_correct:.0f}/{total_jobs})")

    return results


def disagreement_analysis(collection):
    pipeline = [
        {
            "$match": {
                "$expr": {
                    "$or": [
                        {"$ne": ["$predicted_job_title_roberta_model",
                                 "$predicted_job_title_xgb_model"]},
                        {"$ne": ["$predicted_job_title_roberta_model",
                                 "$predicted_job_title_bert_model"]},
                        {"$ne": ["$predicted_job_title_xgb_model",
                                 "$predicted_job_title_bert_model"]}
                    ]
                }
            }
        },
        {
            "$project": {
                "category": 1,
                "roberta_correct": {"$eq": ["$category", "$predicted_job_title_roberta_model"]},
                "xgb_correct": {"$eq": ["$category", "$predicted_job_title_xgb_model"]},
                "bert_correct": {"$eq": ["$category", "$predicted_job_title_bert_model"]}
            }
        },
        {
            "$group": {
                "_id": "$category",
                "total_disagreements": {"$sum": 1},
                "roberta_wins": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    "$roberta_correct",
                                    {"$eq": ["$xgb_correct", False]},
                                    {"$eq": ["$bert_correct", False]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                },
                "xgb_wins": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    "$xgb_correct",
                                    {"$eq": ["$roberta_correct", False]},
                                    {"$eq": ["$bert_correct", False]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                },
                "bert_wins": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    "$bert_correct",
                                    {"$eq": ["$roberta_correct", False]},
                                    {"$eq": ["$xgb_correct", False]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                },
                "all_wrong": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$eq": ["$roberta_correct", False]},
                                    {"$eq": ["$xgb_correct", False]},
                                    {"$eq": ["$bert_correct", False]}
                                ]
                            },
                            1,
                            0
                        ]
                    }
                },
                "multiple_correct": {
                    "$sum": {
                        "$cond": [
                            {
                                "$gt": [
                                    {
                                        "$add": [
                                            {"$cond": [
                                                "$roberta_correct", 1, 0]},
                                            {"$cond": ["$xgb_correct", 1, 0]},
                                            {"$cond": ["$bert_correct", 1, 0]}
                                        ]
                                    },
                                    1
                                ]
                            },
                            1,
                            0
                        ]
                    }
                }
            }
        },
        {
            "$project": {
                "category": "$_id",
                "total_disagreements": 1,
                "roberta_wins": 1,
                "xgb_wins": 1,
                "bert_wins": 1,
                "all_wrong": 1,
                "multiple_correct": 1,
                "roberta_win_rate": {
                    "$cond": [
                        {"$gt": ["$total_disagreements", 0]},
                        {"$divide": ["$roberta_wins", "$total_disagreements"]},
                        0
                    ]
                },
                "xgb_win_rate": {
                    "$cond": [
                        {"$gt": ["$total_disagreements", 0]},
                        {"$divide": ["$xgb_wins", "$total_disagreements"]},
                        0
                    ]
                },
                "bert_win_rate": {
                    "$cond": [
                        {"$gt": ["$total_disagreements", 0]},
                        {"$divide": ["$bert_wins", "$total_disagreements"]},
                        0
                    ]
                },
                "_id": 0
            }
        },
        {"$sort": {"total_disagreements": -1}}
    ]
    return list(collection.aggregate(pipeline))


def plot_bar_chart(df, title, ylabel, columns, colors, rotate_xticks=True):
    ax = df.plot(kind='bar', x='category', y=columns,
                 color=colors, figsize=(12, 6))
    plt.title(title)
    plt.ylabel(ylabel)
    if rotate_xticks:
        plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def compute_confusion_matrix(db, field="final_prediction_label", model="voting", is_source_of_truth=False):
    collection = db["jobs"]
    pipeline = [
        {
            "$match": {
                "new_job": {
                    "$exists": True
                },
                "category": {"$ne": None, "$ne": ""},
                field: {"$ne": None, "$ne": ""},
                "approved": {"$exists": False}
            }
        },
        {
            "$group": {
                "_id": {
                    "true_label": "$category",
                    "predicted_label": "$"+field
                },
                "count": {"$sum": 1}
            }
        }
    ]

    if is_source_of_truth:
        pipeline = [
            {
                "$match": {
                    "data_from": {"$eq": "linkedin"}
                }
            },
            {
                "$group": {
                    "_id": {
                        "true_label": "$category",
                        "predicted_label": "$"+field
                    },
                    "count": {"$sum": 1}
                }
            }
        ]

    results = list(collection.aggregate(pipeline))

    if not results:
        print("No data found for confusion matrix.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df["true_label"] = df["_id"].apply(lambda x: x["true_label"])
    df["predicted_label"] = df["_id"].apply(lambda x: x["predicted_label"])
    df = df[["true_label", "predicted_label", "count"]]

    # Pivot to confusion matrix
    conf_matrix_df = df.pivot_table(
        index="true_label",
        columns="predicted_label",
        values="count",
        fill_value=0
    )

    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(conf_matrix_df.astype(int), annot=True,
                fmt="d", cmap="Blues", cbar=True)
    plt.title(
        f"Confusion Matrix: True vs Predicted {f"Categories with {model} Unseen data" if not is_source_of_truth else f"Categories with {model} source of truth data"}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return conf_matrix_df


def main():
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    per_cat = [c for c in per_category_accuracy(
        collection) if c["category"] is not None]
    df_accuracy = pd.DataFrame(per_cat)
    print("\n=== Per-category Accuracy ===")
    print(df_accuracy.to_string(index=False))
    plot_bar_chart(df_accuracy, "Per-Category Accuracy", "Accuracy",
                   ["roberta_accuracy", "xgb_accuracy", "bert_accuracy"], ['orange', 'green', 'red'])
    # update_view(collection)
    update_final_prediction_labels(collection)
    per_cat = [c for c in per_category_accuracy_with_voting(
        db) if c["category"] is not None]
    df_accuracy = pd.DataFrame(per_cat)
    print("\n=== Per-category Accuracy ===")
    print(df_accuracy.to_string(index=False))
    plot_bar_chart(df_accuracy, "Per-category Accuracy with voting", "accuracy",
                   ["accuracy"], ['green'])

    compute_confusion_matrix(db)

    # === Disagreement Analysis ===
    disagreements = [d for d in disagreement_analysis(
        collection) if d["category"] is not None]
    df_disagreement = pd.DataFrame(disagreements)
    print("\n=== Disagreement Analysis ===")
    print(df_disagreement.to_string(index=False))
    plot_bar_chart(df_disagreement, "Disagreement Analysis", "Count",
                   ["roberta_wins", "xgb_wins", "bert_wins"], ['red', 'blue', 'orange'])

    client.close()


if __name__ == "__main__":
    main()
