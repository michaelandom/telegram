import pymongo
import pandas as pd
import matplotlib.pyplot as plt


class JobPostingsVisualizer:
    def __init__(self, mongo_uri, db_name, collection_name):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def aggregate_and_plot_linkedin(self):
        pipelines = {
            "company_name": [
                {"$group": {"_id": "$company_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ],
            "seniority_level": [
                {"$group": {"_id": "$seniority_level", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ],
            "employment_type": [
                {"$group": {"_id": "$employment_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ],
            "job_location": [
                {"$group": {"_id": "$job_location", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ],
            "category": [
                {"$group": {"_id": "$category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
        }

        plt.style.use('ggplot')

        for field, pipeline in pipelines.items():
            results = list(self.collection.aggregate(pipeline))
            if not results:
                print(f"No data found for {field}. Skipping.")
                continue

            df = pd.DataFrame(results)
            df = df.rename(columns={'_id': field})

            if field in ['seniority_level', 'employment_type']:
                plt.figure(figsize=(6, 6))
                plt.pie(df['count'], labels=df[field],
                        autopct='%1.1f%%', startangle=140)
                plt.title(f'Job Postings by {field.replace("_", " ").title()}')
                plt.tight_layout()
                plt.savefig(f'job_postings_linkedin_by_{field}.png', dpi=300, bbox_inches='tight')

            elif field == 'job_location':
                plt.figure(figsize=(8, 6))
                plt.barh(df[field], df['count'], color='lightgreen')
                plt.xlabel('Number of Postings')
                plt.ylabel('Job Location')
                plt.title('Job Postings by Location')
                plt.tight_layout()
                plt.savefig(f'job_postings_linkedin_by_{field}.png', dpi=300, bbox_inches='tight')

            else:
                plt.figure(figsize=(8, 6))
                plt.bar(df[field], df['count'], color='skyblue')
                plt.xlabel(field.replace("_", " ").title())
                plt.ylabel('Number of Postings')
                plt.title(f'Job Postings by {field.replace("_", " ").title()}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'job_postings_linkedin_by_{field}.png', dpi=300, bbox_inches='tight')


    def aggregate_and_plot_from_messages(self):
        plt.style.use('ggplot')

        pipeline_channel = [
            {"$group": {"_id": "$channel_username", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(self.collection.aggregate(pipeline_channel))
        if results:
            df = pd.DataFrame(results).rename(
                columns={'_id': 'channel_username'})
            plt.figure(figsize=(8, 6))
            plt.bar(df['channel_username'],
                    df['count'], color='mediumseagreen')
            plt.xlabel('Channel Username')
            plt.ylabel('Number of Messages')
            plt.title('Messages by Channel Username')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'job_postings_telegram_by_channel_name.png', dpi=300, bbox_inches='tight')
        

        pipeline_date = [
            {
                "$project": {
                    "year": {"$year": "$date"},
                    "month": {"$month": "$date"}
                }
            },
            {
                "$project": {
                    "quarter": {
                        "$concat": [
                            {"$toString": "$year"},
                            "-Q",
                            {
                                "$toString": {
                                    "$ceil": {"$divide": ["$month", 3]}
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": "$quarter",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        results = list(self.collection.aggregate(pipeline_date))
        if results:
            df = pd.DataFrame(results).rename(columns={'_id': 'quarter'})
            plt.figure(figsize=(10, 6))
            plt.plot(df['quarter'], df['count'], marker='o', linestyle='-')
            plt.xlabel('Date')
            plt.ylabel('Number of Messages')
            plt.title('Messages by Date')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'job_postings_telegram_by_date.png', dpi=300, bbox_inches='tight')
            
        self.collection.delete_many({"category": {"$eq": None}})
        pipeline_category = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(self.collection.aggregate(pipeline_category))
        if results:
            df = pd.DataFrame(results).rename(columns={'_id': 'category'})
            plt.figure(figsize=(8, 6))
            plt.bar(df['category'], df['count'], color='coral')
            plt.xlabel('Category')
            plt.ylabel('Number of Messages')
            plt.title('Messages by Category')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'job_postings_telegram_by_category.png', dpi=300, bbox_inches='tight')


        pipeline_category_channel = [
            {"$group": {"_id": {"category": "$category",
                                "channel": "$channel_username"}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(self.collection.aggregate(pipeline_category_channel))
        if results:
            df = pd.DataFrame(results)
            df['category'] = df['_id'].apply(lambda x: x['category'])
            df['channel_username'] = df['_id'].apply(lambda x: x['channel'])
            df = df.drop(columns=['_id'])

            pivot_table = pd.pivot_table(
                df, values='count', index='category', columns='channel_username', fill_value=0)

            pivot_table.plot(kind='bar', stacked=True,
                             figsize=(10, 6), colormap='tab20')
            plt.xlabel('Category')
            plt.ylabel('Number of Messages')
            plt.title('Messages by Category and Channel Username')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'job_postings_telegram_by_category_and_channel_name.png', dpi=300, bbox_inches='tight')
            

    def aggregate_and_plot_hahu(self):
        source_pipeline = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        source_data = list(self.collection.aggregate(source_pipeline))
        source_df = pd.DataFrame(source_data)
        if not source_df.empty:
            plt.figure(figsize=(8, 4))
            plt.bar(source_df["_id"], source_df["count"], color="skyblue")
            plt.xlabel("Source")
            plt.ylabel("Number of Jobs")
            plt.title("Job Counts by Source")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        pipeline_date = [
            {
                "$project": {
                    "created_at_date": {
                        "$dateFromString": {
                            "dateString": "$created_at"
                        }
                    }
                }
            },
            {
                "$project": {
                    "year": {"$year": "$created_at_date"},
                    "month": {"$month": "$created_at_date"}
                }
            },
            {
                "$project": {
                    "quarter": {
                        "$concat": [
                            {"$toString": "$year"},
                            "-Q",
                            {
                                "$toString": {
                                    "$ceil": {"$divide": ["$month", 3]}
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": "$quarter",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]

        date_data = list(self.collection.aggregate(pipeline_date))
        date_df = pd.DataFrame(date_data)
        if not date_df.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(date_df["_id"], date_df["count"],
                     marker="o", color="green")
            plt.xlabel("Year-Month")
            plt.ylabel("Number of Jobs")
            plt.title("Job Posts Over Time")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'job_postings_hahu_by_date.png', dpi=300, bbox_inches='tight')
            

        type_pipeline = [
            {
                "$group": {
                    "_id": {
                        "$ifNull": ["$type", ""]
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            }
        ]

        type_data = list(self.collection.aggregate(type_pipeline))
        type_df = pd.DataFrame(type_data)
        if not type_df.empty:
            plt.figure(figsize=(8, 4))
            plt.bar(type_df["_id"], type_df["count"], color="orange")
            plt.xlabel("Job Type")
            plt.ylabel("Number of Jobs")
            plt.title("Job Counts by Type")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'job_postings_hahu_by_type.png', dpi=300, bbox_inches='tight')
            

        gender_pipeline = [
            {
                "$group": {
                    "_id": {
                        "$ifNull": ["$gender_priority", "neutral"]
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            }
        ]

        gender_data = list(self.collection.aggregate(gender_pipeline))
        gender_df = pd.DataFrame(gender_data)
        if not gender_df.empty:
            gender_df["_id"] = gender_df["_id"].fillna("Not Specified")
            plt.figure(figsize=(8, 4))
            plt.bar(gender_df["_id"], gender_df["count"], color="purple")
            plt.xlabel("Gender Priority")
            plt.ylabel("Number of Jobs")
            plt.title("Job Counts by Gender Priority")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'job_postings_hahu_by_gender_priority.png', dpi=300, bbox_inches='tight')
            

    def plot_channel_summary(self):
        value = list(self.collection.find({}))
        print(value)
        df = pd.DataFrame(value)
        df.sort_values(by="total_messages", ascending=False, inplace=True)
        plt.figure(figsize=(10, 5))
        plt.bar(df["title"], df["total_messages"], color="skyblue")
        plt.xlabel("Channel")
        plt.ylabel("Total Messages")
        plt.title("Total Messages by Channel")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f'job_postings_channel_summary_by_channel_total_messages.png', dpi=300, bbox_inches='tight')


        plt.figure(figsize=(10, 5))
        plt.bar(df["title"], df["total_views"], color="lightgreen")
        plt.xlabel("Channel")
        plt.ylabel("Total Views")
        plt.title("Total Views by Channel")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f'job_postings_channel_summary_by_channel_total_views.png', dpi=300, bbox_inches='tight')


        plt.figure(figsize=(10, 5))
        plt.bar(df["title"], df["total_forwards"], color="orange")
        plt.xlabel("Channel")
        plt.ylabel("Total Forwards")
        plt.title("Total Forwards by Channel")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f'job_postings_channel_summary_by_channel_total_forwards.png', dpi=300, bbox_inches='tight')


        plt.figure(figsize=(10, 5))
        plt.bar(df["title"], df["avg_forwards_per_message"], color="purple")
        plt.xlabel("Channel")
        plt.ylabel("Avg Forwards per Message")
        plt.title("Average Forwards per Message by Channel")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f'job_postings_channel_summary_by_channel_avg_forwards_per_message.png', dpi=300, bbox_inches='tight')

        plt.figure(figsize=(10, 5))
        plt.bar(df["title"], df["avg_views_per_message"], color="red")
        plt.xlabel("Channel")
        plt.ylabel("Avg Views per Message")
        plt.title("Average Views per Message by Channel")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f'job_postings_channel_summary_by_channel_avg_views_per_message.png', dpi=300, bbox_inches='tight')


    def close(self):
        self.client.close()
