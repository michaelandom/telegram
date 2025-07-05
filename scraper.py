import asyncio
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
import pandas as pd
from datetime import datetime, timezone
import json
import os
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from urllib.parse import urlparse, parse_qs
from afriwork_scraper_mini_app import AfriworketMiniAppScraper
from hahu_web_site_scraper import HahuWebSiteScraper
from extract_and_add_category import ExtractAndAddCategory
from extract_title_and_label_categories_jobs import ExtractTitleAndLabelCategoryAfriwork
from afriwork_to_csv import AfriworkToCsv
from hahu_web_to_csv import HahuToCsv
from job_data_set import JobDataSet
from job_postings_aggregation import JobPostingsVisualizer
from category_test import CategoryAggregator
class Config:
    def __init__(self):
        self.API_ID = '29511239'
        self.API_HASH = '''29501cdd06a5bd2bfdef1e0691dbd1af'''
        self.PHONE_NUMBER = '+49151759448'
        self.MONGO_URI = "mongodb://localhost:27017/"
        self.DB_NAME = "telegram"
        self.LIMIT = 500
        self.SCRAPE_TELEGRAM= False
        self.SCRAPE_AFRIWORKET= False
        self.SCRAPE_HAHU_WEB= False
        self.ADD_CATEGORY=True
        self.UPDATE_CATEGORY_COUNT=True
        self.FIX_TITLE_AFRIWORK=True
        self.FIX_TITLE_GEEZJOB=True
        self.FIX_TITLE_LINKEDIN=True
        self.FIX_TITLE_HAHU_TELEGRAM=True
        self.MOVE_ALL_JOBS=True
        self.EXPORT_CHART=False
        self.EXPORT_AFRIWORK=False
        self.EXPORT_ALL_JOBS=False
        self.EXPORT_HAHU_WEB=False
        self.EXPORT_LINKEDIN=False
        self.CHANNEL_USERNAME_LIST = ['hahujobs','freelance_ethio', 'geezjob']


class TelegramChannelScraper:

    def __init__(self, api_id, api_hash, phone_number, mongo_uri, db_name):

        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.client = TelegramClient(
            'session', api_id, api_hash)
        try:

            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[db_name]
            self.mongo_client.admin.command('ping')
            print(f"Connected to MongoDB successfully! Database: {db_name}")

            self.setup_collections()

        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise

    def setup_collections(self):
        self.messages_collection = self.db.messages
        self.messages_collection.create_index(
            [("channel_id", 1), ("message_id", 1)], unique=True)
        self.messages_collection.create_index("date")
        self.messages_collection.create_index("channel_username")

        self.channels_collection = self.db.channels
        self.channels_collection.create_index("channel_id", unique=True)
        self.channels_collection.create_index("username", unique=True)

        self.participants_collection = self.db.participants
        self.participants_collection.create_index(
            [("channel_id", 1), ("user_id", 1)], unique=True)

        print("MongoDB collections and indexes created successfully!")

    async def connect_and_authenticate(self):
        await self.client.start()

        if not await self.client.is_user_authorized():
            await self.client.send_code_request(self.phone_number)
            try:
                await self.client.sign_in(self.phone_number, input("Enter the code: "))
            except SessionPasswordNeededError:
                password = input(
                    "Two-factor authentication enabled. Please enter your password: ")
                await self.client.sign_in(password=password)

    async def get_channel_info(self, channel_username):

        try:
            entity = await self.client.get_entity(channel_username)

            info = {
                'channel_id': entity.id,
                'title': entity.title,
                'username': entity.username,
                'description': entity.about if hasattr(entity, 'about') else None,
                'participants_count': entity.participants_count if hasattr(entity, 'participants_count') else None,
                'created_date': entity.date if hasattr(entity, 'date') else None,
                'verified': entity.verified if hasattr(entity, 'verified') else False,
                'restricted': entity.restricted if hasattr(entity, 'restricted') else False,
                'last_updated': datetime.now(timezone.utc)
            }

            try:
                self.channels_collection.update_one(
                    {'channel_id': entity.id},
                    {'$set': info},
                    upsert=True
                )
                print(f"Channel info saved to MongoDB: {info['title']}")
            except Exception as e:
                print(f"Error saving channel info to MongoDB: {e}")

            return info
        except Exception as e:
            print(f"Error getting channel info: {e}")
            return None

    async def update_channel_stats(self, channel_id, status):
        """Get basic state about a channel and save to MongoDB"""
        try:
            info = {
                **{k: v for k, v in status.items() if k != '_id'},
                'last_updated': datetime.now(timezone.utc)
            }

            try:
                self.channels_collection.update_one(
                    {'channel_id': channel_id},
                    {'$set': info},
                    upsert=True
                )
                print(
                    f"Channel stats saved to MongoDB: {info['last_updated']}")
            except Exception as e:
                print(f"Error saving channel info to MongoDB: {e}")

            return info
        except Exception as e:
            print(f"Error getting channel info: {e}")
            return None

    async def get_channel_messages(self, channel_username, limit=100, offset_date=None, save_to_db=True):
        messages_data = []

        try:
            entity = await self.client.get_entity(channel_username)
            channel_id = entity.id
            channel_username_clean = entity.username or channel_username.replace(
                '@', '')

            async for message in self.client.iter_messages(entity, limit=limit, offset_date=offset_date):
                url, job_reply_markup_id = self.get_button_url_and_job_reply_markup_id(
                    message)
                message_info = {
                    'message_id': message.id,
                    'channel_id': channel_id,
                    'channel_username': channel_username_clean,
                    'date': message.date,
                    'text': message.text or '',
                    'views': message.views or 0,
                    'forwards': message.forwards or 0,
                    'reply_to_msg_id': message.reply_to_msg_id,
                    'media_type': None,
                    'media_caption': None,
                    'sender_id': message.from_id.user_id if message.from_id else None,
                    'is_outgoing': message.out,
                    'mentioned': message.mentioned,
                    'media_unread': message.media_unread,
                    'silent': message.silent,
                    'post': message.post,
                    'legacy': message.legacy,
                    'edit_hide': message.edit_hide,
                    'pinned': message.pinned,
                    'grouped_id': message.grouped_id,
                    'scraped_at': datetime.now(timezone.utc),
                    'reply_markup': url,
                    'job_reply_markup_id': job_reply_markup_id
                }

                if message.media:
                    message_info['media_type'] = type(message.media).__name__
                    if hasattr(message.media, 'caption'):
                        message_info['media_caption'] = message.media.caption

                if hasattr(message, 'reactions') and message.reactions:
                    reactions = {}
                    for reaction in message.reactions.results:
                        emoji = reaction.reaction.emoticon if hasattr(
                            reaction.reaction, 'emoticon') else str(reaction.reaction)
                        reactions[emoji] = reaction.count
                    message_info['reactions'] = reactions
                else:
                    message_info['reactions'] = {}

                messages_data.append(message_info)

                if save_to_db:
                    try:
                        self.messages_collection.update_one(
                            {'channel_id': channel_id, 'message_id': message.id},
                            {'$set': message_info},
                            upsert=True
                        )
                    except DuplicateKeyError:
                        self.messages_collection.update_one(
                            {'channel_id': channel_id, 'message_id': message.id},
                            {'$set': message_info}
                        )
                    except Exception as e:
                        print(
                            f"Error saving message {message.id} to MongoDB: {e}")

            if save_to_db:
                print(f"Saved {len(messages_data)} messages to MongoDB")

        except Exception as e:
            print(f"Error getting messages: {e}")
            return []

        return messages_data

    def get_button_url_and_job_reply_markup_id(self, message):
        try:
            url = message.reply_markup.rows[0].buttons[0].url
            job_reply_markup_id = None
            if url:
                job_reply_markup_id = self.extract_startapp_id(url)
            return url, job_reply_markup_id
        except (AttributeError, IndexError, TypeError):
            return None, None

    def extract_startapp_id(self, url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get('startapp', [None])[0] or query_params.get('start', [None])[0]

    async def save_to_csv(self, messages_data, filename):
        if not messages_data:
            print("No data to save")
            return

        for msg in messages_data:
            msg['reactions'] = json.dumps(msg['reactions'])

        df = pd.DataFrame(messages_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Data saved to {filename}")

    async def save_to_json(self, messages_data, filename):
        if not messages_data:
            print("No data to save")
            return

        for msg in messages_data:
            if isinstance(msg['date'], datetime):
                msg['date'] = msg['date'].isoformat()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {filename}")

    async def get_channel_participants(self, channel_username, limit=None, save_to_db=True):
        try:
            entity = await self.client.get_entity(channel_username)
            channel_id = entity.id
            participants = []

            async for participant in self.client.iter_participants(entity, limit=limit):
                participant_info = {
                    'user_id': participant.id,
                    'channel_id': channel_id,
                    'username': participant.username,
                    'first_name': participant.first_name,
                    'last_name': participant.last_name,
                    'phone': participant.phone,
                    'is_bot': participant.bot,
                    'is_verified': participant.verified,
                    'is_restricted': participant.restricted,
                    'is_scam': participant.scam if hasattr(participant, 'scam') else False,
                    'scraped_at': datetime.now(timezone.utc)
                }
                participants.append(participant_info)

                if save_to_db:
                    try:
                        self.participants_collection.update_one(
                            {'channel_id': channel_id, 'user_id': participant.id},
                            {'$set': participant_info},
                            upsert=True
                        )
                    except Exception as e:
                        print(
                            f"Error saving participant {participant.id} to MongoDB: {e}")

            if save_to_db:
                print(f"Saved {len(participants)} participants to MongoDB")

            return participants
        except Exception as e:
            print(f"Error getting participants: {e}")
            return []

    def get_messages_from_db(self, channel_username=None, start_date=None, end_date=None, limit=None):
        query = {}

        if channel_username:
            query['channel_username'] = channel_username.replace('@', '')

        if start_date or end_date:
            date_query = {}
            if start_date:
                date_query['$gte'] = start_date
            if end_date:
                date_query['$lte'] = end_date
            query['date'] = date_query

        cursor = self.messages_collection.find(query).sort('date', -1)

        if limit:
            cursor = cursor.limit(limit)

        return list(cursor)

    def get_channel_stats(self, channel_username):
        """Get statistics for a channel from MongoDB"""
        channel_username_clean = channel_username.replace('@', '')

        pipeline = [
            {'$match': {'channel_username': channel_username_clean}},
            {'$group': {
                '_id': '$channel_username',
                'total_messages': {'$sum': 1},
                'total_views': {'$sum': '$views'},
                'total_forwards': {'$sum': '$forwards'},
                'avg_views_per_message': {'$avg': '$views'},
                'avg_forwards_per_message': {'$avg': '$forwards'},
                'first_message_date': {'$min': '$date'},
                'last_message_date': {'$max': '$date'}
            }}
        ]

        result = list(self.messages_collection.aggregate(pipeline))
        return result[0] if result else None

    def close_db_connection(self):
        """Close MongoDB connection"""
        self.mongo_client.close()
        print("MongoDB connection closed")

    async def disconnect(self):
        """Disconnect from Telegram and close MongoDB connection"""
        await self.client.disconnect()
        self.close_db_connection()

async def main():
    print("Starting Job Scraper Application")
    print("=" * 60)
    
    config = Config()
    
    scraper = TelegramChannelScraper(
        config.API_ID, config.API_HASH, config.PHONE_NUMBER, config.MONGO_URI, config.DB_NAME)

    try:
        print("Connecting to Telegram API...")
        await scraper.connect_and_authenticate()
        print("Successfully connected to Telegram!")
        
        if config.SCRAPE_TELEGRAM:
            print("\n TELEGRAM CHANNEL SCRAPING")
            print("-" * 40)
            
            channel_usernames = config.CHANNEL_USERNAME_LIST
            total_channels = len(channel_usernames)
            
            for idx, channel_username in enumerate(channel_usernames, 1):
                print(f"\n Processing Channel {idx}/{total_channels}: @{channel_username}")
                
                # Get channel info
                print(f" Fetching channel information...")
                channel_info = await scraper.get_channel_info(channel_username)
                
                if channel_info:
                    print(f"   Channel: {channel_info['title']}")
                    # print(f"   Participants: {channel_info['participants_count']:,}")
                    
                    if channel_info['description']:
                        desc_preview = channel_info['description'][:100]
                        print(f"   Description: {desc_preview}{'...' if len(channel_info['description']) > 100 else ''}")
                    else:
                        print(f"   Description: No description available")
                else:
                    print(f"   ERROR: Failed to retrieve channel information")
                    continue

                # Get messages
                print(f"   Retrieving messages (limit: {config.LIMIT})...")
                messages = await scraper.get_channel_messages(channel_username, limit=config.LIMIT, save_to_db=True)
                print(f"   Successfully retrieved and saved {len(messages):,} messages")

                # Get and update statistics
                print(f"   Calculating channel statistics...")
                stats = scraper.get_channel_stats(channel_username)
                await scraper.update_channel_stats(channel_info["channel_id"], stats)
                print(f"   Channel statistics updated")

                # Uncommented participants scraping (if needed)
                # print(f"   Retrieving channel participants...")
                # participants = await scraper.get_channel_participants(channel_username, save_to_db=True)
                # print(f"   Retrieved {len(participants):,} participants")
            
            print(f"\nTelegram scraping completed! Processed {total_channels} channels")
        
        if config.SCRAPE_AFRIWORKET:
            print("\nAFRIWORKET MINI APP SCRAPING")
            print("-" * 40)
            print("Starting Afriworket mini app scraper...")
            
            afriworketMiniAppScraper = AfriworketMiniAppScraper(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            afriworketMiniAppScraper.run()
            
            print("Afriworket mini app scraping completed successfully!")
            print("=" * 50)
        
        if config.SCRAPE_HAHU_WEB:
            print("\nHAHU WEBSITE SCRAPING")
            print("-" * 40)
            print("Starting Hahu website scraper...")
            
            hahuWebSiteScraper = HahuWebSiteScraper(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="huhu"
            )
            hahuWebSiteScraper.run()
            
            print("Hahu website scraping completed successfully!")
            print("=" * 50)
        
        if config.FIX_TITLE_AFRIWORK:
            print("\nFIXING AFRIWORK TITLES AND CATEGORIES")
            print("-" * 40)
            print("Extracting titles and labeling categories for Afriwork data...")
            
            extractTitleAndLabelCategoryAfriwork = ExtractTitleAndLabelCategoryAfriwork(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            extractTitleAndLabelCategoryAfriwork.run()
            
            print("Afriwork title extraction and category labeling completed!")
            print("=" * 50)
        if config.FIX_TITLE_GEEZJOB:
            print("\nFIXING GEEZJOB TITLES AND CATEGORIES")
            print("-" * 40)
            print("Extracting titles and labeling categories for GEEZJOB data...")
            
            extractTitleAndLabelCategoryAfriwork = ExtractTitleAndLabelCategoryAfriwork(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            extractTitleAndLabelCategoryAfriwork.run(channel_username = "geezjob")
            
            print("GEEZJOB title extraction and category labeling completed!")
            print("=" * 50)
        if config.FIX_TITLE_LINKEDIN:
            print("\nFIXING LINKEDIN TITLES AND CATEGORIES")
            print("-" * 40)
            print("Extracting titles and labeling categories for LINKEDIN data...")
            
            extractTitleAndLabelCategoryAfriwork = ExtractTitleAndLabelCategoryAfriwork(
                mongo_uri=config.MONGO_URI, 
                db_name="linkedin_jobs", 
                collection_name="jobs"
            )
            extractTitleAndLabelCategoryAfriwork.run(channel_username = "linkedin_jobs")
            
            print("LINKEDIN title extraction and category labeling completed!")
            print("=" * 50)
        if config.FIX_TITLE_HAHU_TELEGRAM:
            print("\nFIXING HAHU TITLES AND CATEGORIES")
            print("-" * 40)
            print("Extracting titles and labeling categories for Afriwork data...")
            
            extractTitleAndLabelCategoryAfriwork = ExtractTitleAndLabelCategoryAfriwork(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            extractTitleAndLabelCategoryAfriwork.run(channel_username = "hahujobs")
            
            print("HAHU title extraction and category labeling completed!")
            print("=" * 50)

        if config.ADD_CATEGORY:
            print("\nADDING CATEGORIES TO DATA")
            print("-" * 40)
            
            print("Adding categories to Hahu data...")
            extractAndAddCategoryHahu = ExtractAndAddCategory(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="huhu"
            )
            extractAndAddCategoryHahu.run()
            print("Hahu category extraction completed!")
            
            print("Adding categories to Messages data...")
            extractAndAddCategoryMessages = ExtractAndAddCategory(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            extractAndAddCategoryMessages.run()
            print("Messages category extraction completed!")
            print("=" * 50)
        
        if config.MOVE_ALL_JOBS:
            
            print("\nMove linkedin Data to job")
            print("-" * 40)
            
            print("Move Data from linkedin data...")
            extractAndAddCategoryHahu = JobDataSet(
                mongo_uri=config.MONGO_URI, 
                db_name="linkedin_jobs", 
                collection_name="jobs"
            )
            extractAndAddCategoryHahu.move_job_from_linkedin()
            print("linkedin  completed!")

            print("\nMove Data to job")
            print("-" * 40)
            
            print("Move Data from Hahu data...")
            extractAndAddCategoryHahu = JobDataSet(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="huhu"
            )
            extractAndAddCategoryHahu.move_job_from_hahu_web()
            print("Hahu  completed!")

            print("\nMove Telegram Data to job")
            print("-" * 40)
            
            print("Move Data from Telegram data...")
            extractAndAddCategoryHahu = JobDataSet(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            extractAndAddCategoryHahu.move_job_from_message()
            print("Telegram  completed!")

            category_aggregator = CategoryAggregator(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="jobs"
            )
            category_aggregator.run()





        if config.EXPORT_CHART:
            
            print("\nEXPORT CHART for  messages")
            print("-" * 40)
            
            print("Adding EXPORT CHART to messages data...")
            jobPostingsVisualizer = JobPostingsVisualizer(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            jobPostingsVisualizer.aggregate_and_plot_from_messages()
            print("EXPORT CHART messages extraction completed!")

            print("\nEXPORT CHART for  hahu")
            print("-" * 40)
            
            print("Adding EXPORT CHART to hahu data...")
            jobPostingsVisualizer = JobPostingsVisualizer(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="huhu"
            )
            jobPostingsVisualizer.aggregate_and_plot_hahu()
            print("EXPORT CHART hahu extraction completed!")

            print("\naggregate_and_plot_linkedin for  linkedin_job")
            print("-" * 40)
            
            print("Adding aggregate_and_plot_linkedin to linkedin_job data...")
            jobPostingsVisualizer = JobPostingsVisualizer(
                mongo_uri=config.MONGO_URI, 
                db_name="linkedin_job", 
                collection_name="jobs"
            )
            jobPostingsVisualizer.aggregate_and_plot_linkedin()
            print("aggregate_and_plot_linkedin linkedin_job extraction completed!")


            print("plot_channel_summary for  telegram")
            print("-" * 40)
            
            print("Adding plot_channel_summary to linkedin_job data...")
            jobPostingsVisualizer = JobPostingsVisualizer(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="channels"
            )
            jobPostingsVisualizer.plot_channel_summary()
            print("plot_channel_summary linkedin_job extraction completed!")


        if config.EXPORT_HAHU_WEB:
            print("\nEXPORTING HAHU DATA")
            print("-" * 40)
            print("Exporting Hahu jobs to CSV format...")
            
            hahuToCsv = HahuToCsv(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="huhu"
            )
            hahuToCsv.export_jobs_to_csv()
            
            print("Hahu data export to CSV completed!")
            print("=" * 50)
        if config.EXPORT_AFRIWORK:
            print("\nEXPORTING AFRIWORK DATA")
            print("-" * 40)
            print("Exporting Afriwork jobs to CSV format...")
            
            afriworkToCsv = AfriworkToCsv(
                mongo_uri=config.MONGO_URI, 
                db_name=config.DB_NAME, 
                collection_name="messages"
            )
            afriworkToCsv.export_jobs_to_csv(get_all_jobs=config.EXPORT_ALL_JOBS)
            
            print("Afriwork data export to CSV completed!")
            print("=" * 50)
        if config.EXPORT_LINKEDIN:
            print("\nEXPORTING LINKEDIN DATA")
            print("-" * 40)
            print("Exporting LINKEDIN jobs to CSV format...")
            
            afriworkToCsv = AfriworkToCsv(
                mongo_uri=config.MONGO_URI, 
                db_name="linkedin_jobs", 
                collection_name="jobs",
                output_file="linkedin_output.csv"
            )
            afriworkToCsv.export_jobs_to_csv(get_all_jobs=True)
            
            print("LINKEDIN data export to CSV completed!")
            print("=" * 50)
        
            if config.EXPORT_ALL_JOBS:
                print("\nEXPORTING ALL JOB into one csv")
                print("-" * 40)
                print("Exporting All Jobs to CSV format...")
                afriworkToCsv.merge_csv_without_duplicates("hahu_output.csv","telegram_output.csv","linkedin_output.csv","job_output.csv")
                print("ALL JOB export to CSV completed!")
                print("=" * 50)
        
        
        print("\nALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")
        print("=" * 60)
        
    finally:
        await scraper.disconnect()
        print("Disconnected from Telegram and MongoDB")


if __name__ == "__main__":
    asyncio.run(main())