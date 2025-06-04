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


class TelegramChannelScraper:
    def __init__(self, api_id, api_hash, phone_number, mongo_uri="mongodb://localhost:27017/", db_name="telegram"):
        """
        Initialize the Telegram scraper

        Args:
            api_id: Your Telegram API ID
            api_hash: Your Telegram API Hash
            phone_number: Your phone number (with country code)
            mongo_uri: MongoDB connection URI
            db_name: Database name in MongoDB
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.client = TelegramClient('session', api_id, api_hash)

        # MongoDB setup
        try:
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[db_name]

            # Test connection
            self.mongo_client.admin.command('ping')
            print(f"Connected to MongoDB successfully! Database: {db_name}")

            # Create collections and indexes
            self.setup_collections()

        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise

    def setup_collections(self):
        """Setup MongoDB collections and indexes"""
        # Messages collection
        self.messages_collection = self.db.messages
        self.messages_collection.create_index(
            [("channel_id", 1), ("message_id", 1)], unique=True)
        self.messages_collection.create_index("date")
        self.messages_collection.create_index("channel_username")

        # Channels collection
        self.channels_collection = self.db.channels
        self.channels_collection.create_index("channel_id", unique=True)
        self.channels_collection.create_index("username", unique=True)

        # Participants collection
        self.participants_collection = self.db.participants
        self.participants_collection.create_index(
            [("channel_id", 1), ("user_id", 1)], unique=True)

        print("MongoDB collections and indexes created successfully!")

    async def connect_and_authenticate(self):
        """Connect to Telegram and authenticate"""
        await self.client.start()

        if not await self.client.is_user_authorized():
            await self.client.send_code_request(self.phone_number)
            try:
                await self.client.sign_in(self.phone_number, input('Enter the code: '))
            except SessionPasswordNeededError:
                password = input(
                    'Two-factor authentication enabled. Please enter your password: ')
                await self.client.sign_in(password=password)

    async def get_channel_info(self, channel_username):
        """Get basic information about a channel and save to MongoDB"""
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

            # Save to MongoDB
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
        """
        Get messages from a channel and save to MongoDB

        Args:
            channel_username: Channel username (with or without @)
            limit: Number of messages to retrieve
            offset_date: Get messages before this date
            save_to_db: Whether to save messages to MongoDB
        """
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

                # Handle media
                if message.media:
                    message_info['media_type'] = type(message.media).__name__
                    if hasattr(message.media, 'caption'):
                        message_info['media_caption'] = message.media.caption

                # Handle reactions
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

                # Save to MongoDB if enabled
                if save_to_db:
                    try:
                        self.messages_collection.update_one(
                            {'channel_id': channel_id, 'message_id': message.id},
                            {'$set': message_info},
                            upsert=True
                        )
                    except DuplicateKeyError:
                        # Message already exists, update it
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
        """Save messages data to CSV file"""
        if not messages_data:
            print("No data to save")
            return

        # Convert reactions dict to string for CSV
        for msg in messages_data:
            msg['reactions'] = json.dumps(msg['reactions'])

        df = pd.DataFrame(messages_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Data saved to {filename}")

    async def save_to_json(self, messages_data, filename):
        """Save messages data to JSON file"""
        if not messages_data:
            print("No data to save")
            return

        # Convert datetime objects to strings for JSON serialization
        for msg in messages_data:
            if isinstance(msg['date'], datetime):
                msg['date'] = msg['date'].isoformat()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {filename}")

    async def get_channel_participants(self, channel_username, limit=None, save_to_db=True):
        """Get channel participants and save to MongoDB (only works for channels where you're admin)"""
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

                # Save to MongoDB if enabled
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
        """Retrieve messages from MongoDB with optional filters"""
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

    def export_to_csv(self, channel_username, filename=None):
        """Export channel messages from MongoDB to CSV"""
        messages = self.get_messages_from_db(channel_username)

        if not messages:
            print("No messages found in database")
            return

        # Convert reactions dict to string for CSV
        for msg in messages:
            msg['reactions'] = json.dumps(msg['reactions'])
            # Convert ObjectId to string
            msg['_id'] = str(msg['_id'])

        df = pd.DataFrame(messages)

        if not filename:
            filename = f"{channel_username.replace('@', '')}_messages_from_db.csv"

        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Data exported to {filename}")

    def close_db_connection(self):
        """Close MongoDB connection"""
        self.mongo_client.close()
        print("MongoDB connection closed")

    async def disconnect(self):
        """Disconnect from Telegram and close MongoDB connection"""
        await self.client.disconnect()
        self.close_db_connection()


async def main():
    API_ID = '29511239'
    API_HASH = '''29501cdd06a5bd2bfdef1e0691dbd1af'''
    PHONE_NUMBER = '+49151759448'
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "telegram"

    scraper = TelegramChannelScraper(
        API_ID, API_HASH, PHONE_NUMBER, MONGO_URI, DB_NAME)

    try:
        await scraper.connect_and_authenticate()
        print("Connected to Telegram successfully!")
        channel_usernames = ['hahujobs',
                             'freelance_ethio', 'jobs_in_ethio', 'geezjob']
        # channel_usernames = ['freelance_ethio']
        for channel_username in channel_usernames:
            print(f"\nGetting channel info for {channel_username}...")
            channel_info = await scraper.get_channel_info(channel_username)
            if channel_info:
                print(f"Channel: {channel_info['title']}")
                print(f"Participants: {channel_info['participants_count']}")
                print(
                    f"Description: {channel_info['description'][:100]}..." if channel_info['description'] else "No description")

            print(f"\nGetting messages from {channel_username}...")
            messages = await scraper.get_channel_messages(channel_username, limit=1500, save_to_db=True)
            print(f"Retrieved {len(messages)} messages")

            print("\nGetting statistics from database...")
            stats = scraper.get_channel_stats(channel_username)
            if stats:
                print(f"Total messages in DB: {stats['total_messages']}")
                print(f"Total views: {stats['total_views']}")
                print(f"Total forwards: {stats['total_forwards']}")
                print(
                    f"Average views per message: {stats['avg_views_per_message']:.2f}")
                print(
                    f"Average forwards per message: {stats['avg_forwards_per_message']:.2f}")
                print(
                    f"Date range: {stats['first_message_date']} to {stats['last_message_date']}")
            updateStats = await scraper.update_channel_stats(channel_info["channel_id"], stats)

            print(f"\nExporting data to CSV...")
            scraper.export_to_csv(channel_username)

            # participants = await scraper.get_channel_participants(channel_username, save_to_db=True)
            # print(f"Retrieved {len(participants)} participants")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        await scraper.disconnect()
        print("Disconnected from Telegram and MongoDB")


if __name__ == "__main__":
    asyncio.run(main())
