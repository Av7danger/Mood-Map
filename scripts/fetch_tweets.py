import snscrape.modules.twitter as sntwitter
import pandas as pd
import time
import random
from fake_useragent import UserAgent
import urllib.robotparser
import logging
import json
from src.utils.logging_utils import setup_logging
from src.utils.data_validation import validate_tweets

# Load configuration
with open("fetch_tweets_config.json", "r") as config_file:
    config = json.load(config_file)

batch_size = config.get("batch_size", 1000)
keywords = config.get("keywords", [])
query = config.get("query", "")

# Setup logging
logger = setup_logging("scraping_activity.log")

# Set maximum number of tweets to a safer limit
max_tweets = 1000  # Reduced from 10,000 to 1,000 for safer scraping
tweets = []

# Initialize user agent and robots.txt parser
ua = UserAgent()
rp = urllib.robotparser.RobotFileParser()
rp.set_url("https://twitter.com/robots.txt")
rp.read()

# Scrape tweets in batches
batch_number = 1
while True:
    tweets = []
    logger.info(f"Starting batch {batch_number}...")
    for attempt in range(5):
        try:
            if not rp.can_fetch(ua.random, "https://twitter.com/search"):
                logger.warning("Scraping is not allowed by robots.txt")
                break

            for tweet in sntwitter.TwitterSearchScraper(query).get_items():
                if tweet.retweetedTweet is not None:
                    continue
                tweets.append([
                    tweet.id,
                    tweet.date,
                    "NO_QUERY",
                    tweet.user.username,
                    tweet.content,
                    tweet.likeCount,
                    tweet.retweetCount
                ])
                if len(tweets) >= batch_size:
                    break
                time.sleep(random.uniform(2, 5))

            logger.info(f"Successfully scraped {len(tweets)} tweets in batch {batch_number}.")
            break

        except Exception as e:
            logger.error(f"Error in batch {batch_number}: {e}")
            time.sleep(2 ** attempt)

    tweets = validate_tweets(tweets)
    if tweets:
        df = pd.DataFrame(tweets, columns=["ID", "Date", "Query", "Username", "Tweet", "LikeCount", "RetweetCount"])
        batch_file = f"tweets_batch_{batch_number}.csv"
        df.to_csv(batch_file, index=False)
        logger.info(f"Saved {len(tweets)} tweets to '{batch_file}'")
    else:
        logger.warning(f"No valid tweets in batch {batch_number}.")
        break

    batch_number += 1
    user_input = input("Scrape another batch? (yes/no): ").strip().lower()
    if user_input != "yes":
        logger.info("Stopping scraping process.")
        break