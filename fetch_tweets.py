import tweepy
import csv

# Twitter API credentials
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
ACCESS_TOKEN = "your_access_token"
ACCESS_TOKEN_SECRET = "your_access_token_secret"

# Authenticate with the Twitter API
def authenticate_twitter():
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

# Fetch tweets based on a query, respecting the free tier limit of 100 tweets per month.
def fetch_tweets(api, query, max_tweets=100):
    """
    Fetch tweets based on a query, respecting the free tier limit of 100 tweets per month.

    Args:
        api: Authenticated Tweepy API object.
        query (str): Search query (e.g., keywords or hashtags).
        max_tweets (int): Maximum number of tweets to fetch (default: 100).

    Returns:
        list: List of tweet texts.
    """
    tweets = []
    try:
        # Ensure we do not exceed the free tier limit
        max_tweets = min(max_tweets, 100)

        for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(max_tweets):
            tweets.append(tweet.full_text)
    except Exception as e:
        print(f"Error fetching tweets: {e}")
    return tweets

# Save tweets to a CSV file
def save_tweets_to_csv(file_path, tweets):
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["text"])
            for tweet in tweets:
                writer.writerow([tweet])
        print(f"Successfully saved {len(tweets)} tweets to {file_path}.")
    except Exception as e:
        print(f"Error saving tweets to CSV: {e}")

if __name__ == "__main__":
    # Authenticate with Twitter
    api = authenticate_twitter()

    # Define the query and number of tweets to fetch
    query = "#example OR #hashtag"  # Replace with your desired keywords or hashtags
    max_tweets = 100

    # Fetch tweets
    tweets = fetch_tweets(api, query, max_tweets)

    # Save tweets to a CSV file
    output_file = "data/raw/fetched_tweets.csv"
    save_tweets_to_csv(output_file, tweets)