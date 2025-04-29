def validate_tweets(tweets):
    valid_tweets = []
    for tweet in tweets:
        if tweet[4] and len(tweet[4].strip()) > 0:  # Ensure tweet content is not empty
            valid_tweets.append(tweet)
    return valid_tweets