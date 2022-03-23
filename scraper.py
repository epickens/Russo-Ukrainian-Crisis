import tweepy

client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAPoUagEAAAAAb%2BP9cYNsVMObf%2Bgg3B03JMHMH68%3D8mr3obWw8Ape83TDUH1rEkFPHUew5QtL6WP94mzHo4uEBXPwnb')

# ids = [1504976054607306753, 1504976055823908864, 1504976055815180294, 1504976055920209920, 1504976055928594436]
# tweets = client.get_tweets(ids=ids)#, tweet_fields=['context_annotations','created_at','geo']

search_words = ["ukraine", "kiev", "kyiv"]



for tweet in tweets.data:
    print(tweet)