import json

def Tweets(index_file):
	index = json.load(open(index_file, 'r'))
	tweets = {}
	for tweet_id, tweet in index.items():
		raw_tweet = open("dataset/"+ tweet['text'], 'r', encoding='utf-8').read()
		raw_tweet = json.loads(raw_tweet)
		tweet['filename'] = tweet['text']
		tweet['text'] = raw_tweet['text'].lower()
		tweet['rt'] = raw_tweet['retweet_count']
		tweet['fav'] = raw_tweet['favorite_count']
		tweet['user_desc'] = raw_tweet['user']['description'].lower()
		tweet['followers'] = raw_tweet['user']['followers_count']
		tweet['userid'] = raw_tweet['user']['id']
		tweets[tweet_id] = tweet
	return tweets

def TrainTweets():
	return Tweets('dataset/training.json')

def TestTweets():
	return Tweets('dataset/testing.json')

def DevTweets():
	return Tweets('dataset/development.json')	

if __name__ == '__main__':
    t = Tweets('dataset/testing.json')
