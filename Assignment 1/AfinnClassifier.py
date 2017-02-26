from nltk.tokenize import TweetTokenizer
from Classifier import Classifier
from afinn import Afinn
import Tweets
import operator

class AfinnTweetClassifier(Classifier):

	def __init__(self):
		self._tknzr = TweetTokenizer(strip_handles=True)
		self._tweets = Tweets.TrainTweets()
		self._afinn = {}
		self._afinn = Afinn(emoticons=True)

	def _tokenize(self, tweet):
		return [token for token in self._tknzr.tokenize(tweet['text'])]

	def _score_words(self, tweet):
		tweet_tokens = self._tokenize(tweet)
		prob = {"positive": 0, "negative": 0, "neutral": 0}
		for token in tweet_tokens:
			score = self._afinn.score(token)
			if score > 0: 
				prob["positive"] += score
			elif score < 0: 
				prob["negative"] += (-score)
			else:
				prob["neutral"] += 0.1
		return prob

	def _normalize(self, result):
		scores = float(sum([result['positive'], result['negative'], result['neutral']]))
		return {"positive": result['positive'] / scores, "negative": result['negative'] / scores, "neutral": result['neutral'] / scores}

	def classify(self, tweet):
		results = self._score_words(tweet)
		results = sorted(results.items(), key=operator.itemgetter(1))
		return results[-1][0]
		
	def classify_prob(self, tweet):
		results = self._score_words(tweet)
		return self._normalize(results)

if __name__ == '__main__':

	af = AfinnTweetClassifier()
	results = af.classify_tweets()
	# af.classify_tweets_prob_export()
	#              precision    recall  f1-score   support

	#    negative       0.65      0.92      0.76       285
	#     neutral       0.70      0.51      0.59       474
	#    positive       0.85      0.86      0.85       720

	# avg / total       0.76      0.76      0.75      1479