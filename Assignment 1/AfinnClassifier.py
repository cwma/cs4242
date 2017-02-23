from nltk.tokenize import TweetTokenizer
from Classifier import Classifier
import Tweets
import operator

class AfinnTweetClassifier(Classifier):

	def __init__(self):
		self._tknzr = TweetTokenizer(strip_handles=True)
		self._tweets = Tweets.TrainTweets()
		self._afinn = {}
		self._train()

	def _tokenize(self, tweet):
		return [token for token in self._tknzr.tokenize(tweet['text'])]

	def _load_afinn(self):
		afinn_file = open("AFINN-111.txt", "r")
		for term in afinn_file:
			term = term.split()
			self._afinn["".join(term[:-1])] = int(term[-1])

	def _train(self):
		self._load_afinn()

	def _score_words(self, tweet):
		tweet_tokens = self._tokenize(tweet)
		prob = {"positive": 0, "negative": 0, "neutral": 0}
		for token in tweet_tokens:
			if token in self._afinn:
				score = self._afinn[token]
				if score > 0: prob["positive"] += score
				else: prob["negative"] += (-score)
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
	# results = af.classify_tweets()
	# print("Correct: {0}, Wrong: {1}, Total: {2}".format(*results))
	# print("Percentage: {0}".format(results[0] / results[2]))
	af.classify_tweets_prob_export()