from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.stem.porter import PorterStemmer
import nltk.sentiment.util as sentiment_utils
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from Classifier import Classifier
import Tweets
import json

class NaiveBayesTweetClassifier(Classifier):

	def __init__(self, negation=False, rt_count=True, fav_count=True, desc_feat=False, user_feat=True, follow_count=True, bigrams=False):
		self._tknzr = TweetTokenizer(strip_handles=True)
		self._tweets = Tweets.TrainTweets()
		self._negation = negation
		self._rt_count = rt_count
		self._fav_count = fav_count
		self._desc_feat = desc_feat
		self._user_feat = user_feat
		self._follow_count = follow_count
		self._bigrams = bigrams
		self._stopwords = stopwords.words('english')
		self._stemmer = PorterStemmer()
		self._train()

	def _tokenize(self, tweet):
		return [self._stemmer.stem(token) for token in self._tknzr.tokenize(tweet['text']) if token not in self._stopwords]

	def _mark_negation(self, tweet_tokens):
		return sentiment_utils.mark_negation(tweet_tokens)

	def _rt_feature(self, tweet):
		rt = tweet['rt']
		return "rt(zero)" if rt is 0 else "rt(low)" if rt < 5 else "rt(med)" if rt < 10 else "rt(high)"

	def _fav_feature(self, tweet):
		rt = tweet['fav']
		return "fav(zero)" if rt is 0 else "fav(low)" if rt < 5 else "fav(med)" if rt < 10 else "fav(high)"

	def _user_desc_features(self, tweet):
		desc = tweet['user_desc']
		desc_tokens = self._tknzr.tokenize(desc)
		return {"userdesc({0})".format(token): True for token in desc_tokens}

	def _user_feature(self, tweet):
		return "isUser({0})".format(tweet['userid'])

	def _follower_count_feature(self, tweet):
		followers = tweet['followers']
		return "follower(zero)" if followers is 0 else "follower(low)" if followers < 200 \
		else "follower(med)" if followers < 500 else "follower(high)"

	def _extract_features(self, tweet):
		tweet_tokens = self._tokenize(tweet)
		if self._negation:
			tweet_tokens = self._mark_negation(tweet_tokens)
		if self._bigrams:
			features = {"contains({0} {1})".format(*token): True for token in zip(tweet_tokens, tweet_tokens[1:])}
		else:
			features = {"contains({0})".format(token): True for token in tweet_tokens}
		if self._rt_count: 
			features[self._rt_feature(tweet)] = True
		if self._fav_count:
			features[self._fav_feature(tweet)] = True
		if self._desc_feat:
			features.update(self._user_desc_features(tweet))
		if self._user_feat:
			features[self._user_feature(tweet)] = True
		if self._follow_count:
			features[self._follower_count_feature(tweet)] = True
		return features

	def _train(self):
		train_set = [(self._extract_features(tweet), tweet['label']) for tweet_id, tweet in self._tweets.items()]
		self._classifier = NaiveBayesClassifier.train(train_set)

	def _normalize(self, result):
		scores = float(sum([result.prob('positive'), result.prob('negative'), result.prob('neutral')]))
		return {"positive": result.prob('positive') / scores, "negative": result.prob("negative") / scores, "neutral": result.prob("neutral") / scores}

	def classify(self, tweet):
		features = self._extract_features(tweet)
		return self._classifier.classify(features)

	def classify_prob(self, tweet):
		features = self._extract_features(tweet)
		result = self._classifier.prob_classify(features)
		return self._normalize(result)

if __name__ == '__main__':

	nb = NaiveBayesTweetClassifier()
	# results = nb.classify_tweets()
	# print("Correct: {0}, Wrong: {1}, Total: {2}".format(*results))
	# print("Percentage: {0}".format(results[0] / results[2]))
	nb.classify_tweets_prob_export()