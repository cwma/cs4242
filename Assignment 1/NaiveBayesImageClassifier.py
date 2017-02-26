from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.stem.porter import PorterStemmer
import nltk.sentiment.util as sentiment_utils
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from Classifier import Classifier
import Tweets
import json

class NaiveBayesImageClassifier(Classifier):

	def __init__(self):
		self._tknzr = TweetTokenizer(strip_handles=True)
		self._tweets = Tweets.TrainTweets()
		self._stopwords = stopwords.words('english')
		self._stemmer = PorterStemmer()
		self._train()

	def _tokenize(self, tweet_text):
		return [self._stemmer.stem(token) for token in self._tknzr.tokenize(tweet_text) if token not in self._stopwords]

	def _extract_features(self, tweet):
		desc_feats = self._tokenize(tweet['vision']['description'])
		tag_feats = tweet['vision']['tags']
		feats = {"vision_tag({0})".format(token): True for token in tag_feats}
		feats.update({"vision_desc({0})".format(token): True for token in desc_feats})
		return feats

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

	nb = NaiveBayesImageClassifier()
	results = nb.classify_tweets()
	#nb.classify_tweets_prob_export()
	#              precision    recall  f1-score   support

	#    negative       0.49      0.54      0.51       285
	#     neutral       0.53      0.42      0.47       474
	#    positive       0.63      0.69      0.66       720

	# avg / total       0.57      0.57      0.57      1479