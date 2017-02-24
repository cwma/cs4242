from NaiveBayesClassifier import NaiveBayesTweetClassifier
from AfinnClassifier import AfinnTweetClassifier
from KnnClassifier import KnnClassifier
import Tweets
import operator
import json

class SentimentClassifier():

	_TAG_COUNT = 3
	# _WEIGHTS = [0.7130202393652626, 0.5637485303341669, 0.36388186000539446,  		# Naive Bayes
	# 			0.2264637344711908, 0.04715857334284706, 0.6079813396761198,  		# simple Afinn text scoring
	# 			-0.06051602616354658, 0.38909289632298605, 0.028136800318485772] 	# knn
	_WEIGHTS = [0.7130202393652626, 0.5637485303341669, 0.36388186000539446,  		# Naive Bayes
				0.2264637344711908, 0.04715857334284706, 0.6079813396761198,  		# simple Afinn text scoring
				-0.06051602616354658, 0.38909289632298605, 0.028136800318485772] 	# knn

	_CLASSIFIERS = [NaiveBayesTweetClassifier, AfinnTweetClassifier, KnnClassifier]

	def __init__(self):
		self._classifiers = []
		for classifier in self._CLASSIFIERS:
			print("initializing {0}".format(classifier.__name__))
			self._classifiers.append(classifier())
		print("classifiers loaded")
		self.classifer_and_weights = list(zip(self._classifiers, [self._WEIGHTS[i:i+self._TAG_COUNT] for i in range(0, len(self._WEIGHTS)*self._TAG_COUNT,self._TAG_COUNT)]))

	def _adjust_scores(self, result, pos_weight, neg_weight, neu_weight):
		result["positive"] = result["positive"] * pos_weight
		result["negative"] = result["negative"] * neg_weight
		result["neutral"] = result["neutral"] * neu_weight
		return result

	def _calculate_final_score(self, results):
		final_score = {"positive": 0, "negative": 0, "neutral": 0}
		for result in results:
			for key, value in result.items():
				final_score[key] += value
		return final_score

	def classify(self, tweet):
		results = []
		# tldr combines [a,b,c] and [1,2,3,4,5,6,7,8,9] to [(a, (1,2,3)), (b,(4,5,6)), (c,(7,8,9))]
		# group classifiers and their respective weights together
		# weights are normalized based on tags, ie pos(a) + pos(b) = 1
		for classifier, (pos_weight, neg_weight, neu_weight) in self.classifer_and_weights:
			result = classifier.classify_prob(tweet)
			result = self._adjust_scores(result, pos_weight, neg_weight, neu_weight)
			results.append(result)
		final_score = self._calculate_final_score(results)
		final_score = sorted(final_score.items(), key=operator.itemgetter(1))
		return final_score[-1][0]

	def classify_tweets(self, test_tweets=Tweets.DevTweets()):
		correct, wrong, total = (0, 0, 0)
		for tweet_id, tweet in test_tweets.items():
			total += 1
			result = self.classify(tweet)
			actual = tweet['label']
			if result == actual:
				correct += 1
			else:
				wrong += 1
		return (correct, wrong, total)

	def classify_tweets_export(self, test_tweets=Tweets.TestTweets(), export="testing_online_prediction.json"):
		tweet_results = {}
		for tweet_id, tweet in test_tweets.items():
			result = self.classify(tweet)
			tweet_results[tweet_id] = {"photo": tweet["photo"], "text": tweet["filename"], "predicted_label": result}
		export_file = open(export, 'w')
		export_file.write(json.dumps(tweet_results))

if __name__ == '__main__':

	sc = SentimentClassifier()
	results = sc.classify_tweets()
	print("Correct: {0}, Wrong: {1}, Total: {2}".format(*results))
	print("Percentage: {0}".format(results[0] / results[2]))
	#sc.classify_tweets_export()