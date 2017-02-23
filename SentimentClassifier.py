from NaiveBayesClassifier import NaiveBayesTweetClassifier
from AfinnClassifier import AfinnTweetClassifier
import Tweets
import operator
import json

class SentimentClassifier():

	_WEIGHTS = [0.5, 0.5]
	_CLASSIFIERS = [NaiveBayesTweetClassifier, AfinnTweetClassifier]

	def __init__(self):
		self._classifiers = []
		for classifier in self._CLASSIFIERS:
			self._classifiers.append(classifier())

	def _adjust_scores(self, result, weight):
		for key, value in result.items():
			result[key] = value * weight
		return result

	def _calculate_final_score(self, results):
		final_score = {"positive": 0, "negative": 0, "neutral": 0}
		for result in results:
			for key, value in result.items():
				final_score[key] += value
		return final_score

	def classify(self, tweet):
		results = []
		for classifier, weight in zip(self._classifiers, self._WEIGHTS):
			result = classifier.classify_prob(tweet)
			result = self._adjust_scores(result, weight)
			results.append(result)
		final_score = self._calculate_final_score(results)
		final_score = sorted(final_score.items(), key=operator.itemgetter(1))
		return final_score[-1][0]

	def classify_tweets(self, test_tweets=Tweets.TestTweets()):
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
	# results = sc.classify_tweets()
	# print("Correct: {0}, Wrong: {1}, Total: {2}".format(*results))
	# print("Percentage: {0}".format(results[0] / results[2]))
	# sc.classify_tweets_export()