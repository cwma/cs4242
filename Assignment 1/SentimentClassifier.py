from NaiveBayesClassifier import NaiveBayesTweetClassifier
from AfinnClassifier import AfinnTweetClassifier
import Tweets
import operator
import json

class SentimentClassifier():

	_TAG_COUNT = 3
	_WEIGHTS =  [0.5310763280288978, 0.48324722636254674, 0.649798028298188, 0.4689236719711023, 0.5167527736374533, 0.4578680242077072]
	_CLASSIFIERS = [NaiveBayesTweetClassifier, AfinnTweetClassifier]

	def __init__(self):
		self._classifiers = []
		for classifier in self._CLASSIFIERS:
			self._classifiers.append(classifier())

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
		classifer_and_weights = zip(self._classifiers, [self._WEIGHTS[i:i+self._TAG_COUNT] for i in range(0, len(self._WEIGHTS)*self._TAG_COUNT,self._TAG_COUNT)])
		for classifier, (pos_weight, neg_weight, neu_weight) in classifer_and_weights:
			result = classifier.classify_prob(tweet)
			result = self._adjust_scores(result, pos_weight, neg_weight, neu_weight)
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
	results = sc.classify_tweets()
	print("Correct: {0}, Wrong: {1}, Total: {2}".format(*results))
	print("Percentage: {0}".format(results[0] / results[2]))
	#sc.classify_tweets_export()