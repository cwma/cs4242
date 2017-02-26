import Tweets
import json
from sklearn import metrics

class Classifier():

    def _metrics(self, results):
        print(metrics.classification_report(results['actual'], results['prediction']))

    def classify_tweets(self, test_tweets=Tweets.TestTweets()):
        correct, wrong, total = (0, 0, 0)
        results = {"prediction": [], "actual": []}
        for tweet_id, tweet in test_tweets.items():
            total += 1
            result = self.classify(tweet)
            actual = tweet['label']
            if result == actual:
                correct += 1
            else:
                wrong += 1
            results["prediction"].append(result)
            results["actual"].append(actual)
        self._metrics(results)
        return (correct, wrong, total)

    def classify_tweets_prob_export(self, test_tweets=Tweets.DevTweets()):
        export = "dataset/" + self.__class__.__name__ + "_results.json"
        tweet_results = {}
        for tweet_id, tweet in test_tweets.items():
            result = self.classify_prob(tweet)
            tweet_results[tweet_id] = result
        export_file = open(export, 'w')
        export_file.write(json.dumps(tweet_results))
