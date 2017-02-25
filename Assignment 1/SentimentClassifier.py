import json
import operator

import Tweets
from sklearn import metrics
from AfinnClassifier import AfinnTweetClassifier
from KnnClassifier import KnnClassifier
from NaiveBayesClassifier import NaiveBayesTweetClassifier
from RandomForestClassifier import RFClassifier
from SVMClassifier import SVMClassifier


class SentimentClassifier():

    _TAG_COUNT = 3
    _WEIGHTS = [0.27540605773653415, 0.40103550689709005, 0.4664623775567014,   # Naive Bayes
                0.29870883549711025, 0.09131176566838402, 0.3654255947842784,   # Afinn text count
                0.16499504457243164, 0.22070883907894973, 0.11490582349642328,  # K-Nearest Neighbours
                0.16259719819330132, 0.195022295088288, 0.040773068557682646,   # Random Forest
                0.0982928640006226, 0.051477733463735596, 0.012433135604914351] # Support Vector Machine
    #           Positive            Negative              Neutral

    _CLASSIFIERS = [NaiveBayesTweetClassifier, AfinnTweetClassifier, KnnClassifier, RFClassifier, SVMClassifier]

    def __init__(self):
        self._classifiers = []
        for classifier in self._CLASSIFIERS:
            print("initializing {0}".format(classifier.__name__))
            self._classifiers.append(classifier())
        print("classifiers loaded")
        self.classifer_and_weights = list(zip(self._classifiers, [self._WEIGHTS[i:i + self._TAG_COUNT] for i in
                                                                  range(0, len(self._WEIGHTS) * self._TAG_COUNT,
                                                                        self._TAG_COUNT)]))

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

    def _metrics(self, results):
        print(metrics.classification_report(results['actual'], results['prediction']))

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
    # sc.classify_tweets_export()
