import json
import operator

import Tweets
from sklearn import metrics
from AfinnClassifier import AfinnTweetClassifier
from KnnClassifier import KnnClassifier
from KnnClassifier_v2 import KnnClassifier2
from NaiveBayesClassifier import NaiveBayesTweetClassifier
from RandomForestClassifier import RFClassifier
from NaiveBayesImageClassifier import NaiveBayesImageClassifier
from SVMClassifier import SVMClassifier


class SentimentClassifier():

    _TAG_COUNT = 3
                #      Positive            Negative              Neutral
    _WEIGHTS = [0.21052403668384623, 0.3532195357238242, 0.36431434614209679,     # Naive Bayes
                0.15185261109972592, 0.065680111586002232, 0.22519304816816038,   # Afinn text count
                0.16145236935404356, 0.17113904683180287, 0.17595262849843762,    # K-Nearest Neighbours  
                0.090646693927382241, 0.094282696845565186, 0.070369178842033198, # Random Forest
                0.088422387032625671, 0.12550730043363162, 0.083583114106813294,  # Support Vector Machine
                0.29710190190237651, 0.19017130857917397, 0.13637946165276033]    # Naive Bayes Image Vision
                #      Positive            Negative              Neutral
    _CLASSIFIERS = [NaiveBayesTweetClassifier, AfinnTweetClassifier, KnnClassifier2, RFClassifier, SVMClassifier, NaiveBayesImageClassifier]

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
        results = {"prediction": [], "actual": []}
        for tweet_id, tweet in test_tweets.items():
            result = self.classify(tweet)
            actual = tweet['label']
            tweet_results[tweet_id] = {"photo": tweet["photo"], "text": tweet["filename"], "predicted_label": result}
            results["prediction"].append(result)
            results["actual"].append(actual)
        export_file = open(export, 'w')
        export_file.write(json.dumps(tweet_results))
        self._metrics(results)

if __name__ == '__main__':
    sc = SentimentClassifier()
    sc.classify_tweets_export(test_tweets=Tweets.InputPrompt())
    # results = sc.classify_tweets(Tweets.DevTweets())

    #              precision    recall  f1-score   support

    #    negative       0.77      0.88      0.82        95
    #     neutral       0.75      0.68      0.71       158
    #    positive       0.86      0.87      0.87       240

    # avg / total       0.81      0.81      0.81       493

    # results = sc.classify_tweets(Tweets.TestTweets())

    # sc.classify_tweets_export()
    #                  precision    recall  f1-score   support

    #    negative       0.78      0.90      0.84       285
    #     neutral       0.76      0.67      0.71       474
    #    positive       0.86      0.88      0.87       720

    # avg / total       0.81      0.82      0.81      1479