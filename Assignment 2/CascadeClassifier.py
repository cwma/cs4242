import json
import operator

import Tweet
from sklearn import metrics
from NaiveBayesClassifier import NaiveBayesCascadeClassifier


class CascadeClassifier():

    _TAG_COUNT = 2
                #      Positive            Negative
    _WEIGHTS = [0.50000000000000000, 0.50000000000000000,    
                0.50000000000000000, 0.50000000000000000]
    _CLASSIFIERS = [NaiveBayesCascadeClassifier, NaiveBayesCascadeClassifier]

    def __init__(self, train_dataset, k):
        self._classifiers = []
        for classifier in self._CLASSIFIERS:
            print("initializing {0}".format(classifier.__name__))
            self._classifiers.append(classifier(train_dataset, k))
        print("classifiers loaded")
        self.classifer_and_weights = list(zip(self._classifiers, [self._WEIGHTS[i:i + self._TAG_COUNT] for i in
                                                                  range(0, len(self._WEIGHTS) * self._TAG_COUNT,
                                                                        self._TAG_COUNT)]))

    def _adjust_scores(self, result, pos_weight, neg_weight):
        result["positive"] = result["positive"] * pos_weight
        result["negative"] = result["negative"] * neg_weight
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
        for classifier, (pos_weight, neg_weight) in self.classifer_and_weights:
            result = classifier.classify_prob(tweet)
            result = self._adjust_scores(result, pos_weight, neg_weight)
            results.append(result)
        final_score = self._calculate_final_score(results)
        final_score = sorted(final_score.items(), key=operator.itemgetter(1))
        return final_score[-1][0]

    def classify_tweets(self, test_dataset):
        results = {"prediction": [], "actual": []}
        for cascade in test_dataset:
            result = self.classify(cascade)
            actual = cascade['label']
            results["prediction"].append(result)
            results["actual"].append(actual)
        self._metrics(results)

    def classify_tweets_export(self, test_dataset, export="testing_online_prediction.json"):
        cascade_results = {}
        results = {"prediction": [], "actual": []}
        for cascade in test_dataset:
            result = self.classify(cascade)
            actual = cascade['label'] and "positive" or "negative"
            cascade_results[cascade['url']] = cascade['cascade']
            cascade_results[cascade['url']].update({"predicted_label" : result is "positive" and "+1" or "-1"})
            results["prediction"].append(result)
            results["actual"].append(actual)
        export_file = open(export, 'w')
        export_file.write(json.dumps(cascade_results))
        self._metrics(results)

if __name__ == '__main__':
    train_dataset, test_dataset = Tweet.get_flattened_data('dataset/k4/training.json', 'dataset/k4/testing.json', 'dataset/k4/root_tweet.json', 4)
    cc = CascadeClassifier(train_dataset, 4)
    cc.classify_tweets_export(test_dataset)
