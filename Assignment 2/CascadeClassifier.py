import json
import operator

import Tweet
from sklearn import metrics
from NaiveBayesClassifier import NaiveBayesCascadeClassifier
from RandomForestClassifier import RandomForestCascadeClassifier
from SVMClassifier import SvmCascadeClassifier
from KnnClassifier import KnnClassifier


class CascadeClassifier():

    _TAG_COUNT = 2
                #      Positive            Negative
    _WEIGHTS =   [0.11412911847966915, 0.068686571202402555,
                  0.30587798598576543, 0.40258393686078336,
                  0.34015527541750068, 0.18918154971629764,
                  0.2398376201170648, 0.33954794222051654]

    _CLASSIFIERS = [NaiveBayesCascadeClassifier, RandomForestCascadeClassifier, KnnClassifier, SvmCascadeClassifier]

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
    train_dataset, test_dataset = Tweet.get_flattened_data('dataset/k2/training.json', 'dataset/k2/testing.json', 'dataset/k2/root_tweet.json', 2)
    cc = CascadeClassifier(train_dataset, 2)
    cc.classify_tweets_export(test_dataset)

# k = 2

#              precision    recall  f1-score   support

#    negative       0.86      0.97      0.91      1088
#    positive       0.85      0.54      0.66       365

# avg / total       0.86      0.86      0.85      1453

# k = 4

#              precision    recall  f1-score   support

#    negative       0.83      0.93      0.88      1022
#    positive       0.76      0.53      0.63       421

# avg / total       0.81      0.82      0.80      1443