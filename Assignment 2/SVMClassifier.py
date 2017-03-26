import json
import os
import pickle

import numpy
from nltk.classify import SklearnClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn import metrics
from sklearn.svm import SVC

import Tweet


class SvmCascadeClassifier():
    def __init__(self, dataset, k, user_followers=True, users_reachable=True, average_time=True, time_to_k=True):
        self.k = k
        self._twtokenize = TweetTokenizer(strip_handles=True)
        self._dataset = dataset
        self._user_followers = user_followers
        self._users_reachable = users_reachable
        self._average_time = average_time
        self._time_to_k = time_to_k
        self._stopwords = stopwords.words('english')
        self._stemmer = PorterStemmer()
        self._f_count = []
        self._r_count = []
        self._rt_count = []
        self._avg = []
        self._time = []
        self._train()

    def _tokenize(self, tweet_text):
        return [self._stemmer.stem(token) for token in self._twtokenize.tokenize(tweet_text) if
                token not in self._stopwords]

    def _sorted_cascade_nodes(self, cascade):
        nodes = cascade['cascade']
        cascade_nodes = [(int(key), nodes[key]) for key in nodes.keys()]
        return sorted(cascade_nodes, key=lambda x: x[0])

    def _tweet_length_feature(self, cascade):
        length = len(cascade['root_tweet']['text'])
        return int(length)

    def _user_followers_feature(self, cascade):
        followers = cascade['root_tweet']['user']['followers_count']
        self._f_count.append(followers)
        return int(followers)

    def _users_reachable_feature(self, nodes):
        reachable = 0
        for kth, node in zip(range(self.k + 1), nodes):
            reachable += node[1]['user_followees_count']
        self._r_count.append(reachable)
        return int(reachable)

    def _average_time_feature(self, nodes):
        timestamp = [int(node[1]['created_at']) for kth, node in zip(range(self.k + 1), nodes)]
        average = (sum(numpy.diff(timestamp)) / float(len(timestamp))) / 1000
        self._avg.append(average)
        return int(average)

    def _users_retweet_feature(self, cascade):
        retweets = cascade['root_tweet']['retweet_count']
        self._rt_count.append(retweets)
        return int(retweets)

    def _time_to_k_feature(self, nodes):
        first = int(nodes[0][1]['created_at'])
        kth = int(list(zip(range(self.k + 1), nodes))[-1][1][1]['created_at'])
        diff = (kth - first) / 1000
        self._time.append(diff)
        return int(diff)

    def _extract_features(self, cascade):
        if cascade['root_tweet']['lang'] == 'en':
            tweet_tokens = self._tokenize(cascade['root_tweet']['text'])
            features = {"contains({0})".format(token): True for token in tweet_tokens}
        else:
            features = {}

        features[self._tweet_length_feature(cascade)] = True
        # features[self._users_retweet_feature(cascade)] = True

        if self._user_followers:
            features[self._user_followers_feature(cascade)] = True

        cascade_nodes = self._sorted_cascade_nodes(cascade)

        if self._users_reachable:
            features[self._users_reachable_feature(cascade_nodes)] = True
        if self._average_time:
            features[self._average_time_feature(cascade_nodes)] = True
        if self._time_to_k:
            features[self._time_to_k_feature(cascade_nodes)] = True

        return features

    def _train(self):
        pickle_filename = "{0}.pickle".format(self.__class__.__name__)
        if os.path.isfile(pickle_filename):
            with open(pickle_filename, "rb") as classifier_f:
                self._classifier = pickle.load(classifier_f)
            classifier_f.close()
        else:
            train_set = [(self._extract_features(cascade), cascade['label']) for cascade in self._dataset]
            # pipeline = Pipeline([('tfidf', TfidfTransformer()),
            #                      ('chi2', SelectKBest(chi2, k=1000)),
            #                      ('svc', SVC(kernel='linear', probability=True))])
            self._classifier = SklearnClassifier(SVC(kernel='linear', probability=True), sparse=False).train(train_set)

            with open(pickle_filename, "wb") as save_classifier:
                pickle.dump(self._classifier, save_classifier)
            save_classifier.close()

    def classify(self, cascade):
        features = self._extract_features(cascade)
        return self._classifier.classify(features)

    def classify_prob(self, cascade):
        features = self._extract_features(cascade)
        result = self._classifier.prob_classify(features)
        return {"positive": result.prob(True), "negative": result.prob(False)}

    def _metrics(self, results):
        print(metrics.classification_report(results['actual'], results['prediction']))

    def classify_cascades(self, test_dataset):
        results = {"prediction": [], "actual": []}

        for cascade in test_dataset:
            result = self.classify(cascade)
            actual = cascade['label']
            results["prediction"].append(result)
            results["actual"].append(actual)

        self._metrics(results)

        print("Average: {0}, Median: {1}, Std: {2}".format(numpy.average(self._f_count), numpy.median(self._f_count),
                                                           numpy.std(self._f_count)))
        print("Average: {0}, Median: {1}, Std: {2}".format(numpy.average(self._r_count), numpy.median(self._r_count),
                                                           numpy.std(self._r_count)))
        print("Average: {0}, Median: {1}, Std: {2}".format(numpy.average(self._avg), numpy.median(self._avg),
                                                           numpy.std(self._avg)))
        print("Average: {0}, Median: {1}, Std: {2}".format(numpy.average(self._time), numpy.median(self._time),
                                                           numpy.std(self._time)))

    def classify_cascades_prob_export(self, test_dataset):
        export = "dataset/" + self.__class__.__name__ + "_results.json"
        results = {}

        for cascade in test_dataset:
            results[cascade['url']] = self.classify_prob(cascade)

        export_file = open(export, 'w')
        export_file.write(json.dumps(results))


if __name__ == '__main__':
    train_x, test_y = Tweet.get_flattened_data('dataset/k4/training.json', 'dataset/k4/testing.json',
                                               'dataset/k4/root_tweet.json', 4)

    svm = SvmCascadeClassifier(train_x, 2)
    results = svm.classify_cascades(test_y)
    svm.classify_cascades_prob_export(test_y)

    #              precision    recall  f1-score   support

    #       False       0.83      0.82      0.82      1022
    #        True       0.57      0.59      0.58       421

    # avg / total       0.75      0.75      0.75      1443
