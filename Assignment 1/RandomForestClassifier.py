#!/usr/bin/python
import Tweets
import json
import os
import re

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper


class RFClassifier():
    def __init__(self):
        self.training_path = "dataset/training.json"
        self.dev_path = "dataset/development.json"
        self.neg_path = "dataset/lexicon/neg.txt"
        self.pos_path = "dataset/lexicon/pos.txt"
        self.tweets_path = "dataset/tweets/"
        self._test = {}
        self.train_tweets = self._extract_tweet(self.training_path)
        self.test_tweets = self._extract_tweet(self.dev_path)

        self.train = pd.DataFrame()
        self.train['tweet_id'] = list(map(lambda tweet: tweet[0], self.train_tweets))
        self.train['text'] = list(map(lambda tweet: self._remove_link(tweet[1]), self.train_tweets))
        self.train['sentiment'] = list(map(lambda tweet: tweet[2], self.train_tweets))
        self.train['afinn'] = list(map(lambda tweet: tweet[3], self.train_tweets))
        self._model = None

    def _extract_tweet(self, file_path):
        tweets = Tweets.Tweets(file_path)
        return [(tweetid, tweet['text'], tweet['label'], tweet['afinn']) for (tweetid, tweet) in tweets.items()]

    def _remove_link(self, text):
        try:
            regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            r = re.sub(regex, "", text)
            return r
        except:
            return ''

    def _parse_tweets(self):
        test_tweets = self._extract_tweet(self.dev_path)

        test = pd.DataFrame()
        test['tweet_id'] = list(map(lambda tweet: tweet[0], test_tweets))
        test['text'] = list(map(lambda tweet: self._remove_link(tweet[1]), test_tweets))
        test['sentiment'] = list(map(lambda tweet: tweet[2], test_tweets))
        test['afinn'] = list(map(lambda tweet: tweet[3], test_tweets))
        return test

    def classify_all(self):
        test = self._parse_tweets()
        pipeline = Pipeline(
            [('featurize', DataFrameMapper([('afinn', None)])), ('rf', RandomForestClassifier(n_estimators=500))])
        X = self.train[self.train.columns.drop(['sentiment', 'tweet_id', 'text'])]
        y = self.train['sentiment']

        test['predict'] = pipeline.fit(X=X, y=y).predict(test)
        prob = pipeline.fit(X=X, y=y).predict_proba(test)
        result = [{'positive': prob[i][2], 'negative': prob[i][0], 'neutral': prob[i][1]} for i in range(len(prob))]

        print(metrics.classification_report(test['sentiment'], test['predict']))
        print(metrics.confusion_matrix(test['sentiment'], test['predict']))
        print(metrics.accuracy_score(test['sentiment'], test['predict']))
        print(metrics.precision_score(test['sentiment'], test['predict'], average='macro'))
        print(metrics.recall_score(test['sentiment'], test['predict'], average='macro'))
        print(metrics.f1_score(test['sentiment'], test['predict'], average='macro'))

        return result

    def classify_prob(self, tweet):
        test = pd.DataFrame()
        test['tweet_id'] = [tweet['id']]
        test['text'] = [self._remove_link(tweet['text'])]
        test['sentiment'] = [tweet['label']]
        test['afinn'] = tweet['afinn']
        if self._model is None:
            pipeline = Pipeline(
                [('featurize', DataFrameMapper([('afinn', None)])), ('rf', RandomForestClassifier(n_estimators=500))])
            X = self.train[self.train.columns.drop(['sentiment', 'tweet_id', 'text'])]
            y = self.train['sentiment']

            self._model = pipeline.fit(X=X, y=y)
        prob = self._model.predict_proba(test)
        return {'positive': prob[0][2], 'negative': prob[0][0], 'neutral': prob[0][1]}

    def classify_export(self):
        test = self._parse_tweets()

        pipeline = Pipeline(
            [('featurize', DataFrameMapper([('afinn', None)])), ('rf', RandomForestClassifier(n_estimators=500))])
        X = self.train[self.train.columns.drop(['sentiment', 'tweet_id', 'text'])]
        y = self.train['sentiment']

        test['predict'] = pipeline.fit(X=X, y=y).predict(test)
        prob = pipeline.fit(X=X, y=y).predict_proba(test)

        result = [{'positive': prob[i][2], 'negative': prob[i][0], 'neutral': prob[i][1]} for i in range(len(prob))]

        print(metrics.classification_report(test['sentiment'], test['predict']))
        print(metrics.confusion_matrix(test['sentiment'], test['predict']))
        print(metrics.accuracy_score(test['sentiment'], test['predict']))
        print(metrics.precision_score(test['sentiment'], test['predict'], average='macro'))
        print(metrics.recall_score(test['sentiment'], test['predict'], average='macro'))
        print(metrics.f1_score(test['sentiment'], test['predict'], average='macro'))

        return result, test

    def classify_tweets_prob_export(self):
        result, test = self.classify_export()
        export = "dataset/" + self.__class__.__name__ + "_results.json"
        tweet_results = {}

        for index, row in test.iterrows():
            tweet_results[row['tweet_id']] = result[index]

        export_file = open(export, 'w')
        export_file.write(json.dumps(tweet_results))


if __name__ == "__main__":
    rf = RFClassifier()
    #prob = rf.classify_all()
    rf.classify_tweets_prob_export()
    #              precision    recall  f1-score   support

    #    negative       0.66      0.81      0.73        95
    #     neutral       0.64      0.54      0.58       158
    #    positive       0.84      0.85      0.84       240

    # avg / total       0.74      0.74      0.74       493