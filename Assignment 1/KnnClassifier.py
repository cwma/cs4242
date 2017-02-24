#!/usr/bin/python

import json
import re
import os
from afinn import Afinn
from nltk.corpus import stopwords 
from nltk.tokenize import wordpunct_tokenize, word_tokenize
import nltk
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline

class KnnClassifier():
    
    def __init__(self):
        self._test = {}
        
    def _extract_tweet(self, file_path):
        with open(file_path, 'r+') as f1:
            index = json.loads(f1.read())
            f1.close()
    
        results = []
        for filename in os.listdir(tweets_path):
            if filename.endswith('.json'):
                f2 = open(tweets_path + filename, 'r+', encoding='utf-8')
                t = json.loads(f2.read())
                try:
                    r = (t['id_str'],t['text'],index[t['id_str']]['label'])
                    results.append(r)
                except Exception as e:
                    continue
                f2.close()
        return results

    def _remove_link(self, text):
        try:
            regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            r = re.sub(regex, "", text)
            return r
        except:
            return ''
    
    def _get_afinn_score(self, text):
        afinn = Afinn(emoticons=True)
        return afinn.score(text)
    
    def _parse_tweets(self):
        train_tweets = []
        test_tweets = []
        train_tweets = self._extract_tweet(training_path)
        test_tweets = self._extract_tweet(dev_path)
        
        train = pd.DataFrame()
        train['tweet_id'] = list(map(lambda tweet: tweet[0], train_tweets))
        train['text'] = list(map(lambda tweet: self._remove_link(tweet[1]), train_tweets))
        train['sentiment'] = list(map(lambda tweet: tweet[2], train_tweets))
        train['afinn'] = train['text'].apply(lambda tweet: self._get_afinn_score(tweet))
    
        test = pd.DataFrame()
        test['tweet_id'] = list(map(lambda tweet: tweet[0], test_tweets))
        test['text'] = list(map(lambda tweet: self._remove_link(tweet[1]), test_tweets))
        test['sentiment'] = list(map(lambda tweet: tweet[2], test_tweets))
        test['afinn'] = test['text'].apply(lambda tweet: self._get_afinn_score(tweet))
        return train, test
    
    def classify(self):
        train, test = self._parse_tweets()
        pipeline = Pipeline([('featurize', DataFrameMapper([('afinn', None)])), ('knn', KNeighborsClassifier())])
        X = train[train.columns.drop(['sentiment', 'tweet_id', 'text'])]
        y = train['sentiment']
        test['predict'] = pipeline.fit(X = X, y = y).predict(test)
        prob = pipeline.fit(X = X, y = y).predict_proba(test)
        result = [{'positive':prob[i][2], 'negative':prob[i][0], 'neutral':prob[i][1]} for i in range(len(prob))]
        print(metrics.classification_report(test['sentiment'], test['predict']))
        print(metrics.confusion_matrix(test['sentiment'], test['predict']))
        return result
    
    def classify_export(self):
        train, test = self._parse_tweets()
        pipeline = Pipeline([('featurize', DataFrameMapper([('afinn', None)])), ('knn', KNeighborsClassifier())])
        X = train[train.columns.drop(['sentiment', 'tweet_id', 'text'])]
        y = train['sentiment']
        test['predict'] = pipeline.fit(X = X, y = y).predict(test)
        prob = pipeline.fit(X = X, y = y).predict_proba(test)
        result = [{'positive':prob[i][2], 'negative':prob[i][0], 'neutral':prob[i][1]} for i in range(len(prob))]
        print(metrics.classification_report(test['sentiment'], test['predict']))
        print(metrics.confusion_matrix(test['sentiment'], test['predict']))
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

    training_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/training.json"
    dev_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/development.json"
    neg_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/lexicon/neg.txt"
    pos_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/lexicon/pos.txt"
    tweets_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/tweets/"    
    knn = KnnClassifier()
    #prob = knn.classify()
    knn.classify_tweets_prob_export()

    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        