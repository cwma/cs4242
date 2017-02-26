#!/usr/bin/python

import json
import re
import os
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 
import nltk
import pandas as pd
import string
from collections import Counter
import Tweets

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline

class KnnClassifier2():
    
    def __init__(self):
        self.training_path = "dataset/training.json"
        self.dev_path = "dataset/development.json"
        self.neg_words_path = "dataset/lexicon/neg.txt"
        self.pos_words_path = "dataset/lexicon/pos.txt"
        self.tweets_path = "dataset/tweets/"
        self._test = {}
        self._load_words()
        self.train_tweets = self._extract_tweet(self.training_path)
        self.test_tweets = self._extract_tweet(self.dev_path)
        
        self.train = pd.DataFrame()
        self.train['tweet_id'] = list(map(lambda tweet: tweet[0], self.train_tweets))
        self.train['text'] = list(map(lambda tweet: self._remove_link(tweet[1]), self.train_tweets))
        self.train['sentiment'] = list(map(lambda tweet: tweet[2], self.train_tweets))
        self.train['tokens'] = list(map(lambda tweet: self._Preprocess(tweet[2]), self.train_tweets))
        self.train['neg_count'] = self.train['tokens'].apply(lambda tokens: self._get_NegativeScore(tokens))
        self.train['pos_count'] = self.train['tokens'].apply(lambda tokens: self._getPositiveScore(tokens))
        self.train['afinn'] = list(map(lambda tweet: tweet[3], self.train_tweets))
        self._model = None

    def _load_words(self):
        pos_file = open(self.pos_words_path, encoding='utf-8')
        neg_file = open(self.neg_words_path, encoding='utf-8')
        pos = [line.strip('\n') for line in pos_file.readlines()]
        neg = [line.strip('\n') for line in neg_file.readlines()]
        self.poswords = pos
        self.negwords = neg

    def _extract_tweet(self, file_path):
        tweets = Tweets.Tweets(file_path)
        return [(tweetid, tweet['text'], tweet['label'], tweet['afinn']) for (tweetid, tweet) in tweets.items()]
    
    def _get_NegativeScore(self, tokens):
        count = 0
        for word in tokens:
            if word in self.negwords:
                count = count + 1
        return count
    
    def _getPositiveScore(self, tokens):
        count = 0
        for word in tokens:
            if word in self.poswords:
                count = count + 1
        return count     
    
    def _remove_link(self, text):
        try:
            regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            r = re.sub(regex, "", text)
            return r
        except:
            return ''

    def _Preprocess(self, tweet):
        # Remove links
        regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        tweet = re.sub(regex, "", tweet)
        # Remove hashtags
        tweet = re.sub(r'#([^\s]+)', "", tweet)
        tokens = []
        # Tokenize
        tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
        tokens = tknzr.tokenize(tweet)
        # Remove punctuations and stopwords
        stop = set(stopwords.words('english') + list(string.punctuation) + list('...'))
        tokens = [term for term in tokens if term not in stop]
        # Remove words shorter than length 2
        tokens = [term for term in tokens if len(term) > 2]
        return tokens
    
    def _parse_tweets(self):
        test_tweets = self._extract_tweet(self.dev_path)
        test = pd.DataFrame()
        test['tweet_id'] = list(map(lambda tweet: tweet[0], test_tweets))
        test['text'] = list(map(lambda tweet: self._remove_link(tweet[1]), test_tweets))
        test['sentiment'] = list(map(lambda tweet: tweet[2], test_tweets))
        test['tokens'] = list(map(lambda tweet: self._Preprocess(tweet[1]), test_tweets))
        test['neg_count'] = test['tokens'].apply(lambda tokens: self._get_NegativeScore(tokens))
        test['pos_count'] = test['tokens'].apply(lambda tokens: self._getPositiveScore(tokens))          
        test['afinn'] = list(map(lambda tweet: tweet[3], test_tweets))
        return test
    
    def classify_all(self):
        test = self._parse_tweets()
        pipeline = Pipeline([('featurize', DataFrameMapper([('neg_count', None), ('pos_count', None), ('afinn', None)])), ('knn', KNeighborsClassifier())])
        #pipeline = Pipeline([('featurize', DataFrameMapper([('neg_count', None), ('pos_count', None)])), ('knn', KNeighborsClassifier())])
        X = self.train[self.train.columns.drop(['sentiment', 'tweet_id', 'text'])]
        y = self.train['sentiment']
        test['predict'] = pipeline.fit(X = X, y = y).predict(test)
        prob = pipeline.fit(X = X, y = y).predict_proba(test)
        result = [{'positive':prob[i][2], 'negative':prob[i][0], 'neutral':prob[i][1]} for i in range(len(prob))]
        print(metrics.classification_report(test['sentiment'], test['predict']))
        print(metrics.confusion_matrix(test['sentiment'], test['predict']))
        return result

    def classify_prob(self, tweet):
        test = pd.DataFrame()
        test['tweet_id'] = [tweet['id']]
        test['text'] = [self._remove_link(tweet['text'])]
        test['sentiment'] = [tweet['label']]
        test['tokens'] = [self._Preprocess(tweet['text'])]
        test['neg_count'] = test['tokens'].apply(lambda tokens: self._get_NegativeScore(tokens))
        test['pos_count'] = test['tokens'].apply(lambda tokens: self._getPositiveScore(tokens))   
        test['afinn'] = tweet['afinn']
        if self._model is None:
            pipeline = Pipeline([('featurize', DataFrameMapper([('neg_count', None), ('pos_count', None), ('afinn', None)])), ('knn', KNeighborsClassifier())])
            #pipeline = Pipeline([('featurize', DataFrameMapper([('neg_count', None), ('pos_count', None)])), ('knn', KNeighborsClassifier())])
            X = self.train[self.train.columns.drop(['sentiment', 'tweet_id', 'text'])]
            y = self.train['sentiment']
            self._model = pipeline.fit(X=X, y=y)
        prob = self._model.predict_proba(test)
        return {'positive': prob[0][2], 'negative': prob[0][0], 'neutral': prob[0][1]}
           
    def classify_export(self):
        test = self._parse_tweets()
        pipeline = Pipeline([('featurize', DataFrameMapper([('neg_count', None), ('pos_count', None), ('afinn', None)])), ('knn', KNeighborsClassifier())])
        #pipeline = Pipeline([('featurize', DataFrameMapper([('neg_count', None), ('pos_count', None)])), ('knn', KNeighborsClassifier())])
        X = self.train[self.train.columns.drop(['sentiment', 'tweet_id', 'text'])]
        y = self.train['sentiment']
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

    knn = KnnClassifier2()
    #prob = knn.classify_all()
    knn.classify_tweets_prob_export()
    #              precision    recall  f1-score   support

    #    negative       0.63      0.89      0.74        95
    #     neutral       0.68      0.49      0.57       158
    #    positive       0.85      0.87      0.86       240

    # avg / total       0.75      0.75      0.74       493
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        