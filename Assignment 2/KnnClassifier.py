#!/usr/bin/python

import pickle
import json
import re
import os
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords 
import nltk
import pandas as pd
import string
from collections import Counter
from afinn import Afinn
import Tweet
from ttp import ttp     # pip install twitter-text-python
#https://github.com/edburnett/twitter-text-python
import datetime
import numpy

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline

class KnnClassifier():
    
    def __init__(self, trainset, k):
        self.k = k
        self.train_tweets = trainset

    # Return the integer id of the root of the cascade (because 1 sometimes may be missing)
    def _sort_cascade(self, cascade):
        tweets = cascade.keys()
        return sorted(map(int, tweets)) 

    # Extract only the first k tweets in the cascade
    def _load_tweets(self, data, k):
        output = []
        for entry in data:
            cascade = entry['cascade']
            sort = self._sort_cascade(cascade)
            output.append({str(sort[i]):cascade[str(sort[i])] for i in range(0, k > len(cascade) and len(cascade) or k)})
        return output
        
    def _extract_urls(self, data):
        return [item['url'] for item in data]
    
    # Get the label viral/non-viral (True or False) for a cascade.
    def _get_label(self, data):
        return [d['label'] for d in data]
    
    def _extract_afinn(self, data):
        results = []
        for item in data:
            tw_content = item['root_tweet']
            if 'en' in tw_content['lang'] and not None:
                afinn = Afinn(emoticons=True)
                text = ' '.join(map(str, self._Preprocess(tw_content['text'])))
                results.append(afinn.score(text))
            else:
                results.append(0)    
        return results
    
    def _extract_wordcount(self, data):
        results = []
        for item in data:  
            tw_content = item['root_tweet']
            if 'en' in tw_content['lang'] and not None:
                results.append(len(self._Preprocess(tw_content['text'])))
            else:
                results.append(0)
        return results

    def _getTermCounter(self, method, data):
        p = ttp.Parser()
        count_all = Counter()
        results = []
        for item in data:  
            tw_content = item['root_tweet']
            if 'en' in tw_content['lang'] and not None:
                r = p.parse(tw_content['text'])
                count_all.update(getattr(r, method))
                results.append(sum(count_all.values()))
            else:
                results.append(0)
        return results
    
    # Extract the number of followees of the root of the tree
    def _extract_followees(self, data):
        return [item['cascade'][item['cascade_root']]['user_followees_count'] for item in data]
    
    def _extract_avgtime(self, data):
        results = []
        for cascade in data:
            t = []
            previous = datetime.datetime.fromtimestamp(int(cascade['1']['created_at'])/1000)
            for i in range(2, len(cascade)+1):
                tweet_timestamp = cascade[str(i)]['created_at']
                current = datetime.datetime.fromtimestamp(int(tweet_timestamp)/1000)
                t.append(current - previous)
                previous = current
            avgtime = numpy.mean(t)
            results.append(avgtime.seconds)
        return results
    
    def _extract_timeToK(self, data):
        results = []
        for cascade in data:
            root_time = datetime.datetime.fromtimestamp(int(cascade['1']['created_at'])/1000)
            k_time = datetime.datetime.fromtimestamp(int(cascade[str(len(cascade))]['created_at'])/1000)
            interval = k_time - root_time
            results.append(interval.seconds)
        return results

    def _extract_numIsolated(self, data):
        results = []
        for cascade in data:
            count = 0
            for i in cascade:
                if 'in_reply_to' in cascade[str(i)].keys():
                    if not cascade[str(i)]['in_reply_to']:
                        count = count + 1
                else:
                    count = count + 1
            results.append(count)
        return results

    def _extract_numEdges(self, data):
        results = []
        for cascade in data:
            count = 0
            for i in cascade:
                if 'in_reply_to' in cascade[str(i)].keys():
                    if cascade[str(i)]['in_reply_to'] != -1:
                        count = count + len(cascade[str(i)]['in_reply_to'])
            results.append(count)
        return results
    
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
    
    def _train(self):
        #k = self.set_k(param)
        train_parse = self._load_tweets(self.train_tweets,self.k)        
        dataframe = pd.DataFrame()
        dataframe['urls'] = self._extract_urls(self.train_tweets)
        dataframe['labels'] = self._get_label(self.train_tweets)
        dataframe['afinn'] = self._extract_afinn(self.train_tweets)
        dataframe['wordcount'] = self._extract_wordcount(self.train_tweets)
        dataframe['num_hashtags'] = self._getTermCounter('tags', self.train_tweets)
        dataframe['num_urls'] = self._getTermCounter('urls', self.train_tweets)
        dataframe['num_mentions'] = self._getTermCounter('users', self.train_tweets)
        dataframe['num_followee'] = self._extract_followees(self.train_tweets)
        dataframe['avg_time'] = self._extract_avgtime(train_parse)
        dataframe['time_to_k'] = self._extract_timeToK(train_parse)
        dataframe['num_isolated'] = self._extract_numIsolated(train_parse)
        dataframe['num_edges'] = self._extract_numEdges(train_parse)
        pipeline = Pipeline([('featurize', DataFrameMapper([('afinn', None), ('wordcount', None), ('num_hashtags', None), ('num_urls', None), ('num_mentions', None), ('num_followee', None), ('avg_time', None), ('time_to_k', None), ('num_isolated', None), ('num_edges', None)])), ('knn', KNeighborsClassifier())])
        X = dataframe[dataframe.columns.drop(['urls', 'labels'])]
        y = dataframe['labels']
        classifier = pipeline.fit(X = X, y = y)
        return classifier

    def _get_train(self):
        pickle_filename = "{0}.pickle".format(self.__class__.__name__)
        if os.path.isfile(pickle_filename):
            with open(pickle_filename, "rb") as classifier_f:
                classifier = pickle.load(classifier_f)
            classifier_f.close()
        else:
            classifier = self._train()

            with open(pickle_filename, "wb") as save_classifier:
                pickle.dump(classifier, save_classifier)
            save_classifier.close()
        return classifier
    
    def _test(self, test_dataset):
        test_parse = self._load_tweets(test_dataset, self.k)        
        dataframe = pd.DataFrame()
        dataframe['urls'] = self._extract_urls(test_dataset)
        dataframe['labels'] = self._get_label(test_dataset)
        dataframe['afinn'] = self._extract_afinn(test_dataset)
        dataframe['wordcount'] = self._extract_wordcount(test_dataset)
        dataframe['num_hashtags'] = self._getTermCounter('tags', test_dataset)
        dataframe['num_urls'] = self._getTermCounter('urls', test_dataset)
        dataframe['num_mentions'] = self._getTermCounter('users', test_dataset)
        dataframe['num_followee'] = self._extract_followees(test_dataset)
        dataframe['avg_time'] = self._extract_avgtime(test_parse)
        dataframe['time_to_k'] = self._extract_timeToK(test_parse)
        dataframe['num_isolated'] = self._extract_numIsolated(test_parse)
        dataframe['num_edges'] = self._extract_numEdges(test_parse)
        return dataframe    
    
    def _parse_cascade(self, cascade):
        cascade_parse = self._load_tweets([cascade], self.k)
        dataframe = pd.DataFrame()
        dataframe['urls'] = self._extract_urls([cascade])
        dataframe['afinn'] = self._extract_afinn([cascade])
        dataframe['wordcount'] = self._extract_wordcount([cascade])
        dataframe['num_hashtags'] = self._getTermCounter('tags', [cascade])
        dataframe['num_urls'] = self._getTermCounter('urls', [cascade])
        dataframe['num_mentions'] = self._getTermCounter('users', [cascade])
        dataframe['num_followee'] = self._extract_followees([cascade])
        dataframe['avg_time'] = self._extract_avgtime(cascade_parse)
        dataframe['time_to_k'] = self._extract_timeToK(cascade_parse)
        dataframe['num_isolated'] = self._extract_numIsolated(cascade_parse)
        dataframe['num_edges'] = self._extract_numEdges(cascade_parse)
        return dataframe
        
    def classify_prob(self, cascade):
        features = self._parse_cascade(cascade)
        classifier = self._get_train()
        prob = classifier.predict_proba(features)
        return {"positive": prob[0][1], "negative": prob[0][0]}
    
    def dev(self, test_dataset):
        #return self._train(), self._test()
        for cascade in test_dataset:
            result = self.classify_prob(cascade)
            print(result)
    
    def classify_all(self, test_dataset):
        test = self._test(test_dataset)
        classifier = self._get_train()
        test['predict'] = classifier.predict(test)
        prob = classifier.predict_proba(test)
        result = [{'positive':prob[i][1], 'negative':prob[i][0]} for i in range(len(prob))]
        print(metrics.classification_report(test['labels'], test['predict']))
        print(metrics.confusion_matrix(test['labels'], test['predict']))
        return result
    
    def classify_export(self, test_dataset):
        test = self._test(test_dataset)
        classifier = self._get_train()
        test['predict'] = classifier.predict(test)
        prob = classifier.predict_proba(test)
        result = [{'positive':prob[i][1], 'negative':prob[i][0]} for i in range(len(prob))]
        print(metrics.classification_report(test['labels'], test['predict']))
        print(metrics.confusion_matrix(test['labels'], test['predict']))
        return result, test
    
    def classify_tweets_prob_export(self, test_dataset):
        result, test = self.classify_export(test_dataset)
        #export = "dataset/" + self.__class__.__name__ + "_results.json"
        export = "dataset/" + self.__class__.__name__ + "_results.json"
        tweet_results = {}
        for index, row in test.iterrows():
            tweet_results[row['urls']] = result[index]
        export_file = open(export, 'w')
        export_file.write(json.dumps(tweet_results))

if __name__ == "__main__":

    #train_dataset, test_dataset = Tweet.get_flattened_data('dataset/k2/training.json', 'dataset/k2/testing.json', 'dataset/k2/root_tweet.json', 2)
    train_dataset, test_dataset = Tweet.get_flattened_data('dataset/k4/training.json', 'dataset/k4/testing.json', 'dataset/k4/root_tweet.json', 4)
    knn = KnnClassifier(train_dataset, 4)
    #knn.dev(test_dataset)
    prob = knn.classify_all(test_dataset)
    #knn.classify_tweets_prob_export(test_dataset)
    
    #             precision    recall  f1-score   support

    #  False       0.81      0.89      0.85      1022
    #   True       0.65      0.48      0.55       421

#avg / total       0.76      0.77      0.76      1443