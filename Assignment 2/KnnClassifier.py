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
    
    def __init__(self, trainset, testset, k):
        self.k = k
        self.root_path = "dataset/k4/root_tweet.json"
        self.network_path = "social_network.json"
        self.content_path = "tweets/"
        self.tw_info = self._load_data(self.root_path)
        self.social_network = self._load_data(self.network_path)
        self.train_tweets = trainset
        self.test_tweets = testset
        self.train = self._train()
    
    # Return the integer id of the root of the cascade (because 1 sometimes may be missing)
    def _sort_cascade(self, cascade):
        tweets = cascade.keys()
        return sorted(map(int, tweets))   
    
    # Extract only the first k tweets in the cascade
    def _load_tweets(self, data, k):
        output = {}
        for entry in data:
            cascade = entry['cascade']
            sort = self._sort_cascade(cascade)
            output.update({entry['url']: {str(sort[0]):cascade[str(sort[0])]} })
            for i in range(1,k):
                output[entry['url']][str(sort[i])] = cascade[str(sort[i])]
        return output
    
    # Extract only the first k tweets in the cascade
    def _load_cascade(self, data, k):
        output = {}
        cascade = data['cascade']
        sort = self._sort_cascade(cascade)
        output.update({data['url']: {str(sort[0]):cascade[str(sort[0])]} })
        for i in range(1,k):
            output[data['url']][str(sort[i])] = cascade[str(sort[i])]
        return output    
    
    def _extract_urls(self, data):
        urls = []
        for url in data.keys():
            urls.append(url)
        return urls    
    
    # Get the label viral/non-viral (True or False) for a cascade.
    def _get_label(self, url, data):
        for entry in data:
            if url in entry['url']:
                return entry['label']
    
    def _extract_afinn(self, url):
        tw_id = self.tw_info[url]
        with open(self.content_path + tw_id + '.json', 'r') as f:
            tw_content = json.load(f)
            if 'en' in tw_content['lang'] and not None:
                afinn = Afinn(emoticons=True)
                text = ' '.join(map(str, self._Preprocess(tw_content['text'])))
                return afinn.score(text)
            else:
                return 0    
    
    def _extract_wordcount(self, url):
        tw_id = self.tw_info[url]
        with open(self.content_path + tw_id + '.json', 'r') as f:
            tw_content = json.load(f)
            if 'en' in tw_content['lang'] and not None:
                return len(self._Preprocess(tw_content['text']))
            else:
                return 0    

    def _getTermCounter(self, url, method):
        p = ttp.Parser()
        count_all = Counter()
        tw_id = self.tw_info[url]
        with open(self.content_path + tw_id + '.json', 'r') as f:
            tw_content = json.load(f)
            if 'en' in tw_content['lang'] and not None:
                r = p.parse(tw_content['text'])
                count_all.update(getattr(r, method))
                return sum(count_all.values())
            else:
                return 0
    
    # Extract the number of followees of the root of the tree
    def _extract_followees(self, url, data):
        cascade = data[url]
        sort = self._sort_cascade(cascade)
        root = str(sort[0])
        if cascade[root]['user'] in self.social_network:
            return len(self.social_network[cascade[root]['user']])
        else:
            #print('Not in network data user: ' + cascade[root]['user'])
            return 0
    
    def _extract_avgtime(self, url, data):
        cascade = data[url]
        t = []
        previous = datetime.datetime.fromtimestamp(int(cascade['1']['created_at'])/1000)
        for i in range(2, len(cascade)+1):
            tweet_timestamp = cascade[str(i)]['created_at']
            current = datetime.datetime.fromtimestamp(int(tweet_timestamp)/1000)
            t.append(current - previous)
            previous = current
        avgtime = numpy.mean(t)
        return avgtime.seconds
    
    def _extract_timeToK(self, url, data):
        cascade = data[url]
        root_time = datetime.datetime.fromtimestamp(int(cascade['1']['created_at'])/1000)
        k_time = datetime.datetime.fromtimestamp(int(cascade[str(len(cascade))]['created_at'])/1000)
        interval = k_time - root_time
        return interval.seconds

    def _extract_numIsolated(self, url, data):
        cascade = data[url]
        count = 0
        for i in cascade:
            if 'in_reply_to' in cascade[str(i)].keys():
                if not cascade[str(i)]['in_reply_to']:
                    count = count + 1
            else:
                count = count + 1
        return count

    def _extract_numEdges(self, url, data):
        cascade = data[url]
        count = 0
        for i in cascade:
            if 'in_reply_to' in cascade[str(i)].keys():
                if cascade[str(i)]['in_reply_to'] != -1:
                    count = count + len(cascade[str(i)]['in_reply_to'])
        return count
    
    # Load training and test data
    def _load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

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
        dataframe['urls'] = self._extract_urls(train_parse)
        dataframe['labels'] = dataframe['urls'].apply(lambda url: self._get_label(url, self.train_tweets))     
        dataframe['afinn'] = dataframe['urls'].apply(lambda url: self._extract_afinn(url))
        dataframe['wordcount'] = dataframe['urls'].apply(lambda url: self._extract_wordcount(url))
        dataframe['num_hashtags'] = dataframe['urls'].apply(lambda url: self._getTermCounter(url, 'tags'))
        dataframe['num_urls'] = dataframe['urls'].apply(lambda url: self._getTermCounter(url, 'urls'))
        dataframe['num_mentions'] = dataframe['urls'].apply(lambda url: self._getTermCounter(url, 'users'))
        # 292 users are not found in social_network data
        dataframe['num_followee'] = dataframe['urls'].apply(lambda url: self._extract_followees(url, train_parse))
        dataframe['avg_time'] = dataframe['urls'].apply(lambda url: self._extract_avgtime(url, train_parse))
        dataframe['time_to_k'] = dataframe['urls'].apply(lambda url: self._extract_timeToK(url, train_parse))
        dataframe['num_isolated'] = dataframe['urls'].apply(lambda url: self._extract_numIsolated(url, train_parse))
        dataframe['num_edges'] = dataframe['urls'].apply(lambda url: self._extract_numEdges(url, train_parse))
        return dataframe
    
    def _test(self):
        test_parse = self._load_tweets(self.test_tweets,self.k)        
        dataframe = pd.DataFrame()
        dataframe['urls'] = self._extract_urls(test_parse)
        dataframe['labels'] = dataframe['urls'].apply(lambda url: self._get_label(url, self.test_tweets))        
        dataframe['afinn'] = dataframe['urls'].apply(lambda url: self._extract_afinn(url))
        dataframe['wordcount'] = dataframe['urls'].apply(lambda url: self._extract_wordcount(url))
        dataframe['num_hashtags'] = dataframe['urls'].apply(lambda url: self._getTermCounter(url, 'tags'))
        dataframe['num_urls'] = dataframe['urls'].apply(lambda url: self._getTermCounter(url, 'urls'))
        dataframe['num_mentions'] = dataframe['urls'].apply(lambda url: self._getTermCounter(url, 'users'))
        dataframe['num_followee'] = dataframe['urls'].apply(lambda url: self._extract_followees(url, test_parse))
        dataframe['avg_time'] = dataframe['urls'].apply(lambda url: self._extract_avgtime(url, test_parse))
        dataframe['time_to_k'] = dataframe['urls'].apply(lambda url: self._extract_timeToK(url, test_parse))
        dataframe['num_isolated'] = dataframe['urls'].apply(lambda url: self._extract_numIsolated(url, test_parse))
        dataframe['num_edges'] = dataframe['urls'].apply(lambda url: self._extract_numEdges(url, test_parse))
        return dataframe    
    
    def _parse_cascade(self, cascade):
        test_parse = self._load_cascade(cascade, self.k)
        dataframe = pd.DataFrame()
        dataframe['urls'] = pd.Series(cascade['url'])
        dataframe['afinn'] = pd.Series(self._extract_afinn(cascade['url']))
        dataframe['wordcount'] = pd.Series(self._extract_wordcount(cascade['url']))
        dataframe['num_hashtags'] = pd.Series(self._getTermCounter(cascade['url'], 'tags'))
        dataframe['num_urls'] = pd.Series(self._getTermCounter(cascade['url'], 'urls'))
        dataframe['num_mentions'] = pd.Series(self._getTermCounter(cascade['url'], 'users'))
        dataframe['num_followee'] = pd.Series(self._extract_followees(cascade['url'], test_parse))
        dataframe['avg_time'] = pd.Series(self._extract_avgtime(cascade['url'], test_parse))
        dataframe['time_to_k'] = pd.Series(self._extract_timeToK(cascade['url'], test_parse))
        dataframe['num_isolated'] = pd.Series(self._extract_numIsolated(cascade['url'], test_parse))
        dataframe['num_edges'] = pd.Series(self._extract_numEdges(cascade['url'], test_parse))
        return dataframe
        
    def classify_prob(self, cascade):
        features = self._parse_cascade(cascade)
        test = features
        pipeline = Pipeline([('featurize', DataFrameMapper([('afinn', None), ('wordcount', None), ('num_hashtags', None), ('num_urls', None), ('num_mentions', None), ('num_followee', None), ('avg_time', None), ('time_to_k', None), ('num_isolated', None), ('num_edges', None)])), ('knn', KNeighborsClassifier())])
        X = self.train[self.train.columns.drop(['urls', 'labels'])]
        y = self.train['labels']
        test['predict'] = pipeline.fit(X = X, y = y).predict(test)
        prob = pipeline.fit(X = X, y = y).predict_proba(test)
        return {"positive": prob[0][1], "negative": prob[0][0]}
    
    def dev(self, test_dataset):
        #return self._train(), self._test()
        for cascade in test_dataset:
            result = self.classify_prob(cascade)
            print(result)
    
    def classify_all(self):
        train = self._train()
        test = self._test()
        pipeline = Pipeline([('featurize', DataFrameMapper([('afinn', None), ('wordcount', None), ('num_hashtags', None), ('num_urls', None), ('num_mentions', None), ('num_followee', None), ('avg_time', None), ('time_to_k', None), ('num_isolated', None), ('num_edges', None)])), ('knn', KNeighborsClassifier())])
        X = train[train.columns.drop(['urls', 'labels'])]
        y = train['labels']
        test['predict'] = pipeline.fit(X = X, y = y).predict(test)
        prob = pipeline.fit(X = X, y = y).predict_proba(test)
        result = [{'positive':prob[i][1], 'negative':prob[i][0]} for i in range(len(prob))]
        print(metrics.classification_report(test['labels'], test['predict']))
        print(metrics.confusion_matrix(test['labels'], test['predict']))
        return result
    
    def classify_export(self):
        train = self._train()
        test = self._test()
        pipeline = Pipeline([('featurize', DataFrameMapper([('afinn', None), ('wordcount', None), ('num_hashtags', None), ('num_urls', None), ('num_mentions', None), ('num_followee', None), ('avg_time', None), ('time_to_k', None), ('num_isolated', None), ('num_edges', None)])), ('knn', KNeighborsClassifier())])
        X = train[train.columns.drop(['urls', 'labels'])]
        y = train['labels']
        test['predict'] = pipeline.fit(X = X, y = y).predict(test)
        prob = pipeline.fit(X = X, y = y).predict_proba(test)
        result = [{'positive':prob[i][1], 'negative':prob[i][0]} for i in range(len(prob))]
        print(metrics.classification_report(test['labels'], test['predict']))
        print(metrics.confusion_matrix(test['labels'], test['predict']))
        return result, test
    
    def classify_tweets_prob_export(self):
        result, test = self.classify_export()
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
    knn = KnnClassifier(train_dataset, test_dataset, 4)
    knn.dev(test_dataset)
    prob = knn.classify_all()
    knn.classify_tweets_prob_export()
    
    #             precision    recall  f1-score   support

    #  False       0.81      0.89      0.85      1022
    #   True       0.65      0.48      0.55       421

#avg / total       0.76      0.77      0.76      1443