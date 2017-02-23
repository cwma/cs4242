#!/usr/bin/python

# Tokenization and pre-processing
# Feature extraction
# Syntactic features
# Semantic features
# Classification

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

training_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/training.json"
dev_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/development.json"
neg_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/lexicon/neg.txt"
pos_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/lexicon/pos.txt"
tweets_path = "/Users/jasonngchangwei/Documents/social_media/assignment1/tweets/"

def extract_tweet(file_path):
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
                print (t['id_str'] + " is not in training set")
            f2.close()
    return results

def extract_link(text):
    try:
        regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        match = re.search(regex, text)
        if match:
            return match.group()
        return ''
    except:
        return ''

def remove_link(text):
    try:
        regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        r = re.sub(regex, "", text)
        return r
    except:
        return ''

def get_afinn_score(text):
    afinn = Afinn(emoticons=True)
    return afinn.score(text)

# SentiWordNet is for twitter

if __name__ == "__main__":
    
    test_tweet = "816220705352192000"
    for i in range(0,len(train_tweets)):
        if(train_tweets[i][0] == test_tweet):
            test = train_tweets[i][1]

    train_tweets = []
    test_tweets = []
    train_tweets = extract_tweet(training_path)
    test_tweets = extract_tweet(dev_path)
    
    train = pd.DataFrame()
    train['user_id'] = list(map(lambda tweet: tweet[0], train_tweets))
    train['text'] = list(map(lambda tweet: remove_link(tweet[1]), train_tweets))
    train['sentiment'] = list(map(lambda tweet: tweet[2], train_tweets))
    train['afinn'] = train['text'].apply(lambda tweet: get_afinn_score(tweet))

    test = pd.DataFrame()
    test['user_id'] = list(map(lambda tweet: tweet[0], train_tweets))
    test['text'] = list(map(lambda tweet: remove_link(tweet[1]), train_tweets))
    test['sentiment'] = list(map(lambda tweet: tweet[2], train_tweets))
    test['afinn'] = test['text'].apply(lambda tweet: get_afinn_score(tweet))
    
    
    text = wordpunct_tokenize(tweet)
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    pretokens = tknzr.tokenize(tweet)
    stop_words = set(stopwords.words('english'))

    stemmer = PorterStemmer()
    tagged = nltk.pos_tag(tokenized)
    
    pipeline = Pipeline([('featurize', DataFrameMapper([('afinn', None)])), ('knn', KNeighborsClassifier())])
    X = train[train.columns.drop(['sentiment', 'user_id', 'text'])]
    y = train['sentiment']
    test['predict'] = pipeline.fit(X = X, y = y).predict(test)
    prob = pipeline.fit(X = X, y = y).predict_proba(test)[:, 1]
    
    print(metrics.classification_report(test['sentiment'], test['predict']))
    print(metrics.confusion_matrix(test['sentiment'], test['predict']))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        