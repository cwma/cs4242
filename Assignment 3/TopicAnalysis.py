import csv
import json
import nltk
import re
import os
import pickle
from gensim import corpora, models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamulticore import LdaModel
from gensim.utils import simple_preprocess
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import pyLDAvis.gensim as gensimvis
import pyLDAvis

class TopicAnalysis():

    def __init__(self):
        self.labelfile = 'subsetlabels.csv'
        self.tknzr = RegexpTokenizer(r'\w+')
        self.stopwords = stopwords.words('english')
        self.stopwords.extend(["rt"])
        self.labels = self.getLabels()

    def removeLinks(self, text):
        regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        r = re.sub(regex, "", text)
        return r

    def notInt(self, s):
        try: 
            int(s)
            return False
        except ValueError:
            return True

    def getLabels(self):
        labels = {}
        with open(self.labelfile, "r") as labelfile:
            labelcsv = csv.reader(labelfile)
            for num, line in enumerate(labelcsv):
                if num > 0:
                    labels[line[0]] = line[1]
            return labels

    def getTweetTexts(self, tweetid):
        tweetfile = "./TweetsByUserID/{0}_tweets.json".format(tweetid)
        tweet_texts = []
        with open(tweetfile, "r", encoding='utf-8') as tweets:
            for tweet in tweets:
                tweet = json.loads(tweet)
                if tweet['lang'] == "en":
                    tweet_texts.append(tweet['text'])
        return tweet_texts

    def buildModel(self, tweet_texts):
        texts = [[token for token in self.tknzr.tokenize(self.removeLinks(tweet_text.lower())) if token not in self.stopwords and self.notInt(token) and len(token) > 2] for tweet_text in tweet_texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        ldamodel = LdaMulticore(corpus, num_topics=50, id2word=dictionary, passes=30, workers=3)
        return dictionary, corpus, ldamodel

    def build(self, _label):
        modelfile = "./models/{0}.model".format(_label)
        dictfile = "./models/{0}.dict".format(_label)
        corpusfile = "./models/{0}.mm".format(_label)
        if os.path.isfile(modelfile):
            dictionary = corpora.Dictionary.load(dictfile)
            corpus = corpora.MmCorpus(corpusfile)
            ldamodel = LdaModel.load(modelfile)
        else:
            texts = []
            for tweetid, label in self.labels.items():
                if label == _label:
                    texts.extend(self.getTweetTexts(tweetid))
            dictionary, corpus, ldamodel = self.buildModel(texts)
            dictionary.save(dictfile)
            corpora.MmCorpus.serialize(corpusfile, corpus)
            ldamodel.save(modelfile)
        return dictionary, corpus, ldamodel

    def visualize(self, _label):
        visfile = "./models/{0}.vis".format(_label)
        if os.path.isfile(visfile):
            vis_data = pickle.load(open(visfile, "rb"))
        else:
            dictionary, corpus, ldamodel = self.build(_label)
            vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
            pickle.dump(vis_data, open(visfile, "wb"))
        pyLDAvis.show(vis_data)

    def preprocess(self, _label):
        visfile = "./models/{0}.vis".format(_label)
        dictionary, corpus, ldamodel = self.build(_label)
        vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
        pickle.dump(vis_data, open(visfile, "wb"))

if __name__ == "__main__":
    ta = TopicAnalysis()
    #ta.visualize("human")
    #ta.visualize("consumption")
    ta.visualize("spam-promotion")
    #ta.visualize("broadcast")

    # extract models.7z to ./models