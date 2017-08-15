#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.externals import joblib
import jieba
import jieba.analyse
from collections import OrderedDict
import numpy as np

jieba.load_userdict('userdict.txt')
jieba.analyse.set_stop_words('stopwords.txt')

label = ['汽车','财经','IT','健康','体育','旅游','教育','招聘','文化','军事']

def loadStopwords():
    f = open('stopwords.txt','r')
    stopwords = []
    for line in f.readlines():
        stopwords.append(line.strip())
    return stopwords

def seg(text):
    stopwords = loadStopwords()

    segs = jieba.cut(text.replace('&nbsp',''))
    segs = list(segs)
    for word in segs:
        if word in stopwords:
            segs.remove(word)
    return ' '.join(segs)

def predict(text):
    segs = seg(text)
    test = []
    test.append(segs)
    test = np.array(test)
    tfidf = joblib.load('tfidf.model')
    vector = tfidf.transform(test)
    model = joblib.load('logistic.model')
    predict_label = model.predict(vector)
    return label[predict_label[0]]
