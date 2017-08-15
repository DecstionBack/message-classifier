#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import numpy as np
from math import log
import codecs
from sklearn.externals import joblib

import jieba
import jieba.analyse
import time
from collections import OrderedDict

from DBUtils import *

#计算idf值
def idf(item, X_train):
    num = 0
    for text in X_train:
        if item in "".join(text):
            num  += 1
    return log(len(X_train) * 1.0 / (num+1))

#计算一个词语的tf
def tf(item, items):
    num = 0
    for i in items:
        if (item == i):
            num += 1
    return num

#加载文本矩阵，减少计算量
def loadMatrix():
    matrix = []
    f  = codecs.open('matrix.txt','r')
    index = 0
    for line in f.readlines():
        vector = line.strip().split(" ")
        for i in xrange(len(vector)):
            vector[i] = float(vector[i])
        matrix.append(vector)
        index += 1
    f.close()
    return np.array(matrix)

#对文本进行分类，训练过程
def classifier(x, y):

    #建立词典，并且设置每个词语的向量
    dict = {}
    for items in x:
        for item in items.split(" "):
            dict[item]=0

    #词典的尺寸大小
    print len(dict.keys())

    #看一下词典中都有什么词语
    # for item in dict.keys():
    #     print item

    idf = {}
    for items in x:
        for item in set(items.split(" ")):
            idf[item] = idf.get(item , 0 ) + 1
    for item in dict.keys():
        idf[item] = log(len(x) * 1.0 / (idf[item] + 1))

    f = open('feature.txt','w')
    f2 = open('idf.txt', 'w')
    feature = []
    for item in idf.keys():
        if (idf[item]>5.4 and idf[item]<6.63):
            if (len(item)) < 2:
                continue
            f.write(item + "\n")
            f2.write(item + " " + str(idf[item]) + "\n")

            feature.append(item)
    f.close()
    f2.close()
    print len(feature)
    #下面的dict要求是排好序的
    dict = { }
    for item in feature:
        dict[item]=0

    dict = OrderedDict(sorted(dict.items(), key=lambda t: t[0]))
    for key in dict.keys():
        print key

#feature 23984个
    matrix = np.zeros((3000, len(feature)))
    print matrix.shape
    index = 0

    for items in x:
        print index
        items = list(items.split(" "))
        #传下来时dict中的全为0
        changedFeature = []
        for item in items:
            dict[item] = dict.get(item, 0) + 1
            changedFeature.append(item)
        for item in changedFeature:
            dict[item] = dict.get(item) * 1.0 / len(items) * idf.get(item)
        matrix[index] = np.array((dict.values())[0:len(feature)])
        for item in changedFeature:
            dict[item]=0
        index += 1
    print matrix.shape

    X_train, X_test, y_train, y_test = train_test_split(matrix,y,test_size=0.1,random_state=1)

#idf 6.63-5.4

    print '朴素贝叶斯分类'
    model = BernoulliNB()
    model.fit(X_train, y_train)
    predict_label = model.predict(X_test)

    print classification_report(y_test, predict_label)


def main():

    segs = selectBySql('select seg, label from sogou')
    x = []
    y = []

    print len(segs)
    for seg in segs:
        x.append(seg[0])
        y.append(seg[1])
    x = np.array(x)
    y = np.array(y)
    classifier(x,y)

if __name__ == '__main__':
    main()

#贝努利朴素贝叶斯：0.82 0.50 0.55

