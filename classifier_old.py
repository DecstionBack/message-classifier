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

import jieba
import jieba.analyse
import time

from DBUtils import *


def idf(item, X_train):
    num = 0
    for text in X_train:
        if item in "".join(text):
            num  += 1
    return log(len(X_train) * 1.0 / (num+1))

def tf(item, items):
    num = 0
    for i in items:
        if (item == i):
            num += 1
    return num

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

    #计算每个词语的IDF值
    # for item in dict.keys():
    #     result = idf(item,x)
    #     dict[item] = result
    #     print item + " " + str(result)
    for items in x:
        for item in set(items.split(" ")):
            dict[item] += 1
    for item in dict.keys():
        dict[item] = log(len(x) * 1.0 / (dict[item] + 1))
    # for item in dict:
    #     print item[0]


    #dict = sorted(dict.items(), lambda x,y: cmp(x[1],y[1]),reverse=True)
    f = open('feature.txt','w')
    feature = []
    for item in dict.keys():

        if (dict[item]>5.4 and dict[item]<6.63):
            f.write(item + "\n")
            if (len(item))<2:
                continue
            feature.append(item)
    f.close()
    print len(feature)

#feature 23984个
    # matrix = np.zeros((3000, len(feature)))
    # matrix = np.random.randn(3000,len(feature))


    matrix = loadMatrix()
    print matrix.shape
    # print feature
    # index = 0
    # for items in x:
    #     print index
    #     vector = np.zeros((1,len(feature)))
    #     items = list(items.split(" "))
    #     for item in items:
    #         if item in feature:
    #             vector[0,feature.index(item)] = dict[item] * items.count(item)
    #     matrix[index] = vector
    #     index += 1
    # f = open('matrix.txt','w')
    # for i in xrange(3000):
    #     for j in xrange(len(feature)):
    #         f.write(str(matrix[i][j])+" ")
    #     f.write("\n")
    # f.close()
    print matrix[0]
    print matrix[1]
    print matrix[2999]

    X_train, X_test, y_train, y_test = train_test_split(matrix,y,test_size=0.2,random_state=1)

#idf 6.63-5.4


    # vectorizer = TfidfVectorizer(max_features=20000)
    # train_vector = vectorizer.fit_transform(X_train)
    # test_vector = vectorizer.transform(X_test)

    print '逻辑回归分类'
    t1 =time.time()
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    predict_label = model.predict(X_test)
    print classification_report(y_test, predict_label)
    t2 =time.time()
    print t2-t1


def main():
    segs = selectBySql('select seg, label from sogou')
    x = []

    y=[]
    print len(segs)
    for seg in segs:
        x.append(seg[0])
        y.append(seg[1])
    x = np.array(x)
    y = np.array(y)
    classifier(x,y)

if __name__ == '__main__':
    main()

#逻辑回归：0.81 0.80 0.80
#贝努利朴素贝叶斯：0.82 0.49 0.55
#多项式朴素贝叶斯：0.82 0.82 0.82
#线性SVM：0。80 0.58 0.64
