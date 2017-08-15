#encoding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np

from DBUtils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def train(X_train, X_test, y_train, y_test):

    tfidf  = TfidfVectorizer(max_features=30000)
    train_vector = tfidf.fit_transform(X_train)

    print train_vector.shape
    test_vector = tfidf.transform(X_test)
    joblib.dump(tfidf, 'tfidf.model')

    print '逻辑回归分类'
    model = LogisticRegression()
    model.fit(train_vector, y_train)
    predict_label = model.predict(test_vector)
    print (classification_report(y_test, predict_label))
    joblib.dump(model, 'logistic.model')

def cross(x,y):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

    train_data = np.array(x)
    train_label = np.array(y)
    # 分成10份去交叉验证，得到结果
    num_folds = 10
    train_data_folds = np.array_split(train_data, num_folds)
    train_label_folds = np.array_split(train_label, num_folds)
    for i in xrange(num_folds):
        train_data = np.concatenate((train_data_folds[0:i] + train_data_folds[i + 1:num_folds]))
        train_label = np.concatenate((train_label_folds[0:i] + train_label_folds[i + 1:num_folds]))
        test_data = train_data_folds[i]
        test_label = train_label_folds[i]
        print "fold:" + str(i)
        train(train_data, test_data, train_label, test_label)

def getSeg(file):
    data = [ ]
    f  = open(file, 'r')
    for line in f.readlines():
        data.append(line.strip())
    f.close()
    return data

def getLabel(file):
    data = [ ]
    f = open(file, 'r')
    for line in f.readlines():
        data.append(int(line.strip()))
    f.close()
    return data

def main():
    x = getSeg('seg.txt')
    y = getLabel('label.txt')
    # segs = selectBySql('select seg, label from sogou')
    # x = []
    # y = []
    # print len(segs)
    # for seg in segs:
    #     x.append(seg[0])
    #     y.append(seg[1])
    # x = np.array(x)
    # y = np.array(y)
    # cross(x, y)
    X_train, X_test, y_train,y_test = train_test_split(x,y, test_size=0.1,random_state=0)
    train(X_train, X_test, y_train,y_test)


if __name__ == '__main__':
    main()



