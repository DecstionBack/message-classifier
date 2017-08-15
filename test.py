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

#加载停用词
def loadStopwords():
    f = open('stopwords.txt','r')
    stopwords = []
    for line in f.readlines():
        stopwords.append(line.strip())
    return stopwords

#对输入的文本进行分词
def seg(text):
    stopwords = loadStopwords()

    segs = jieba.cut(text.replace('&nbsp',''))
    segs = list(segs)
    for word in segs:
        if word in stopwords:
            segs.remove(word)
    return ' '.join(segs)

#预测分类
def test_classifier(text):
    segs = seg(text)
    test = []
    test.append(segs)
    test = np.array(test)
    #加载训练好的模型
    tfidf = joblib.load('tfidf.model')
    vector = tfidf.transform(test)
    model = joblib.load('logistic.model')
    predict_label = model.predict(vector)
    print label[predict_label[0]]

def main():
    stop = '1'
    #可以运行一次程序判断多篇文本的类别
    while(stop!='0'):
        print '请输入文本（文本不要包含回车符号）:',
        text = raw_input()
        test_classifier(text)
        print '请确认是否继续？0代表停止，1代表继续'
        stop = raw_input()

if __name__ == '__main__':
    main()