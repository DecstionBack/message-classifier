#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import jieba
import jieba.analyse
from DBUtils import *

jieba.load_userdict('userdict.txt')
jieba.analyse.set_stop_words('stopwords.txt')

def loadStopwords():
    f = open('stopwords.txt','r')
    stopwords = []
    for line in f.readlines():
        stopwords.append(line.strip())
    return stopwords

def seg(alltext):
    stopwords = loadStopwords()
    for text in alltext:
        segs = jieba.cut(text[1].replace('&nbsp',''))
        segs = list(segs)
        for word in segs:
            if word in stopwords:
                segs.remove(word)

        result = " ".join(segs)
        print type(result)
        sql = 'update sogou set seg="%s" where id=%s' % (result.replace("\n",""), text[0])
        print sql
        updateBySql(sql)


def main():
    alltext = selectBySql('select id, text from sogou')
    seg(alltext)

if __name__ == '__main__':
    main()