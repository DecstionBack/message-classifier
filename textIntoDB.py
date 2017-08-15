#encoding=utf8
import os
import mysql.connector


def writeIntoDB(dirpath):
    conn = mysql.connector.connect(host='localhost',port=3306,user='root',password='root',db='homework')
    cursor = conn.cursor()
    for file in os.listdir(dirpath):
        print file
        f = open(os.path.join('%s\%s' % (dirpath, file)), 'r')
        text = f.read().decode('gbk','ignore').encode('utf8').replace("\"","")
        print text
        sql = 'insert into sogou(text, class) values("%s","汽车")' % text

        try:
            cursor.execute(sql)
        except IOError, e:
            print e
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    writeIntoDB("E:\pyCharmProjects\NLP-Project\sogou\C000007")