#encoding=utf8

import mysql.connector

def selectBySql(sql):
    conn = mysql.connector.connect(host='localhost',port=3306,user='root',password='root',db='homework')
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
    except IOError,e:
        print e

    alldata = cursor.fetchall()

    cursor.close()
    conn.close()

    return alldata

def updateBySql(sql):
    conn = mysql.connector.connect(host='localhost', port=3306, user='root', password='root', db='homework')
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
    except IOError, e:
        print e

    conn.commit()
    cursor.close()
    conn.close()