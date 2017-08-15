#coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os.path
import random

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from web_predict import *

label = ['汽车','财经','IT','健康','体育','旅游','教育','招聘','文化','军事']

from tornado.options import define, options
define("port", default=80, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class ClassifiMessage(tornado.web.RequestHandler):
    def post(self):
        source_text = self.get_argument('source')
        label = predict(source_text)
        self.render('result.html', source=source_text,result=label)

if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[(r'/', IndexHandler), (r'/result',ClassifiMessage)],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()