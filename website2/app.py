#!/usr/bin/env python

import logging
import optparse
import tornado.wsgi
import tornado.httpserver

from flask import render_template, Flask

def start_tornado(app, port=5003):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5003)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode, specify gpu to use",
        type='int', default=-1)

    opts, args = parser.parse_args()

    start_tornado(app, opts.port)

# Obtain the flask app object
app = Flask(__name__)

@app.route('/', defaults={'page': 'index'})
@app.route('/<path:page>')
def show(page):
    return render_template('index.html')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)