from flask import Flask, request, redirect, render_template
from shutil import move
from os import listdir


app = Flask(__name__)


@app.route('/')
def hello(name=None):
    file = listdir('./static/test_set')[0]
    return render_template('index.html', file=file)


@app.route('/label', methods=['POST'])
def label(name=None):
    file = request.form['file']
    label = request.form['label']

    move('./static/test_set/%s' % file, './static/labeled/%s.png' % label)
    return redirect('/')
