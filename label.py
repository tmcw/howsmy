from flask import Flask, request, redirect, render_template
from os import listdir
from shutil import move
import toml

app = Flask(__name__)

@app.route('/')
def hello(name=None):
    file = listdir('./static')[0];
    return render_template('index.html', file=file)

@app.route('/label', methods=['POST'])
def label(name=None):
    file = request.form['file']
    label  = request.form['label']
    training = toml.load('./training.toml')
    max = sorted(map(lambda x: int(x), training.keys()), reverse=True)[0]
    new_key = str(max + 1)
    training[new_key] = label
    toml.dump(training, open('./training.toml', 'w'))
    move('./static/%s' % file, './images/captcha-%s.png' % new_key.zfill(3))
    return redirect('/')
