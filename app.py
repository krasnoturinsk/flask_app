import flask
from flask import render_template
import pickle
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['POST', 'GET'])

@app.route('/index', methods=['POST', 'GET'])

def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('welding.pkl', 'rb') as F:
            loaded_model = pickle.load(F)
            