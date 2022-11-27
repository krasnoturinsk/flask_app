import numpy as np
import flask
from flask import render_template
import pickle
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

app = flask.Flask(__name__, template_folder='templates')

#@app.route('/', methods=['POST', 'GET'])

#@app.route('/index', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('jupyter\welding.pkl', 'rb') as F:
            loaded_model = pickle.load(F)

        welding_params = []
        welding_params.append(float(flask.request.form['IW']))
        welding_params.append(float(flask.request.form['IF']))
        welding_params.append(float(flask.request.form['VW']))
        welding_params.append(float(flask.request.form['FP']))
#        welding_params = float(flask.request.form['IF'])
        #preds = loaded_model.predict([[welding_params]])
        # мы используем "деревянную" модель, поэтому предобработка
        #  признаков не требуется, подаем на вход модели как есть
        #preds = welding_params
        #preds = loaded_model.predict([44.0, 146.0, 9.0, 60.0])
        preds = [44.0, 146.0, 9.0, 60.0]

        return render_template('main.html', result = preds)

if __name__ == '__main__':
    app.run()
            
            