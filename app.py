import numpy as np
import pandas as pd
import flask
from flask import render_template
import pickle
from sklearn.ensemble import ExtraTreesRegressor

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['POST', 'GET'])
def main():
    # если запрс без передачи параметров из формы
    if flask.request.method == 'GET':
        return render_template('main.html')

    # если запрос с параметрами
    if flask.request.method == 'POST':
        # загружаем готовую модель из файла
        with open('jupyter\welding.pkl', 'rb') as F:
            loaded_model = pickle.load(F)
        # загружаем названия принаков
        with open('jupyter\X_names.pkl', 'rb') as F:
            X_columns, y_columns = pickle.load(F)

        # получаем технические параметры сварки из формы
        welding_params = []
        # величина сварочного тока
        welding_params.append(float(flask.request.form['IW']))
        # ток фокусировки электронного пучка
        welding_params.append(float(flask.request.form['IF']))
        # скорость сварки
        welding_params.append(float(flask.request.form['VW']))
        # расстояние от поверхности образцов до электронно-оптической системы
        welding_params.append(float(flask.request.form['FP']))
        
        # формируем датафрейм из признаков
        X = pd.DataFrame([welding_params], columns=X_columns)

        # получаем предсказания модели
        predictions = loaded_model.predict(X)
        y = pd.DataFrame(predictions, columns=y_columns)

        width = round(y.Width.values[0], 3)
        depth = round(y.Depth.values[0], 3)
        
        return render_template('main.html', result = f'Ширина: {width}, Глубина: {depth}')

if __name__ == '__main__':
    app.run(debug=True)