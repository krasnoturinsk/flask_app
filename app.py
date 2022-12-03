import numpy as np
import pandas as pd
import flask
from flask import render_template
import pickle
from sklearn.ensemble import ExtraTreesRegressor

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['POST', 'GET'])
def main():
    # если GET запрос
    if flask.request.method == 'GET':
        return render_template('main.html')

    # если POST запроc
    if flask.request.method == 'POST':

        # загружаем готовую модель из файла
        with open('jupyter\welding.pkl', 'rb') as F:
            loaded_model = pickle.load(F)

        # загружаем названия принаков
        with open('jupyter\X_names.pkl', 'rb') as F:
            X_columns, y_columns = pickle.load(F)

        # получаем технические параметры сварки из формы
        iw_value = int(flask.request.form['IW'])   # величина сварочного тока
        if_value = int(flask.request.form['IF'])   # ток фокусировки электронного пучка
        vw_value = float(flask.request.form['VW']) # скорость сварки
        fp_value = int(flask.request.form['FP'])   # расстояние до поверхности образцов
        submit_value = flask.request.form['predict_btn']
        
        # формируем датафрейм из признаков
        welding_params = []
        welding_params.append(iw_value)
        welding_params.append(if_value)
        welding_params.append(vw_value)
        welding_params.append(fp_value)
        X = pd.DataFrame([welding_params], columns=X_columns)

        # получаем предсказания модели
        predictions = loaded_model.predict(X)
        y = pd.DataFrame(predictions, columns=y_columns)

        width = round(y.Width.values[0], 3)
        depth = round(y.Depth.values[0], 3)

        data = {
            'iw_value': iw_value,
            'if_value': if_value,
            'vw_value': vw_value,
            'fp_value': fp_value,
            'width':  width,
            'depth': depth,
            'submit': submit_value
        }
        
        return render_template('main.html', result = data)

if __name__ == '__main__':
    app.run(debug=True)