import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesRegressor

# загружаем готовую модель из файла
with open('jupyter\welding.pkl', 'rb') as F:
    loaded_model = pickle.load(F)
# загружаем названия признаков
with open('jupyter\X_names.pkl', 'rb') as F:
    X_columns, y_columns = pickle.load(F)

# технические параметры сварки
data = [[44.0, 146.0, 9.0, 60.0],]

# формируем датафрейм с данными для модели
X = pd.DataFrame(data, columns=X_columns)

# получаем предсказания модели
predictions = loaded_model.predict(X)
y = pd.DataFrame(predictions, columns=y_columns)

print(f'''
Технические параметры сварки:
    Расстояние до поверхности образца (FP):  {X.FP.values[0]}
    Ток фокусировки электронного пучка (IF):  {X.IF.values[0]}
    Величина сварочного тока (IW):  {X.IW.values[0]}
    Скорость сварки (VW):  {X.VW.values[0]}

Прогнозируемые параметры саврного шва:
    Глубина шва:  {y.Depth.values[0]:.2f}
    Ширина шва:  {y.Width.values[0]:.2f}
''')
