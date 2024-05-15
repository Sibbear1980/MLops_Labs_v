import pandas as pd
import numpy as np
from pathlib import Path
import pickle #Библиотека для сохранения моделей
import matplotlib.pyplot as plt # библиотека Matplotlib для визуализации

#Зададим функцию визуализации значимости признаков
def feature_importance_plotter(model, features_names):
    """Отрисовка значимости признаков в виде горизонтальных столбчатых диаграмм.
    Параметры:
    model: модель
    features_names: список имен признаков
    """
    feature_importance = model.feature_importances_
    sorted = np.argsort(feature_importance)
    ypos = np.arange(len(features_names))
    fig= plt.figure(figsize=(8,4))
    plt.barh(ypos, feature_importance[sorted])
    #plt.xlim([0,1])
    plt.ylabel('Параметры')
    plt.xlabel('Значимость')
    plt.yticks(ypos, features_names[sorted])
    plt.show()

features_all = ['age', 'education', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'native-country', 'income']
#features_all_wo = ['age', 'education', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'native-country']
features_all_wo = ['age', 'race', 'sex', 'hours-per-week', 'native-country']


# прочитаем из csv-файла подготовленный датасет для обучения
filepath_train = r'data/data_train.csv'
data_train = pd.read_csv(filepath_train)
X = data_train[features_all_wo].values
y = data_train['income'].values

# загрузим модель машинного обучения. Будем использовать бустинг от CatBoost
# from catboost import CatBoostRegressor, Pool
# #print(features_all_wo)
# cat_features_names = ['education', 'marital-status', 'occupation']
# train_data_reg = Pool(
#     data=X_train,
#     label=Y_train,
#     cat_features=cat_features_names,
#     feature_names=features_all_wo,
# )
# eval_data_reg = Pool(
#     data=X_train,
#     label=Y_train,
#     cat_features=cat_features_names,
#     feature_names=features_all_wo
# )
# model_y = CatBoostRegressor(iterations = 500,
#                            early_stopping_rounds=10,
#                            verbose = 100,
#                            depth = 3,
#                            objective  = 'MAE',
#                            eval_metric= 'MAE',
#                            random_state = 42
#                            )
# model_y.fit(X=train_data_reg,
#           eval_set=eval_data_reg,
#           )

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model_y = RandomForestClassifier(max_features='log2', n_estimators=300, random_state=73)
# разбиваем на тестовую и валидационную
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.3,
                                                  random_state=73)
model_y.fit(X_train, y_train)

feature_importance_plotter(model_y, np.array(features_all_wo))
#С помощью графического отображения видно признаки, оказывающие влияние на целевую функцию

# сохраним обученную модель
filepath_model = Path('model/model_y.pkl')
filepath_model.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(model_y, open(filepath_model, 'wb'))