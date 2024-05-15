import pickle
import pandas as pd
from sklearn import metrics

features_all_wo = ['age', 'race', 'sex', 'hours-per-week', 'native-country']

#Загружаем модель
filepath_model = r'model/model_y.pkl'
loaded_model = pickle.load(open(filepath_model, 'rb'))

#Сделаем кросс-валидацию и посчитаем метрики
filepath_train = r'data/data_train.csv'
data_train = pd.read_csv(filepath_train)
X_train = data_train[features_all_wo].values
y_train = data_train['income'].values
y_pred = loaded_model.predict(X_train)

print('Качество предсказания - Accuracy = ',metrics.accuracy_score(y_train, y_pred))


