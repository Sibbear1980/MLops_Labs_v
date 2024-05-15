import pandas as pd
from pathlib import Path
import pickle #Библиотека для сохранения моделей


#features_all = ['age', 'education', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'native-country', 'income']
features_all_wo = ['age', 'education', 'marital-status', 'race', 'sex', 'hours-per-week', 'native-country']


# прочитаем из csv-файла подготовленный датасет для обучения
filepath_train = r'data/data_train.csv'
data_train = pd.read_csv(filepath_train)
X = data_train[features_all_wo].values
y = data_train['income'].values

#print(data_train.head(40))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model_y = RandomForestClassifier(max_features='log2', n_estimators=300, random_state=73)
# разбиваем на тестовую и валидационную
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.3,
                                                  random_state=73)
model_y.fit(X_train, y_train)

# сохраним обученную модель
filepath_model = Path('model/model_y.pkl')
filepath_model.parent.mkdir(parents=True, exist_ok=True)
pickle.dump(model_y, open(filepath_model, 'wb'))