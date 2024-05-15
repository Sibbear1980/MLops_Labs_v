from pathlib import Path
import pandas as pd

# прочитаем из csv-файла подготовленный датасет для обучения
filepath_train = r'data/data_train_src.csv'
data_train_src = pd.read_csv(filepath_train)
#X_train = data_train_src.values
#Y_train = data_train_src['income'].values

#Исследуем датасет
pd.set_option('display.max_columns', None)
print(data_train_src.info())
print(data_train_src.head())
print(data_train_src.columns)
print(data_train_src.groupby('income').agg({'age':'count'}))
#Выберем часть параметров для исследования
features_all = ['age', 'education', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'native-country', 'income']
print(data_train_src.groupby('age').agg({'sex':'count'}))
print(data_train_src.groupby('education').agg({'age':'count'}))
print(data_train_src.groupby('marital-status').agg({'age':'count'}))
print(data_train_src.groupby('occupation').agg({'age':'count'}))
print(data_train_src.groupby('race').agg({'age':'count'}))
print(data_train_src.groupby('sex').agg({'age':'count'}))
print(data_train_src.groupby('hours-per-week').agg({'age':'count'}))
print(data_train_src.groupby('native-country').agg({'age':'count'}))


# обработка данных
# исправим значение функции как 0 - если доход <= 50000, 1 если > 50000
data_train_src['income'] = data_train_src['income'].apply(lambda x: 0 if x == '<=50K' else (0 if x == '<=50K.' else 1))

# заполним данные о поле числовыми данными (0 или 1) вместо текстовых ('male' или 'female')
data_train_src['sex'] = data_train_src['sex'].apply(lambda x: 0 if x == 'Male' else 1)

# в признаке "Возраст" много пропущенных (NaN) значений, заполним их средним значением возраста
data_train_src['age'] = data_train_src['age'].fillna(data_train_src.age.mean())

# в признаке "Race" числовыми данными (0, 1, 2, 3) вместо текстовых
data_train_src['race'] = data_train_src['race'].apply(lambda x: 1 if x == 'White' else (2 if x == 'Black' else 3))

# в признаке "Occupation" много пропущенных (NaN) значений, заполним их значением "Other"
data_train_src['occupation'] = data_train_src['occupation'].fillna('other')

# заполним данные об образовании  числовыми данными (1 высшее или 0-нет высшего) вместо текстовых
data_train_src['education'] = data_train_src['education'].apply(lambda x: 1 if x == 'HS-grad'
                                                                            else (1 if x == 'Masters'
                                                                            else (1 if x == 'Doctorate'
                                                                            else (1 if x == 'Bachelors' else 0)
                                                                )))
# заполним данные о семейном статусе числовыми данными (0 нет супруга/супруги или 1-есть) вместо текстовых
data_train_src['marital-status'] = data_train_src['marital-status'].apply(lambda x: 1 if x == 'Married-AF-spous'
                                                                            else (1 if x == 'Married-civ-spouse'
                                                                            else (1 if x == 'Married-spouse-absent' else 0)
                                                                                )
                                                                        )

# заполним данные о стране числовыми данными (0 или 1) вместо текстовых
data_train_src['native-country'] = data_train_src['native-country'].apply(lambda x: 1 if x == 'United-States' else 0)

# запишем предобработанный датасет во внешний csv-файл
filepath_train = Path('data/data_train.csv')
filepath_train.parent.mkdir(parents=True, exist_ok=True)
data_train_src[features_all].to_csv(filepath_train, index=False)
