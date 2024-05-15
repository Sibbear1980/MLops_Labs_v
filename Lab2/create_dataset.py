from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

# загрузка данных
#list_available_datasets()
# fetch dataset
adult = fetch_ucirepo(name='Adult')

# # metadata
# print(adult.metadata)
#
# # variable information
# print(adult.variables)

#Объеденим признаки и функцию для удобства хранения
train_src = pd.concat([adult.data.features, adult.data.targets], sort=False, axis=1)
# запишем созданные датасеты во внешние csv-файлы
filepath_train = Path('data/data_train_src.csv')
filepath_train.parent.mkdir(parents=True, exist_ok=True)
train_src.to_csv(filepath_train, index=False)

