import pickle
import pandas as pd
import streamlit as st

#Далее запускаем создание образа
#docker image build -t adult:0.1 .

#Потом создаем контейнер
#Рабочий вариант запуска http://localhost:8501
#docker container run -p 8501:8501 adult:0.1

#Тоже рабочий вариант запуска - открывать ссылку нужно из дэшборда докера http://localhost:8501
#docker container run -d -p 8501:8501 adult:0.1

print('Приложение по классификации дохода (менее или более 50тыс$')
features_all_wo = ['age', 'education', 'marital-status', 'race', 'sex', 'hours-per-week', 'native-country']

filepath_model = r'model/model_y.pkl'
loaded_model = pickle.load(open(filepath_model, 'rb'))

#Зададим функцию предсказания
def prediction(Age, Education, Marital_status, Race, Sex, Hours_week, Native_country):
    y_predict = loaded_model.predict(pd.DataFrame([[Age, Education, Marital_status, Race, Sex, Hours_week, Native_country]]))
    return y_predict

# сделаем предсказание для вводимых пользователем данных
st.title('Приложение для классификации дохода гражданина США')
st.image('Logo.jpeg')
st.header('Для оценки дохода введите параметры:')

# Input text
Age = st.number_input('Возраст гражданина в годах :', min_value=17, max_value=100, value=24)
Race = st.selectbox('Раса (1 - Белый, 2- Негр, 3-Прочее):', [1, 2, 3])
Sex = st.selectbox('Пол пассажира (0 - мужской, 1- женский):', [0, 1])
Hours_week = st.number_input('Занятость часов в неделю :', min_value=0, max_value=100, value=40)
Native_country = st.selectbox('Страна рождения (1 - США, 0- иная страна):', [1, 0])
Education = st.selectbox('Есть высшее образование (1 - есть высшее и более ; 0 - нет высшего):', [1, 0])
Marital_status = st.selectbox('Cемейный статус (0 - нет супруга/супруги; 1-есть): ', [0, 1])

if st.button('Определить доход'):
    p = prediction(Age, Education, Marital_status, Race, Sex, Hours_week, Native_country)
    if p == 0:
        p_str = 'Доход гражданина менее 50 тыс.$'
    else:
        p_str = 'Доход гражданина более 50 тыс.$'

    st.success(p_str)

print('Модель протестирована')

