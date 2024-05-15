FROM python:3.10
USER root

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
EXPOSE 8501
LABEL authors="URFU_2. MLops. Vorobjev Vasiliy"
CMD ["python", "create_dataset.py"]
CMD ["python", "prepare_dataset.py"]
CMD ["python", "model_train.py"]
CMD ["streamlit", "run", "app.py","--server.port=8501"]
