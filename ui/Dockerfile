FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# install as a package
COPY requirements.txt /home/user/
RUN pip install -r requirements.txt

# copy code
COPY utils.py /home/user/
COPY webapp.py /home/user/
COPY eval_labels_example.csv /home/user/
COPY SessionState.py /home/user/

EXPOSE 8501

# cmd for running the API
CMD ["streamlit", "run", "webapp.py"]
