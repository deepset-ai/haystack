FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y curl git pkg-config cmake

# copy code
COPY setup.py /home/user/
COPY utils.py /home/user/
COPY webapp.py /home/user/
COPY eval_labels_example.csv /home/user/

# install as a package
RUN pip install .

EXPOSE 8501

# cmd for running the API
CMD ["streamlit", "run", "webapp.py"]
