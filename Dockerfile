FROM python:3.7.4-stretch

WORKDIR /home/user

# install as a package
COPY setup.py requirements.txt README.rst /home/user/
RUN pip install -r requirements.txt
RUN pip install -e .

# copy code
COPY haystack /home/user/haystack

# copy saved FARM models
COPY saved_models /home/user/saved_models

# cmd for running the API
CMD FLASK_APP=haystack.api.inference flask run --host 0.0.0.0
