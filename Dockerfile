FROM python:3.7.4-stretch

WORKDIR /home/user

# copy code
COPY haystack /home/user/haystack

# install as a package
COPY setup.py requirements.txt README.rst /home/user/
RUN pip install -r requirements.txt
RUN pip install -e .

# copy saved models
COPY README.rst models* /home/user/models/

# Copy REST API code
COPY rest_api /home/user/rest_api

# optional : copy sqlite db if needed for testing
#COPY qa.db /home/user/

# optional: copy data directory containing docs for ingestion
#COPY data /home/user/data

EXPOSE 8000

# cmd for running the API
CMD ["gunicorn", "rest_api.application:app",  "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "180", "--preload"]
