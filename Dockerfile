FROM python:3.7.4-stretch

WORKDIR /home/user

# install as a package
COPY setup.py requirements.txt README.rst /home/user/
RUN pip install -r requirements.txt
RUN pip install -e .

# copy code
COPY haystack /home/user/haystack

# copy saved FARM models
COPY models /home/user/models

# copy sqlite db if needed for testing
#COPY qa.db /home/user/

EXPOSE 8000

# cmd for running the API
CMD ["uvicorn", "haystack.api.inference:app", "--host", "0.0.0.0", "--port", "8000"]