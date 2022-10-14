FROM python:3.7.4-stretch

# RUN apt-get update && apt-get install -y curl git pkg-config cmake

# copy code
RUN mkdir ui/
COPY . /opt/ui

# install as a package
RUN pip install --upgrade pip && \
    pip install /opt/ui/

WORKDIR /opt
EXPOSE 8501

# cmd for running the API
CMD ["python", "-m", "streamlit", "run", "ui/webapp.py"]
