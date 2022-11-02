#
# DEPRECATION NOTICE
#
# This Dockerfile and the relative image deepset/haystack-cpu
# have been deprecated in 1.9.0 in favor of:
# https://github.com/deepset-ai/haystack/tree/main/docker
#
FROM python:3.7.4-stretch

WORKDIR /home/user

RUN apt-get update && apt-get install -y \
    curl  \
    git  \
    pkg-config  \
    cmake \
    libpoppler-cpp-dev  \
    tesseract-ocr  \
    libtesseract-dev  \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install PDF converter
RUN wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz && \
    tar -xvf xpdf-tools-linux-4.04.tar.gz && cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin

# Copy Haystack code
COPY haystack /home/user/haystack/
# Copy package files & models
COPY pyproject.toml VERSION.txt LICENSE README.md models* /home/user/
# Copy REST API code
COPY rest_api /home/user/rest_api/

# Install package
RUN pip install --upgrade pip
RUN pip install --no-cache-dir .[docstores,crawler,preprocessing,ocr,ray]
RUN pip install --no-cache-dir rest_api/
RUN ls /home/user
RUN pip freeze
RUN python3 -c "from haystack.utils.docker import cache_models;cache_models()"

# create folder for /file-upload API endpoint with write permissions, this might be adjusted depending on FILE_UPLOAD_PATH
RUN mkdir -p /home/user/rest_api/file-upload
RUN chmod 777 /home/user/rest_api/file-upload

# optional : copy sqlite db if needed for testing
#COPY qa.db /home/user/

# optional: copy data directory containing docs for ingestion
#COPY data /home/user/data

EXPOSE 8000
ENV HAYSTACK_DOCKER_CONTAINER="HAYSTACK_CPU_CONTAINER"

# cmd for running the API
CMD ["gunicorn", "rest_api.application:app", "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "180"]
