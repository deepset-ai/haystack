# Haystack Annotation Tool

This document describes setting up the Haystack Annotation Tool with the publicly available Docker Images. Alternatively,
a hosted version of the tool is available at https://annotate.deepset.ai/login.



# Setup Annotation Tool with Docker

1. Configure credentials & database in the `docker-compose.yml` file:

The credentials should match in database image and application configuration.

    DEFAULT_ADMIN_EMAIL: "example@example.com"
    DEFAULT_ADMIN_PASSWORD: "DEMO-PASSWORD"

    PROD_DB_NAME: "databasename"
    PROD_DB_USERNAME: "somesafeuser"
    PROD_DB_PASSWORD: "somesafepassword"


    POSTGRES_USER: "somesafeuser"
    POSTGRES_PASSWORD: "somesafepassword"
    POSTGRES_DB: "databasename"


2. Run docker-compose by executing `docker-compose up`.


3. The UI should be available at `localhost:7001`.