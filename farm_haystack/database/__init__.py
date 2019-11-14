import logging
import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DATABASE_URL = os.getenv("DATABASE_URL", None)
if not DATABASE_URL:
    try:
        from qa_config import DATABASE_URL
    except ModuleNotFoundError:
        logging.info(
            "Using localhost sqlite as the database backend. as Database not configured. Add a qa_config.py file in the Python path with DATABASE_URL set."
            "Continuing with the default sqlite on localhost."
        )
        DATABASE_URL = "sqlite://"

app = Flask(__name__)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_DATABASE_URI"] = f"{DATABASE_URL}"
db = SQLAlchemy(app)
db.create_all()
