import logging

import uvicorn
from rest_api.utils import get_app, get_pipelines


logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


app = get_app()
pipelines = get_pipelines()  # Unused here, called to init the pipelines early


logger.info("Open http://127.0.0.1:8000/docs to see Swagger API Documentation.")
logger.info(
    """
    Or just try it out directly: curl --request POST --url 'http://127.0.0.1:8000/query' 
    -H "Content-Type: application/json"  --data '{"query": "Who is the father of Arya Stark?"}'
    """
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
