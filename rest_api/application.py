import logging

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.openapi.utils import get_openapi
from starlette.middleware.cors import CORSMiddleware

from rest_api.controller.errors.http_error import http_error_handler
from rest_api.config import ROOT_PATH
from rest_api.controller.router import router as api_router
from haystack import __version__ as haystack_version


def get_application() -> FastAPI:
    application = FastAPI(title="Haystack REST API", debug=True, version=haystack_version, root_path=ROOT_PATH)

    # This middleware enables allow all cross-domain requests to the API from a browser. For production
    # deployments, it could be made more restrictive.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_exception_handler(HTTPException, http_error_handler)
    application.include_router(api_router)

    return application


def get_openapi_specs() -> dict:
    """
    Used to autogenerate OpenAPI specs file to use in the documentation.

    See `docs/_src/api/openapi/generate_openapi_specs.py`
    """
    app = get_application()
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """
    Simplify operation IDs so that generated API clients have simpler function
    names (see https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#using-the-path-operation-function-name-as-the-operationid).
    The operation IDs will be the same as the route names (i.e. the python method names of the endpoints)
    Should be called only after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


app = get_application()
use_route_names_as_operation_ids(app)

logger.info("Open http://127.0.0.1:8000/docs to see Swagger API Documentation.")
logger.info(
    """
    Or just try it out directly: curl --request POST --url 'http://127.0.0.1:8000/query' -H "Content-Type: application/json"  --data '{"query": "Who is the father of Arya Stark?"}'
    """
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
