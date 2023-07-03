from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.routing import APIRoute
from fastapi.openapi.utils import get_openapi
from starlette.middleware.cors import CORSMiddleware
from haystack import __version__ as haystack_version

from rest_api.pipeline import setup_pipelines
from rest_api.controller.errors.http_error import http_error_handler


app = None
pipelines = None


def get_app() -> FastAPI:
    """
    Initializes the App object and creates the global pipelines as possible.
    """
    global app  # pylint: disable=global-statement
    if app:
        return app

    from rest_api.config import ROOT_PATH

    app = FastAPI(title="Haystack REST API", debug=True, version=haystack_version, root_path=ROOT_PATH)

    # Creates the router for the API calls
    from rest_api.controller import file_upload, search, feedback, document, health

    router = APIRouter()
    router.include_router(search.router, tags=["search"])
    router.include_router(feedback.router, tags=["feedback"])
    router.include_router(file_upload.router, tags=["file-upload"])
    router.include_router(document.router, tags=["document"])
    router.include_router(health.router, tags=["health"])

    # This middleware enables allow all cross-domain requests to the API from a browser. For production
    # deployments, it could be made more restrictive.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    app.add_exception_handler(HTTPException, http_error_handler)
    app.include_router(router)

    # Simplify operation IDs so that generated API clients have simpler function
    # names (see https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#using-the-path-operation-function-name-as-the-operationid).
    # The operation IDs will be the same as the route names (i.e. the python method names of the endpoints)
    # Should be called only after all routes have been added.
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    return app


def get_pipelines():
    global pipelines  # pylint: disable=global-statement
    if not pipelines:
        pipelines = setup_pipelines()
    return pipelines


def get_openapi_specs() -> dict:
    """
    Used to autogenerate OpenAPI specs file to use in the documentation.

    Returns `servers` to specify base URL for OpenAPI Playground (see https://swagger.io/docs/specification/api-host-and-base-path/)

    See `.github/utils/generate_openapi_specs.py`
    """

    app = get_app()
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        servers=[{"url": "http://localhost:8000"}],
    )
