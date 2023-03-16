import logging
from pathlib import Path

import click
import uvicorn

from haystack.preview.rest_api.app import get_app


@click.group()
def rest_api():
    pass


@click.command(name="serve")
@click.option(
    "--pipelines",
    default=Path(__file__).parent / "rest_api" / "pipelines" / "default.json",
    help="Path to the file containing your pipelines",
)
@click.option("--host", default="0.0.0.0", type=str, help="The hostname")
@click.option("--port", default=8000, type=int, help="The port")
@click.option("--no-debug", is_flag=True, help="disable the debug mode of FastAPI")
@click.option("--log-level", default="WARNING", help="the log level of the output")
def serve(host: str, port: int, pipelines: Path, log_level: str, no_debug: bool = False):
    app = get_app(debug=not no_debug, pipelines_path=pipelines)

    @app.on_event("startup")
    async def startup_event():
        logger = logging.getLogger("uvicorn.error")
        logger.info(r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info(r"|    _______                                                  |")
        logger.info(r"|   / _____ \       _   _                 _             _     |")
        logger.info(r"|  / /  _  \ \     | | | |               | |           | |    |")
        logger.info(r"|  | | | | | |     | |_| | __ _ _   _ ___| |_ __ _  ___| | __ |")
        logger.info(r"|  | |_| | | |     |  _  |/ _` | | | / __| __/ _` |/ __| |/ / |")
        logger.info(r"|  | ___ | | |     | | | | (_| | |_| \__ \ || (_| | (__|   <  |")
        logger.info(r"|  |_| | | |_|     \_| |_/\__,_|\__, |___/\__\__,_|\___|_|\_\ |")
        logger.info(r"|      |_|                       __/ |                        |")
        logger.info(r"|                               |___/                         |")
        logger.info(r"|                                                             |")
        logger.info(r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info(r"| Haystack REST API server is running!                        |")
        logger.info(f"| Debug Mode: {'OFF' if no_debug else 'ON'}".ljust(62) + "|")
        logger.info(f"| API Docs: http://{host}:{port}/docs".ljust(62) + "|")
        logger.info(r"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    uvicorn.run(app, host=host, port=port)


def main():
    rest_api.add_command(serve)
    rest_api()
