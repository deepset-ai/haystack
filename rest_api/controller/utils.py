# coding: utf8
import logging
import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from contextlib import contextmanager
from threading import Semaphore
from fastapi import HTTPException
from rest_api.config import ACTIVE_PIPELINE_FILE, PIPELINES_DIR, LOG_LEVEL
from haystack import Pipeline

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


class RequestLimiter:
    def __init__(self, limit):
        self.semaphore = Semaphore(limit - 1)

    @contextmanager
    def run(self):
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail="The server is busy processing requests.")
        try:
            yield acquired
        finally:
            self.semaphore.release()


class PipelineSchema(BaseModel):
    name: str
    type: str
    status: str = "deactivate"
    nodes: list
    yaml_file: str


class PipelineHelper:

    def __init__(self, yaml_files_path: str):
        """
        Pipeline Helper class
        :param yaml_files_path:
        """
        self.yaml_files_path = Path(yaml_files_path)
        self.pipelines: list = []

    def parse_yaml_file(self, yaml_file: str) -> dict:
        """
        Parse yaml file return python dict object
        :param yaml_file: Path to yaml file
        :return: Dict object
        """
        with open(yaml_file, "r", encoding='utf8') as stream:
            return yaml.safe_load(stream)

    def load_files(self) -> List[dict]:
        """
        Load yaml files from directory
        :return: list of dict
        """
        list_of_dicts: List = []
        if not self.yaml_files_path.is_dir():
            raise ValueError(f"{self.yaml_files_path} is not valid directory.")

        for yaml_file in self.yaml_files_path.glob('*.yaml'):
            _dict: dict = self.parse_yaml_file(str(yaml_file.absolute()))
            _dict['file_name'] = yaml_file.name
            list_of_dicts.append(_dict)

        return list_of_dicts

    def load_data(self):
        """
        Load the data from yaml files
        :return: None
        """
        self.pipelines = []
        yaml_data: list = self.load_files()
        active_pipeline: str = ""
        for file_content in yaml_data:
            if 'active_pipeline' in file_content:
                active_pipeline = file_content['active_pipeline']

        for file_content in yaml_data:
            if 'pipelines' in file_content:
                for pipeline in file_content['pipelines']:
                    pipeline_schema = PipelineSchema(name=pipeline['name'],
                                                     type=pipeline['type'],
                                                     nodes=pipeline['nodes'],
                                                     yaml_file=file_content['file_name'])
                    if pipeline['name'] == active_pipeline:
                        pipeline_schema.status = "active"

                    self.pipelines.append(pipeline_schema)

    def get_pipelines(self, name: Optional[str] = None) -> list:
        """
         Get all pipelines or If name is not none then fetch specific pipeline
        :param name: pipeline name
        :return:
        """
        self.load_data()
        if name is not None:
            return list(filter(lambda pipeline: pipeline.name == name, self.pipelines))

        return self.pipelines

    def activate_pipeline(self, name: str) -> bool:
        """
        Activate the pipeline by name
        :param name: pipeline name
        :return:Bool
        """
        pipeline = self.get_pipelines(name)
        if len(pipeline) > 0:
            with open(ACTIVE_PIPELINE_FILE, 'w', encoding='utf8') as file:
                yaml.dump({'active_pipeline': name}, file)
            return True
        else:
            return False

    def get_active_pipeline(self) -> tuple:
        """
        Return tuple of active pipeline name and file_path
        :return:
        """
        self.load_data()
        active_pipeline = list(filter(lambda pipeline: pipeline.status == "active", self.pipelines))[0]

        return active_pipeline.name, f"{PIPELINES_DIR}/{active_pipeline.yaml_file}"


class Model:
    """Class hold the Pipeline Object"""
    def __init__(self, pipeline_helper):
        self._pipeline_helper = pipeline_helper
        self._pipeline = None
        self.load()

    def load(self):
        """ Load the active model """
        active_pipeline, active_pipeline_path = self._pipeline_helper.get_active_pipeline()
        self._pipeline = Pipeline.load_from_yaml(Path(active_pipeline_path), pipeline_name=active_pipeline)
        logger.info(f"Loaded pipeline nodes: {self._pipeline.graph.nodes.keys()}")

        return self

    def get_active_pipeline(self):
        """Return the active pipline object"""
        return self._pipeline


pipeline_helper = PipelineHelper(PIPELINES_DIR)
model = Model(pipeline_helper)


def get_pipeline_helper():
    """Return pipelineHelper object"""
    return pipeline_helper


def get_model():
    """Return Model object"""
    return model
