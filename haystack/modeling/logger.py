from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, Union
import mlflow
from requests.exceptions import ConnectionError

from haystack.modeling.utils import flatten_dict


logger = logging.getLogger(__name__)


class BaseExperimentTracker(ABC):
    """
    Base class for tracking experiments.

    This class can be extended to implement custom logging backends like MLFlow, Tensorboard, or Sacred.
    """

    @abstractmethod
    def init_experiment(self, experiment_name: str, run_name: str = None, tags: Dict[str, Any] = None, nested: bool = False):
        raise NotImplementedError()
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        raise NotImplementedError()

    @abstractmethod
    def log_artifacts(self, dir_path: Union[str, Path], artifact_path: str = None):
        raise NotImplementedError()

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def end_run(self):
        raise NotImplementedError()


class NoExperimentTracker(BaseExperimentTracker):
    def init_experiment(self, experiment_name: str, run_name: str = None, tags: Dict[str, Any] = None, nested: bool = False):
        pass

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        pass

    def log_artifacts(self, dir_path: Union[str, Path], artifact_path: str = None):
        pass

    def log_params(self, params: Dict[str, Any]):
        pass

    def end_run(self):
        pass


class ExperimentTracker:
    """
    Facade for tracking experiments.
    """
    tracker: BaseExperimentTracker = NoExperimentTracker()

    @classmethod
    def init_experiment(cls, experiment_name: str, run_name: str = None, tags: Dict[str, Any] = None, nested: bool = False):
        cls.tracker.init_experiment(experiment_name=experiment_name, run_name=run_name, tags=tags, nested=nested)

    @classmethod
    def log_metrics(cls, metrics: Dict[str, Any], step: int):
        cls.tracker.log_metrics(metrics=metrics, step=step)

    @classmethod
    def log_artifacts(cls, dir_path: Union[str, Path], artifact_path: str = None):
        cls.tracker.log_artifacts(dir_path=dir_path, artifact_path=artifact_path)

    @classmethod
    def log_params(cls, params: Dict[str, Any]):
        cls.tracker.log_params(params=params)

    @classmethod
    def end_run(cls):
        cls.tracker.end_run()

    @classmethod
    def set_tracker(cls, tracker: BaseExperimentTracker):
        cls.tracker = tracker


class StdoutExperimentTracker(BaseExperimentTracker):
    """Minimal logger printing metrics and params to stdout.
    Useful for services like AWS SageMaker, where you parse metrics from the actual logs"""

    def init_experiment(self, experiment_name: str, run_name: str = None, tags: Dict[str, Any] = None, nested: bool = False):
        logger.info(f"\n **** Starting experiment '{experiment_name}' (Run: {run_name})  ****")

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        logger.info(f"Logged metrics at step {step}: \n {metrics}")

    def log_params(self, params: Dict[str, Any]):
        logger.info(f"Logged parameters: \n {params}")

    def log_artifacts(self, dir_path: Union[str, Path], artifact_path: str = None):
        logger.warning(f"Cannot log artifacts with StdoutLogger: \n {dir_path}")

    def end_run(self):
        logger.info(f"**** End of Experiment **** ")


class MLFlowExperimentTracker(BaseExperimentTracker):
    """
    Logger for MLFlow experiment tracking.
    """
    def __init__(self, tracking_uri: str) -> None:
        super().__init__()
        self.tracking_uri = tracking_uri

    def init_experiment(
        self, experiment_name: str, run_name: str = None, tags: Dict[str, Any] = None, nested: bool = False
    ):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name, nested=nested, tags=tags)
        except ConnectionError:
            raise Exception(
                f"MLFlow cannot connect to the remote server at {self.tracking_uri}.\n"
                f"MLFlow also supports logging runs locally to files. Set the MLFlowLogger "
                f"tracking_uri to an empty string to use that."
            )

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        try:
            metrics = flatten_dict(metrics)
            mlflow.log_metrics(metrics, step=step)
        except ConnectionError:
            logger.warning(f"ConnectionError in logging metrics to MLFlow.")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_params(self, params: Dict[str, Any]):
        try:
            params = flatten_dict(params)
            mlflow.log_params(params)
        except ConnectionError:
            logger.warning("ConnectionError in logging params to MLFlow")
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    def log_artifacts(self, dir_path: Union[str, Path], artifact_path: str = None):
        try:
            mlflow.log_artifacts(dir_path, artifact_path)
        except ConnectionError:
            logger.warning(f"ConnectionError in logging artifacts to MLFlow")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")

    def end_run(self):
        mlflow.end_run()
