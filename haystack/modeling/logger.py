import logging
import mlflow
from requests.exceptions import ConnectionError


logger = logging.getLogger(__name__)


class BaseMLLogger:
    """
    Base class for tracking experiments.

    This class can be extended to implement custom logging backends like MLFlow, Tensorboard, or Sacred.
    """
    disable_logging = False

    def __init__(self, tracking_uri, **kwargs):
        self.tracking_uri = tracking_uri

    def init_experiment(self, tracking_uri):
        raise NotImplementedError()

    @classmethod
    def log_metrics(cls, metrics, step):
        raise NotImplementedError()

    @classmethod
    def log_artifacts(cls, self):
        raise NotImplementedError()

    @classmethod
    def log_params(cls, params):
        raise NotImplementedError()


class StdoutLogger(BaseMLLogger):
    """ Minimal logger printing metrics and params to stdout.
    Useful for services like AWS SageMaker, where you parse metrics from the actual logs"""

    def init_experiment(self, experiment_name, run_name=None, nested=True):
        logger.info(f"\n **** Starting experiment '{experiment_name}' (Run: {run_name})  ****")

    @classmethod
    def log_metrics(cls, metrics, step):
        logger.info(f"Logged metrics at step {step}: \n {metrics}")

    @classmethod
    def log_params(cls, params):
        logger.info(f"Logged parameters: \n {params}")

    @classmethod
    def log_artifacts(cls, dir_path, artifact_path=None):
        raise NotImplementedError

    @classmethod
    def end_run(cls):
        logger.info(f"**** End of Experiment **** ")


class MLFlowLogger(BaseMLLogger):
    """
    Logger for MLFlow experiment tracking.
    """

    def init_experiment(self, experiment_name, run_name=None, nested=True):
        if not self.disable_logging:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(experiment_name)
                mlflow.start_run(run_name=run_name, nested=nested)
            except ConnectionError:
                raise Exception(
                    f"MLFlow cannot connect to the remote server at {self.tracking_uri}.\n"
                    f"MLFlow also supports logging runs locally to files. Set the MLFlowLogger "
                    f"tracking_uri to an empty string to use that."
                )

    @classmethod
    def log_metrics(cls, metrics, step):
        if not cls.disable_logging:
            try:
                mlflow.log_metrics(metrics, step=step)
            except ConnectionError:
                logger.warning(f"ConnectionError in logging metrics to MLFlow.")
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")

    @classmethod
    def log_params(cls, params):
        if not cls.disable_logging:
            try:
                mlflow.log_params(params)
            except ConnectionError:
                logger.warning("ConnectionError in logging params to MLFlow")
            except Exception as e:
                logger.warning(f"Failed to log params: {e}")

    @classmethod
    def log_artifacts(cls, dir_path, artifact_path=None):
        if not cls.disable_logging:
            try:
                mlflow.log_artifacts(dir_path, artifact_path)
            except ConnectionError:
                logger.warning(f"ConnectionError in logging artifacts to MLFlow")
            except Exception as e:
                logger.warning(f"Failed to log artifacts: {e}")

    @classmethod
    def end_run(cls):
        if not cls.disable_logging:
            mlflow.end_run()

    @classmethod
    def disable(cls):
        logger.info("ML Logging is turned off. No parameters, metrics or artifacts will be logged to MLFlow.")
        cls.disable_logging = True


class TensorBoardLogger(BaseMLLogger):
    """
    PyTorch TensorBoard Logger
    """

    def __init__(self, **kwargs):
        from tensorboardX import SummaryWriter
        TensorBoardLogger.summary_writer = SummaryWriter()
        super().__init__(**kwargs)

    @classmethod
    def log_metrics(cls, metrics, step):
        for key, value in metrics.items():
            TensorBoardLogger.summary_writer.add_scalar(
                tag=key, scalar_value=value, global_step=step
            )

    @classmethod
    def log_params(cls, params):
        for key, value in params.items():
            TensorBoardLogger.summary_writer.add_text(tag=key, text_string=str(value))

