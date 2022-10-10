import tempfile
from dataclasses import dataclass, field
import datetime
from typing import Optional
from matplotlib.figure import Figure
import wandb
import pandas as pd


@dataclass
class WandBLogger:
    """
    :param project_name e.g. monopoly_faq
    :param job_name e.g. train, finetune, evaluate
    :param run_id: optional, if you want to resume a previous run
    """

    project_name: str
    job_name: str
    run_id: Optional[str] = None
    run: wandb.run = field(init=False)

    def __post_init__(self):
        self.run = wandb.init(
            project=self.project_name,
            entity="boardgame-rules-explainer",
            job_type=self.job_name,
            group=datetime.datetime.utcnow().strftime("%Y%m%d"),
            resume="allow",
            id=self.run_id,
        )

    @staticmethod
    def commit_logs():
        wandb.finish()

    @staticmethod
    def log_config(config: dict):
        wandb.config = config

    def log_metrics(self, metrics: dict):
        self.run.summary.update(metrics)

    def log_table(self, df: pd.DataFrame, title: str):
        tbl = wandb.Table(data=df)
        self.run.log({title: tbl})

    @staticmethod
    def log_figure(figure: Figure, name: str, description: str, meta_data: Optional[dict] = None):
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            figure.savefig(f.name)

            artifact = wandb.Artifact(name=name, type="plot", description=description, metadata=meta_data)
            artifact.add_file(f.name)
            wandb.log_artifact(artifact)

    @staticmethod
    def log_model(path_to_model_artifacts: str, name: str, description: str, meta_data: Optional[dict] = None):
        artifact = wandb.Artifact(name=name, type="model", description=description, metadata=meta_data)
        artifact.add_dir(path_to_model_artifacts)
        wandb.log_artifact(artifact)
