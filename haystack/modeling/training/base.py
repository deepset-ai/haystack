from typing import Optional, Tuple, List

import sys
import shutil
import logging
import dill
import numpy
import torch
from tqdm import tqdm
from pathlib import Path

from haystack.modeling.data_handler.data_silo import DataSilo
from haystack.modeling.evaluation.eval import Evaluator
from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.model.optimization import get_scheduler
from haystack.modeling.utils import GracefulKiller
from haystack.modeling.logger import MLFlowLogger as MlLogger

try:
    from apex import amp
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Can be used to control early stopping with a Trainer class. Any object can be used instead which
    implements the method check_stopping and and provides the attribute save_dir
    """
    def __init__(
            self,
            head: int = 0,
            metric: str = "loss",
            save_dir: Optional[str] = None,
            mode: str = "min",
            patience: int = 0,
            min_delta: float = 0.001,
            min_evals: int = 0,
    ):
        """
        :param head: the prediction head referenced by the metric.
        :param save_dir: the directory where to save the final best model, if None, no saving.
        :param metric: name of dev set metric to monitor (default: loss) to get extracted from the 0th head or
                       a function that extracts a value from the trainer dev evaluation result.
                       NOTE: this is different from the metric to get specified for the processor which defines how
                       to calculate one or more evaluation matric values from prediction/target sets, while this
                       specifies the name of one particular such metric value or a method to calculate that value
                       from the result returned from a processor metric.
        :param mode: "min" or "max"
        :param patience: how many evaluations to wait after the best evaluation to stop
        :param min_delta: minimum difference to a previous best value to count as an improvement.
        :param min_evals: minimum number of evaluations to wait before using eval value
        """
        self.head = head
        self.metric = metric
        self.save_dir = save_dir
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.min_evals = min_evals
        # for more complex modes
        self.eval_values = []  # type: List
        self.n_since_best = None  # type: Optional[int]
        if mode == "min":
            self.best_so_far = 1.0E99
        elif mode == "max":
            self.best_so_far = -1.0E99
        else:
            raise Exception("Mode must be 'min' or 'max'")

    def check_stopping(self, eval_result) -> Tuple[bool, bool, float]:
        """
        Provide the evaluation value for the current evaluation. Returns true if stopping should occur.
        This will save the model, if necessary.

        :param eval_result: the current evaluation result
        :return: a tuple (stopprocessing, savemodel, evalvalue) indicating if processing should be stopped
                 and if the current model should get saved and the evaluation value used.
        """
        if isinstance(self.metric, str):
            eval_value = float(eval_result[self.head][self.metric])
        else:
            eval_value = float(self.metric(eval_result))
        self.eval_values.append(eval_value)
        stopprocessing, savemodel = False, False
        if len(self.eval_values) <= self.min_evals:
            return stopprocessing, savemodel, eval_value
        if self.mode == "min":
            delta = self.best_so_far - eval_value
        else:
            delta = eval_value - self.best_so_far
        if delta > self.min_delta:
            self.best_so_far = eval_value
            self.n_since_best = 0
            if self.save_dir:
                savemodel = True
        else:
            self.n_since_best += 1  # type: ignore
        if self.n_since_best > self.patience:
            stopprocessing = True
        return stopprocessing, savemodel, eval_value


class Trainer:
    """
    Handles the main model training procedure. This includes performing evaluation on the dev set at regular
    intervals during training as well as evaluation on the test set at the end of training.
    """
    def __init__(
        self,
        model,
        optimizer,
        data_silo: DataSilo,
        epochs: int,
        n_gpu: int,
        device,
        lr_schedule=None,
        evaluate_every: int = 100,
        eval_report: bool = True,
        use_amp: Optional[str] = None,
        grad_acc_steps: int = 1,
        local_rank: int = -1,
        early_stopping: Optional[EarlyStopping] = None,
        log_learning_rate: bool = False,
        log_loss_every: int = 10,
        checkpoint_on_sigterm: bool = False,
        checkpoint_every: Optional[int] = None,
        checkpoint_root_dir: Optional[Path] = None,
        checkpoints_to_keep: int = 3,
        from_epoch: int = 0,
        from_step: int = 0,
        global_step: int = 0,
        evaluator_test: bool = True,
        disable_tqdm: bool = False,
        max_grad_norm: float = 1.0
    ):
        """
        :param optimizer: An optimizer object that determines the learning strategy to be used during training
        :param data_silo: A DataSilo object that will contain the train, dev and test datasets as PyTorch DataLoaders
        :param epochs: How many times the training procedure will loop through the train dataset
        :param n_gpu: The number of gpus available for training and evaluation.
        :param device: The device on which the train, dev and test tensors should be hosted. Choose from "cpu" and "cuda".
        :param lr_schedule: An optional scheduler object that can regulate the learning rate of the optimizer
        :param evaluate_every: Perform dev set evaluation after this many steps of training.
        :param eval_report: If evaluate_every is not 0, specifies if an eval report should be generated when evaluating
        :param use_amp: Whether to use automatic mixed precision with Apex. One of the optimization levels must be chosen.
                        "O1" is recommended in almost all cases.
        :param grad_acc_steps: Number of training steps for which the gradients should be accumulated.
                               Useful to achieve larger effective batch sizes that would not fit in GPU memory.
        :param local_rank: Local rank of process when distributed training via DDP is used.
        :param early_stopping: an initialized EarlyStopping object to control early stopping and saving of best models.
        :param log_learning_rate: Whether to log learning rate to Mlflow
        :param log_loss_every: Log current train loss after this many train steps.
        :param checkpoint_on_sigterm: save a checkpoint for the Trainer when a SIGTERM signal is sent. The checkpoint
               can be used to resume training. It is useful in frameworks like AWS SageMaker with Spot instances where
               a SIGTERM notifies to save the training state and subsequently the instance is terminated.
        :param checkpoint_every: save a train checkpoint after this many steps of training.
        :param checkpoint_root_dir: the Path of directory where all train checkpoints are saved. For each individual
               checkpoint, a subdirectory with the name epoch_{epoch_num}_step_{step_num} is created.
        :param checkpoints_to_keep: maximum number of train checkpoints to save.
        :param from_epoch: the epoch number to start the training from. In the case when training resumes from a saved
               checkpoint, it is used to fast-forward training to the last epoch in the checkpoint.
        :param from_step: the step number to start the training from. In the case when training resumes from a saved
               checkpoint, it is used to fast-forward training to the last step in the checkpoint.
        :param global_step: the global step number across the training epochs.
        :param evaluator_test: whether to perform evaluation on the test set
        :param disable_tqdm: Disable tqdm progress bar (helps to reduce verbosity in some environments)
        :param max_grad_norm: Max gradient norm for clipping, default 1.0, set to None to disable
        """
        self.model = model
        self.data_silo = data_silo
        self.epochs = int(epochs)
        self.optimizer = optimizer
        self.evaluate_every = evaluate_every
        self.eval_report = eval_report
        self.evaluator_test = evaluator_test
        self.n_gpu = n_gpu
        self.grad_acc_steps = grad_acc_steps
        self.use_amp = use_amp
        self.lr_schedule = lr_schedule
        self.device = device
        self.local_rank = local_rank
        self.log_params()
        self.early_stopping = early_stopping
        self.log_learning_rate = log_learning_rate
        self.log_loss_every = log_loss_every
        self.disable_tqdm = disable_tqdm
        self.max_grad_norm = max_grad_norm
        self.test_result = None

        if use_amp and not AMP_AVAILABLE:
            raise ImportError(f'Got use_amp = {use_amp}, but cannot find apex. '
                              'Please install Apex if you want to make use of automatic mixed precision. '
                              'https://github.com/NVIDIA/apex')
        self.checkpoint_on_sigterm = checkpoint_on_sigterm
        if checkpoint_on_sigterm:
            self.sigterm_handler = GracefulKiller()  # type: Optional[GracefulKiller]
        else:
            self.sigterm_handler = None
        self.checkpoint_root_dir = checkpoint_root_dir
        self.checkpoints_to_keep = checkpoints_to_keep
        self.checkpoint_every = checkpoint_every
        if self.checkpoint_every and not checkpoint_root_dir:
            raise Exception("checkpoint_path needs to be supplied when using checkpoint_every.")
        if checkpoint_on_sigterm and not checkpoint_root_dir:
            raise Exception("checkpoint_path needs to be supplied when using checkpoint_on_sigterm.")

        self.from_epoch = from_epoch
        self.from_step = from_step
        self.global_step = global_step

    def train(self):
        """
        Perform the training procedure.

        The training is visualized by a progress bar. It counts the epochs in a zero based manner.
        For example, when you specify ``epochs=20`` it starts to count from 0 to 19.

        If trainer evaluates the model with a test set the result of the
        evaluation is stored in ``test_result``.

        :return: Returns the model after training. When you do ``early_stopping``
            with a ``save_dir`` the best model is loaded and returned.
        """
        # connect the prediction heads with the right output from processor
        self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)
        # Check that the tokenizer(s) fits the language model(s)
        if hasattr(self.model, "language_model3"):
            self.model.verify_vocab_size(vocab_size1=len(self.data_silo.processor.query_tokenizer),
                                         vocab_size2=len(self.data_silo.processor.passage_tokenizer),
                                         vocab_size3=len(self.data_silo.processor.table_tokenizer))
        elif hasattr(self.model, "language_model2"):
            self.model.verify_vocab_size(vocab_size1=len(self.data_silo.processor.query_tokenizer),
                                         vocab_size2=len(self.data_silo.processor.passage_tokenizer))
        else:
            self.model.verify_vocab_size(vocab_size=len(self.data_silo.processor.tokenizer))
        self.model.train()

        do_stopping = False
        evalnr = 0
        loss = 0
        resume_from_step = self.from_step

        for epoch in range(self.from_epoch, self.epochs):
            early_break = False
            self.from_epoch = epoch
            train_data_loader = self.data_silo.get_data_loader("train")
            progress_bar = tqdm(train_data_loader, disable=self.local_rank not in [0, -1] or self.disable_tqdm)
            for step, batch in enumerate(progress_bar):
                # when resuming training from a checkpoint, we want to fast forward to the step of the checkpoint
                if resume_from_step and step <= resume_from_step:
                    if step % 10000 == 0:
                        logger.info(f"Skipping {step} out of {resume_from_step} steps ...")
                    if resume_from_step == step:
                        logger.info(f"Finished skipping {resume_from_step} steps ...")
                        resume_from_step = None
                    else:
                        continue

                progress_bar.set_description(f"Train epoch {epoch}/{self.epochs-1} (Cur. train loss: {loss:.4f})")

                # Only for distributed training: we need to ensure that all ranks still have a batch left for training
                if self.local_rank != -1:
                    if not self._all_ranks_have_data(has_data=1, step=step):
                        early_break = True
                        break

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}
                # Forward & backward pass through model
                logits = self.model.forward(**batch)
                per_sample_loss = self.model.logits_to_loss(logits=logits, global_step=self.global_step, **batch)
                loss = self.backward_propagate(per_sample_loss, step)

                # Perform  evaluation
                if self.evaluate_every != 0 \
                        and self.global_step % self.evaluate_every == 0 \
                        and self.global_step != 0\
                        and self.local_rank in [0,-1]:
                    dev_data_loader = self.data_silo.get_data_loader("dev")
                    if dev_data_loader is not None:
                        evaluator_dev = Evaluator(
                            data_loader=dev_data_loader, tasks=self.data_silo.processor.tasks, device=self.device, report=self.eval_report
                        )
                        evalnr += 1
                        result = evaluator_dev.eval(self.model)
                        evaluator_dev.log_results(result, "Dev", self.global_step)
                        if self.early_stopping:
                            do_stopping, save_model, eval_value = self.early_stopping.check_stopping(result)
                            if save_model:
                                logger.info(
                                    "Saving current best model to {}, eval={}".format(
                                        self.early_stopping.save_dir, eval_value))
                                self.model.save(self.early_stopping.save_dir)
                                self.data_silo.processor.save(self.early_stopping.save_dir)
                            if do_stopping:
                                # log the stopping
                                logger.info("STOPPING EARLY AT EPOCH {}, STEP {}, EVALUATION {}".format(epoch, step, evalnr))
                if do_stopping:
                    break

                self.global_step += 1
                self.from_step = step + 1

                # save the current state as a checkpoint before exiting if a SIGTERM signal is received
                if self.sigterm_handler and self.sigterm_handler.kill_now:
                    logger.info("Received a SIGTERM signal. Saving the current train state as a checkpoint ...")
                    if self.local_rank in [0, -1]:
                        self._save()
                        torch.distributed.destroy_process_group()
                        sys.exit(0)

                # save a checkpoint and continue train
                if self.checkpoint_every and step % self.checkpoint_every == 0:
                    if self.local_rank in [0, -1]:
                        self._save()
                    # Let other ranks wait until rank 0 has finished saving
                    if self.local_rank != -1:
                        torch.distributed.barrier()

            if do_stopping:
                break

            # Only for distributed training: we need to ensure that all ranks still have a batch left for training
            if self.local_rank != -1 and not early_break:
                self._all_ranks_have_data(has_data=False)

        # With early stopping we want to restore the best model
        if self.early_stopping and self.early_stopping.save_dir:
            logger.info("Restoring best model so far from {}".format(self.early_stopping.save_dir))
            lm_name = self.model.language_model.name
            self.model = AdaptiveModel.load(self.early_stopping.save_dir, self.device, lm_name=lm_name)
            self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Eval on test set
        if self.evaluator_test and self.local_rank in [0, -1]:
            test_data_loader = self.data_silo.get_data_loader("test")
            if test_data_loader is not None:
                evaluator_test = Evaluator(
                    data_loader=test_data_loader, tasks=self.data_silo.processor.tasks, device=self.device
                )
                self.test_result = evaluator_test.eval(self.model)
                evaluator_test.log_results(self.test_result, "Test", self.global_step)
        return self.model

    def backward_propagate(self, loss: torch.Tensor, step: int):
        loss = self.adjust_loss(loss)
        if self.global_step % self.log_loss_every == 0 and self.local_rank in [-1, 0]:
            if self.local_rank in [-1, 0]:
                MlLogger.log_metrics(
                    {"Train_loss_total": float(loss.detach().cpu().numpy())},
                    step=self.global_step,
                )
                if self.log_learning_rate:
                    MlLogger.log_metrics({"learning_rate": self.lr_schedule.get_last_lr()[0]},
                                         step=self.global_step)
        if self.use_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if step % self.grad_acc_steps == 0:
            if self.max_grad_norm is not None:
                if self.use_amp:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_schedule:
                self.lr_schedule.step()
        return loss

    def adjust_loss(self, loss: torch.Tensor):
        loss = loss.mean()
        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps
        return loss

    def log_params(self):
        params = {"epochs": self.epochs, "n_gpu": self.n_gpu, "device": self.device}
        MlLogger.log_params(params)

    @classmethod
    def create_or_load_checkpoint(cls, data_silo: DataSilo, checkpoint_root_dir: Path, model, optimizer,
                                  local_rank: int = -1, resume_from_checkpoint: str = "latest", **kwargs):
        """
        Try loading a saved Trainer checkpoint. If no checkpoint found, it creates a new instance of Trainer.

        :param data_silo: A DataSilo object that will contain the train, dev and test datasets as PyTorch DataLoaders
        :param checkpoint_root_dir: Path of the directory where all train checkpoints are saved. Each individual
               checkpoint is stored in a sub-directory under it.
        :param resume_from_checkpoint: the checkpoint name to start training from, e.g., "epoch_1_step_4532". It
               defaults to "latest", using the checkpoint with the highest train steps.
        """
        checkpoint_to_load = None
        if checkpoint_root_dir:
            if checkpoint_root_dir.exists():
               if resume_from_checkpoint == "latest":
                   saved_checkpoints = cls._get_checkpoints(checkpoint_root_dir)
                   if saved_checkpoints:
                       checkpoint_to_load = saved_checkpoints[0]  # latest checkpoint
                   else:
                       checkpoint_to_load = None
               else:
                   checkpoint_to_load = checkpoint_root_dir / resume_from_checkpoint

        if checkpoint_to_load:
            #TODO load empty model class from config instead of passing here?
            trainer = cls._load_checkpoint(path=checkpoint_to_load, data_silo=data_silo,
                                           model=model, optimizer=optimizer, local_rank=local_rank)
            logging.info(f"Resuming training from the train checkpoint at {checkpoint_to_load} ...")
        else:
            logging.info(f"No train checkpoints found. Starting a new training ...")
            trainer = Trainer(data_silo=data_silo, model=model, optimizer=optimizer, local_rank=local_rank,
                              checkpoint_root_dir=checkpoint_root_dir, **kwargs)
        return trainer

    @classmethod
    def _load_checkpoint(cls, path: Path, data_silo: DataSilo, model, optimizer, local_rank: int = -1):
        """
        Load the train checkpoint at given path.

        :param path: The checkpoint path is subdirectory under checkpoint_root_dir. The individual checkpoint dirs have
               a default naming convention of "epoch_{epoch_num}_step_{step_num}".
        :param data_silo: A DataSilo object that will contain the train, dev and test datasets as PyTorch DataLoaders
        """
        if not path.exists():
            raise Exception(f"The checkpoint path {path} does not exists.")

        # In distributed mode, we save the model only once from process 0 (using cuda:0)
        # At loading time, we need to load the model to the current cuda device (instead of back to cuda:0)
        # Note: This assumes exactly one GPU per process (as recommended by PyTorch)
        if local_rank == -1:
            map_location = None
        else:
            device = torch.device(f"cuda:{local_rank}")
            map_location = {'cuda:0': f'cuda:{local_rank}'}

        trainer_checkpoint = torch.load(path / "trainer", map_location=map_location)
        trainer_state_dict = trainer_checkpoint["trainer_state"]
        if local_rank != -1:
            trainer_state_dict["device"] = device
            trainer_state_dict["local_rank"] = local_rank

        # Just setting seeds is not sufficient to have deterministic results when resuming
        # training from a checkpoint. Additionally, the previous states of Random Number
        # Generators also need to be restored from the saved checkpoint.
        numpy_rng_state = trainer_checkpoint["numpy_rng_state"]
        numpy.random.set_state(numpy_rng_state)
        rng_state = trainer_checkpoint["rng_state"]
        cuda_rng_state = trainer_checkpoint["cuda_rng_state"]
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

        model.load_state_dict(trainer_checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(trainer_checkpoint["optimizer_state"])

        scheduler_state_dict = trainer_checkpoint["scheduler_state"]
        scheduler_opts = trainer_checkpoint["scheduler_opts"]
        scheduler = get_scheduler(optimizer, scheduler_opts)
        scheduler.load_state_dict(scheduler_state_dict)

        trainer = Trainer(
            data_silo=data_silo,
            model=model,
            optimizer=optimizer,
            lr_schedule=scheduler,
            **trainer_state_dict
        )

        logger.info(f"Loaded a train checkpoint from {path}")
        return trainer

    @classmethod
    def _get_checkpoints(cls, checkpoint_root_dir: Path):
        """
        Get a list of checkpoint dirs sorted by the number of training steps.
        """
        dirs = [d for d in checkpoint_root_dir.iterdir() if d.is_dir() and d.name.startswith("epoch")]

        checkpoints_with_epoch_and_step = []  # list of tuple(checkpoint_dir, epoch, step)
        for d in dirs:
            epoch, step = [int(s) for s in str(d).split("_") if s.isdigit()]
            checkpoints_with_epoch_and_step.append((d, epoch, step))

        sorted_checkpoints_with_epoch_and_step = sorted(checkpoints_with_epoch_and_step,
                                                        key=lambda tup: (tup[1], tup[2]),  # sort by epoch and step
                                                        reverse=True)
        sorted_checkpoints = [tup[0] for tup in sorted_checkpoints_with_epoch_and_step]

        return sorted_checkpoints

    def _save(self):
        """
        Save a train checkpoint at the Trainer's checkpoint_path.

        Some objects(eg, scheduler) in the Trainer are not serializable using the Pickle module. For these objects,
        the state_dict is stored for the checkpoint, that can be used to reconstruct a similar state upon resuming
        train from the checkpoint.

        #TODO The model is currently saved as a whole serialized object. The disadvantage of this approach is that it is
        bound to specifics Python version, haystack version, directory structures etc. A more modular and reusable approach
        is to save using AdaptiveModel's save() method where the model and the state_dict are stored separately.

        # TODO custom defined evaluators are not saved in the checkpoint.
        """
        logger.info("Saving a train checkpoint ...")
        checkpoint_path = self.checkpoint_root_dir / "checkpoint_in_progress"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        trainer_state_dict = self._get_state_dict()

        # save as a regular AdaptiveModel (e.g. for down-stream eval during training from scratch)
        self.model.save(checkpoint_path)

        # save all state dicst (incl. the model) to have full reproducibility
        torch.save(
            {
                "trainer_state": trainer_state_dict,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_opts": self.lr_schedule.opts,
                "scheduler_state": self.lr_schedule.state_dict(),
                "numpy_rng_state": numpy.random.get_state(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
            },
            checkpoint_path / "trainer",
            pickle_module=dill,
        )

        checkpoint_name = f"epoch_{self.from_epoch}_step_{self.from_step-1}"
        checkpoint_path.replace(Path(checkpoint_path.parent) / checkpoint_name)

        saved_checkpoints = self._get_checkpoints(self.checkpoint_root_dir)
        if len(saved_checkpoints) > self.checkpoints_to_keep:
            for cp in saved_checkpoints[self.checkpoints_to_keep:]:
                shutil.rmtree(cp)

        logger.info(f"Saved a training checkpoint after {checkpoint_name}")

    def _get_state_dict(self):
        """
        Serializable state dictionary of a Trainer object
        """
        state_dict = {
            "evaluate_every": self.evaluate_every,
            "n_gpu": self.n_gpu,
            "grad_acc_steps": self.grad_acc_steps,
            "device": self.device,
            "local_rank": self.local_rank,
            "early_stopping": self.early_stopping,
            "epochs": self.epochs,
            "checkpoint_on_sigterm": self.checkpoint_on_sigterm,
            "checkpoint_root_dir": self.checkpoint_root_dir,
            "checkpoint_every": self.checkpoint_every,
            "checkpoints_to_keep": self.checkpoints_to_keep,
            "from_epoch": self.from_epoch,
            "from_step": self.from_step,
            "global_step": self.global_step,
            "log_learning_rate": self.log_learning_rate,
            "log_loss_every": self.log_loss_every,
            "disable_tqdm": self.disable_tqdm
        }

        return state_dict

    def _all_ranks_have_data(self, has_data: bool, step: Optional[int] = None):
        """
        Verify in distributed training if all ranks still have data left. We send a "1" from here if this rank has data
        and a "0" if a process has none .
        If all ranks have data, they'll all send a 1, our sum equals world_size and we continue training.
        If not, we must break the loop for those who still have data to synchronize again.
        :param has_data: bool, whether the current rank has still data
        :param step: int, current step (only used for logging)
        :return: bool, whether all ranks have training data left
        """
        if has_data:
            ranks_with_data = torch.ones(1).to(self.device)
        else:
            ranks_with_data = torch.zeros(1).to(self.device)

        torch.distributed.all_reduce(ranks_with_data, op=torch.distributed.ReduceOp.SUM)

        if ranks_with_data < torch.distributed.get_world_size():
            if step is not None:
                logger.info(
                    f"Stopping epoch {self.from_epoch} at step {step} for rank {self.local_rank} since at least one other rank "
                    f"(~ one GPU) in distributed training doesn't have any more batches... ")
            return False
        else:
            return True
