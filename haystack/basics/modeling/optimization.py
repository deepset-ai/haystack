from importlib import import_module
import logging
import inspect
from haystack.basics.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


def get_scheduler(optimizer, opts):
    """ Get the scheduler based on dictionary with options. Options are passed to the scheduler constructor.

    :param optimizer: optimizer whose learning rate to control
    :param opts: dictionary of args to be passed to constructor of schedule
    :return: created scheduler
    """
    schedule_name = opts.get('name')
    try:
        sched_constructor = getattr(import_module('torch.optim.lr_scheduler'), schedule_name)
    except AttributeError:
        try:
            # The method names in transformers became quite long and unhandy.
            # for convenience we offer usage of shorter alias (e.g. "LinearWarmup")
            scheduler_translations = {"LinearWarmup": "get_linear_schedule_with_warmup",
                                      "ConstantWarmup": "get_constant_schedule_with_warmup",
                                      "Constant": "get_constant_schedule",
                                      "CosineWarmup": "get_cosine_schedule_with_warmup",
                                     "CosineWarmupWithRestarts": "get_cosine_with_hard_restarts_schedule_with_warmup"
            }
            if schedule_name in scheduler_translations.keys():
                schedule_name = scheduler_translations[schedule_name]
            # in contrast to torch, we actually get here a method and not a class
            sched_constructor = getattr(import_module('transformers.optimization'), schedule_name)
        except AttributeError:
            raise AttributeError(f"Scheduler '{schedule_name}' not found in 'torch' or 'transformers'")

    logger.info(f"Using scheduler '{schedule_name}'")

    # get supported args of constructor
    allowed_args = inspect.signature(sched_constructor).parameters.keys()

    # convert from warmup proportion to steps if required
    if 'num_warmup_steps' in allowed_args and 'num_warmup_steps' not in opts and 'warmup_proportion' in opts:
        opts['num_warmup_steps'] = int(opts["warmup_proportion"] * opts["num_training_steps"])
        MlLogger.log_params({"warmup_proportion": opts["warmup_proportion"]})

    # only pass args that are supported by the constructor
    constructor_opts = {k: v for k, v in opts.items() if k in allowed_args}

    # Logging
    logger.info(f"Loading schedule `{schedule_name}`: '{constructor_opts}'")
    MlLogger.log_params(constructor_opts)
    MlLogger.log_params({"schedule_name": schedule_name})

    scheduler = sched_constructor(optimizer, **constructor_opts)
    scheduler.opts = opts  # save the opts with the scheduler to use in load/save
    return scheduler


