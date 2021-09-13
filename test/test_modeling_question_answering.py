import logging
from pathlib import Path

from haystack.modeling.data_handler.data_silo import DataSilo
from haystack.modeling.data_handler.processor import SquadProcessor
from haystack.modeling.model.adaptive_model import AdaptiveModel
from haystack.modeling.model.language_model import LanguageModel
from haystack.modeling.model.optimization import initialize_optimizer
from haystack.modeling.model.prediction_head import QuestionAnsweringHead
from haystack.modeling.model.tokenization import Tokenizer
from haystack.modeling.training.base import Trainer
from haystack.modeling.utils import set_all_seeds, initialize_device_settings


def test_training(caplog=None):
    if caplog:
        caplog.set_level(logging.CRITICAL)

    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    batch_size = 2
    n_epochs = 1
    evaluate_every = 4
    base_LM_model = "distilbert-base-uncased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=base_LM_model,
        do_lower_case=True,
        use_fast=True  # TODO parametrize this to test slow as well
    )
    label_list = ["start_token", "end_token"]
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=256,
        doc_stride=10,
        max_query_length=6,
        train_filename="train-sample.json",
        dev_filename="dev-sample.json",
        test_filename=None,
        data_dir=Path("samples/qa"),
        label_list=label_list,
        metric="squad"
    )

    data_silo = DataSilo(processor=processor, batch_size=batch_size, max_processes=1)
    language_model = LanguageModel.load(base_LM_model)
    prediction_head = QuestionAnsweringHead()
    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_token"],
        device=device,
    )

    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        # optimizer_opts={'name': 'AdamW', 'lr': 2E-05},
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs,
        device=device
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device
    )
    trainer.train()

    assert type(model) == AdaptiveModel
    assert type(processor) == SquadProcessor


if __name__ == "__main__":
    test_training()
