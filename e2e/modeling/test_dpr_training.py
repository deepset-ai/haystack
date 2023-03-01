##################################################
# FIXME: No test here seems to have any asserts! #
##################################################
import pytest

from haystack.nodes.retriever.dense import DensePassageRetriever
from haystack.utils.early_stopping import EarlyStopping


from ..conftest import SAMPLES_PATH


@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_dpr_training(document_store, tmp_path):
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        max_seq_len_query=8,
        max_seq_len_passage=8,
    )

    save_dir = f"{tmp_path}/test_dpr_training"
    retriever.train(
        data_dir=str(SAMPLES_PATH / "dpr"),
        train_filename="sample.json",
        dev_filename="sample.json",
        test_filename="sample.json",
        n_epochs=1,
        batch_size=1,
        grad_acc_steps=1,
        save_dir=save_dir,
        evaluate_every=10,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
    )


@pytest.mark.parametrize("document_store", ["memory"], indirect=True)
def test_dpr_training_with_earlystopping(document_store, tmp_path):
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        max_seq_len_query=8,
        max_seq_len_passage=8,
    )

    save_dir = f"{tmp_path}/test_dpr_training"
    retriever.train(
        data_dir=str(SAMPLES_PATH / "dpr"),
        train_filename="sample.json",
        dev_filename="sample.json",
        test_filename="sample.json",
        n_epochs=1,
        batch_size=1,
        grad_acc_steps=1,
        save_dir=save_dir,
        evaluate_every=1,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
        early_stopping=EarlyStopping(save_dir=save_dir),
    )


# TODO fix CI errors (test pass locally or on AWS, next steps: isolate PyTorch versions once FARM dependency is removed)
# def test_dpr_training():
#     batch_size = 1
#     n_epochs = 1
#     distributed = False  # enable for multi GPU training via DDP
#     evaluate_every = 1
#     question_lang_model = "microsoft/MiniLM-L12-H384-uncased"
#     passage_lang_model = "microsoft/MiniLM-L12-H384-uncased"
#     do_lower_case = True
#     use_fast = True
#     similarity_function = "dot_product"
#
#     device, n_gpu = initialize_device_settings(use_cuda=False)
#
#     query_tokenizer = get_tokenizer(pretrained_model_name_or_path=question_lang_model,
#                                      do_lower_case=do_lower_case, use_fast=use_fast)
#     passage_tokenizer = get_tokenizer(pretrained_model_name_or_path=passage_lang_model,
#                                        do_lower_case=do_lower_case, use_fast=use_fast)
#     label_list = ["hard_negative", "positive"]
#
#     processor = TextSimilarityProcessor(query_tokenizer=query_tokenizer,
#                                         passage_tokenizer=passage_tokenizer,
#                                         max_seq_len_query=10,
#                                         max_seq_len_passage=10,
#                                         label_list=label_list,
#                                         metric="text_similarity_metric",
#                                         data_dir="samples/dpr/",
#                                         train_filename="sample.json",
#                                         dev_filename="sample.json",
#                                         test_filename=None,
#                                         embed_title=True,
#                                         num_hard_negatives=1,
#                                         dev_split=0,
#                                         max_samples=2)
#
#     data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)
#
#     question_language_model = get_language_model(pretrained_model_name_or_path=question_lang_model,
#                                                  language_model_class="DPRQuestionEncoder")
#     passage_language_model = get_language_model(pretrained_model_name_or_path=passage_lang_model,
#                                                 language_model_class="DPRContextEncoder")
#
#     prediction_head = TextSimilarityHead(similarity_function=similarity_function)
#
#     model = BiAdaptiveModel(
#         language_model1=question_language_model,
#         language_model2=passage_language_model,
#         prediction_heads=[prediction_head],
#         embeds_dropout_prob=0.1,
#         lm1_output_types=["per_sequence"],
#         lm2_output_types=["per_sequence"],
#         device=device,
#     )
#
#     model, optimizer, lr_schedule = initialize_optimizer(
#         model=model,
#         learning_rate=1e-5,
#         optimizer_opts={"name": "TransformersAdamW", "correct_bias": True, "weight_decay": 0.0, \
#                         "eps": 1e-08},
#         schedule_opts={"name": "LinearWarmup", "num_warmup_steps": 100},
#         n_batches=len(data_silo.loaders["train"]),
#         n_epochs=n_epochs,
#         grad_acc_steps=1,
#         device=device,
#         distributed=distributed
#     )
#
#     trainer = Trainer(
#         model=model,
#         optimizer=optimizer,
#         data_silo=data_silo,
#         epochs=n_epochs,
#         n_gpu=n_gpu,
#         lr_schedule=lr_schedule,
#         evaluate_every=evaluate_every,
#         device=device,
#     )
#
#     trainer.train()
#
#     ######## save and load model again
#     save_dir = Path("testsave/dpr-model")
#     model.save(save_dir)
#     del model
#
#     model2 = BiAdaptiveModel.load(save_dir, device=device)
#     model2, optimizer2, lr_schedule = initialize_optimizer(
#         model=model2,
#         learning_rate=1e-5,
#         optimizer_opts={"name": "TransformersAdamW", "correct_bias": True, "weight_decay": 0.0, \
#                         "eps": 1e-08},
#         schedule_opts={"name": "LinearWarmup", "num_warmup_steps": 100},
#         n_batches=len(data_silo.loaders["train"]),
#         n_epochs=n_epochs,
#         grad_acc_steps=1,
#         device=device,
#         distributed=distributed
#     )
#     trainer2 = Trainer(
#         model=model2,
#         optimizer=optimizer,
#         data_silo=data_silo,
#         epochs=n_epochs,
#         n_gpu=n_gpu,
#         lr_schedule=lr_schedule,
#         evaluate_every=evaluate_every,
#         device=device,
#     )
#
#     trainer2.train()
