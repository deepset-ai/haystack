def tutorial9_dpr_training():
    # Training Your Own "Dense Passage Retrieval" Model

    # Here are some imports that we'll need

    from haystack.nodes import DensePassageRetriever
    from haystack.utils import fetch_archive_from_http
    from haystack.document_stores import InMemoryDocumentStore

    # Download original DPR data
    # WARNING: the train set is 7.4GB and the dev set is 800MB

    doc_dir = "data/dpr_training/"

    s3_url_train = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"
    s3_url_dev = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"

    fetch_archive_from_http(s3_url_train, output_dir=doc_dir + "train/")
    fetch_archive_from_http(s3_url_dev, output_dir=doc_dir + "dev/")

    ## Option 1: Training DPR from Scratch

    # Here are the variables to specify our training data, the models that we use to initialize DPR
    # and the directory where we'll be saving the model

    doc_dir = "data/dpr_training/"

    train_filename = "train/biencoder-nq-train.json"
    dev_filename = "dev/biencoder-nq-dev.json"

    query_model = "bert-base-uncased"
    passage_model = "bert-base-uncased"

    save_dir = "../saved_models/dpr"

    # ## Option 2: Finetuning DPR
    #
    # # Here are the variables you might want to use instead of the set above
    # # in order to perform pretraining
    #
    # doc_dir = "PATH_TO_YOUR_DATA_DIR"
    # train_filename = "TRAIN_FILENAME"
    # dev_filename = "DEV_FILENAME"
    #
    # query_model = "facebook/dpr-question_encoder-single-nq-base"
    # passage_model = "facebook/dpr-ctx_encoder-single-nq-base"
    #
    # save_dir = "..saved_models/dpr"

    ## Initialize DPR model

    retriever = DensePassageRetriever(
        document_store=InMemoryDocumentStore(),
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
        max_seq_len_query=64,
        max_seq_len_passage=256
    )

    # Start training our model and save it when it is finished

    retriever.train(
        data_dir=doc_dir,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=dev_filename,
        n_epochs=1,
        batch_size=16,
        grad_acc_steps=8,
        save_dir=save_dir,
        evaluate_every=3000,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1
    )

    ## Loading

    reloaded_retriever = DensePassageRetriever.load(load_dir=save_dir, document_store=None)

if __name__ == "__main__":
    tutorial9_dpr_training()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/