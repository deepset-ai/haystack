

#%%

from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.dense import DensePassageRetriever
import wget
import gzip

#%%
from haystack.preprocessor.utils import fetch_archive_from_http


doc_dir = "data/dpr_training"
train_filename = "biencoder-nq-train.json.gz"
dev_filename = "biencoder-nq-dev.json.gz"
s3_url_train = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz"
s3_url_dev = "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz"

fetch_archive_from_http(s3_url_train, output_dir=doc_dir)
fetch_archive_from_http(s3_url_dev, output_dir=doc_dir)


# Download datasets?
# format


#%%


# RECOMMENDED for finetuning "facebook/dpr-ctx_encoder-single-nq-base"

document_store = InMemoryDocumentStore()
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="bert-base-uncased",
    passage_embedding_model="bert-base-uncased",
    max_seq_len_query=64,
    max_seq_len_passage=256,
    embed_title=True,
)


#%% 

# Train
# Saves


retriever.train(
    data_dir=doc_dir,
    train_filename=dev_filename,
    dev_filename=dev_filename,
    test_filename=dev_filename,
    save_dir="..saved_models/dpr",
    n_epochs=1,
    batch_size=4,
    embed_title=True,
    num_hard_negatives=1,
    num_positives=1,
    evaluate_every=1000,
    learning_rate=1e-5,
    grad_acc_steps=1
)

#%%

# Load?