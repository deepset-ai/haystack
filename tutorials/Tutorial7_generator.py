from typing import List

import numpy
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, RagSequenceForGeneration
import torch

from haystack import Document
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever


# Copy pasted from RagRetriever
def postprocess_docs(tokenizer, docs, input_strings, prefix, n_docs, return_tensors=None):
    r"""
    Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

    Args:
        doc_scores (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`):
            Retrieval scores of respective docs - passed for logging.
        docs  (:obj:`dict`):
            Retrieved documents.
        input_strings (:obj:`str`):
            Input strings decoded by ``preprocess_query``.
        prefix (:obj:`str`):
            Prefix added at the beginning of each input, typically used with T5-based models.

    Return:
        :obj:`tuple(tensors)`:
            a tuple consisting of two elements: contextualized ``input_ids`` and a compatible ``attention_mask``.
    """

    def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
        # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
        # TODO(piktus): better handling of truncation
        if doc_title.startswith('"'):
            doc_title = doc_title[1:]
        if doc_title.endswith('"'):
            doc_title = doc_title[:-1]
        if prefix is None:
            prefix = ""
        out = (prefix + doc_title + " / " + doc_text + " // " + input_string).replace(
            "  ", " "
        )
        return out

    rag_input_strings = [
        cat_input_and_doc(
            docs[i]["title"][j],
            docs[i]["text"][j],
            input_strings[i],
            prefix,
        )
        for i in range(len(docs))
        for j in range(n_docs)
    ]

    contextualized_inputs = tokenizer.batch_encode_plus(
        rag_input_strings,
        max_length=300,
        return_tensors=return_tensors,
        padding="max_length",
        truncation=True,
    )

    return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]


# Copied from https://github.com/huggingface/transformers/pull/7763/files
def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        for passage in split_text(text):
            titles.append(title)
            texts.append(passage)
    return {"title": titles, "text": texts}


# haystack part

documents = [
    Document(
        text="""Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother's spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from""",
        meta={"name": "0"}
    ),
    Document(
        text="""Democratic Republic of the Congo to the south. Angola's capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola's climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with""",
    ),
    Document(
        text="""Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are""",
        meta={"name": "1"}
    ),
    Document(
        text="""The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio""",
        meta={"name": "2"}
    ),
    Document(
        text="""The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin""",
        meta={}
    )
]

document_store = FAISSDocumentStore(faiss_index_factory_str="HNSW")
document_store.delete_all_documents()
document_store.write_documents(documents)

retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  use_gpu=False, embed_title=True,
                                  remove_sep_tok_from_untitled_passages=True)

document_store.update_embeddings(retriever=retriever)

docs_with_emb = document_store.get_all_documents()

question = "Who is the brother of Moses?"
retriever_results = retriever.retrieve(query=question, top_k=5)

stored_emb = []
retriever_texts = []
rag_format_doc = {"text": [], "title": []}
for retriever_result in retriever_results:
    retriever_texts.append(retriever_result.text)
    rag_format_doc['text'].append(retriever_result.text)
    rag_format_doc['title'].append("")
    stored_emb.append(document_store.faiss_index.reconstruct(int(retriever_result.meta["vector_id"])))

rag_format_docs = [rag_format_doc]

# Without RagRetriever
# Currently not working
def generator_test_1():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

    input_dict = tokenizer.prepare_seq2seq_batch(src_texts=[question], return_tensors="pt")
    print("input_dict: ", input_dict)

    doc_dict = tokenizer.prepare_seq2seq_batch(src_texts=[question],
                                               tgt_texts=retriever_texts, return_tensors="pt")

    print("doc_dict: ", doc_dict)

    question_hidden_states = model.question_encoder(input_dict["input_ids"])[0]
    embedding_in_tensor = torch.from_numpy(numpy.array(stored_emb)).to(question_hidden_states).float()

    doc_scores = torch.bmm(question_hidden_states.unsqueeze(1),
                           embedding_in_tensor.unsqueeze(0).transpose(1, 2)).squeeze(1)

    context_input_ids, context_attention_mask = postprocess_docs(tokenizer=retriever.query_tokenizer,
                docs=rag_format_docs, input_strings=[question], prefix=None, n_docs=len(retriever_texts), return_tensors="pt"
            )
    outputs = model.generate(
        input_ids=input_dict["input_ids"],
        context_input_ids=context_input_ids, # doc_dict["input_ids"],
        context_attention_mask=context_attention_mask, # doc_dict["attention_mask"],
        doc_scores=doc_scores
    )

    generated_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("====>1): ", generated_string)

    # outputs = model.generate(
    #     input_ids=input_dict["input_ids"],
    #     context_input_ids=doc_dict["input_ids"],
    #     context_attention_mask=doc_dict["attention_mask"],
    #     doc_scores=doc_scores
    # )
    # generated_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print("====>2): ", generated_string)


generator_test_1()
