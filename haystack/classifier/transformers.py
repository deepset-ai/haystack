from typing import List, Optional

from transformers import pipeline

from haystack import Document
from haystack.classifier.base import BaseClassifier


class TransformersClassifier(BaseClassifier):
    """
    Transformer based model for document classification using the HuggingFace's transformers framework
    (https://github.com/huggingface/transformers).
    While the underlying model can vary (BERT, Roberta, DistilBERT ...), the interface remains the same.
    This node classifies documents and adds the output from the classification step to the document's meta data.
    The meta field of the document is a dictionary with the following format:
    'meta': {'name': '450_Baelor.txt', 'classification': {'label': 'neutral', 'probability' = 0.9997646, ...} }

    With this classifier, you can directly get predictions via predict()

    Usage example:
    ...
    retriever = ElasticsearchRetriever(document_store=document_store)
    classifier = TransformersClassifier(model_name_or_path="bhadresh-savani/distilbert-base-uncased-emotion")
    p = Pipeline()
    p.add_node(component=retriever, name="Retriever", inputs=["Query"])
    p.add_node(component=classifier, name="Classifier", inputs=["Retriever"])
    res = p.run(
        query="Who is the father of Arya Stark?",
        params={"Retriever": {"top_k": 10}, "Classifier": {"top_k": 5}}
    )
    print(res["documents"][0].to_dict()["meta"]["classification"]["label"])
    # Note that print_documents() does not output the content of the classification field in the meta data
    # document_dicts = [doc.to_dict() for doc in res["documents"]]
    # res["documents"] = document_dicts
    # print_documents(res, max_text_len=100)
    """

    def __init__(
        self,
        model_name_or_path: str = "bhadresh-savani/distilbert-base-uncased-emotion",
        model_version: Optional[str] = None,
        tokenizer: Optional[str] = None,
        use_gpu: int = 0,
        max_seq_len: int = 256,
        doc_stride: int = 128
    ):
        """
        Load a text classification model from Transformers.
        Available models include:

        - ``'bhadresh-savani/distilbert-base-uncased-emotion'``
        - ``'Hate-speech-CNERG/dehatebert-mono-english'``

        See https://huggingface.co/models for full list of available text classification models or filter for them:
        https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads

        :param model_name_or_path: Directory of a saved model or the name of a public model e.g. 'bhadresh-savani/distilbert-base-uncased-emotion'.
        See https://huggingface.co/models for full list of available models.
        :param model_version: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :param tokenizer: Name of the tokenizer (usually the same as model)
        :param use_gpu: If < 0, then use cpu. If >= 0, this is the ordinal of the gpu to use
        :param max_seq_len: max sequence length of one input text for the model
        :param doc_stride: length of striding window for splitting long texts (used if len(text) > max_seq_len)

        """

        # save init parameters to enable export of component config as YAML
        self.set_config(
            model_name_or_path=model_name_or_path, model_version=model_version, tokenizer=tokenizer,
            use_gpu=use_gpu, doc_stride=doc_stride, max_seq_len=max_seq_len,
        )

        self.model = pipeline('text-classification', model=model_name_or_path, tokenizer=tokenizer, device=use_gpu, revision=model_version)
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride


    def predict(self, documents: List[Document]) -> List[Document]:
        """
        Returns documents containing classification result in meta field

        :param documents: List of Document in which to search for the answer
        :return: List of Document enriched with meta information

        """
        texts = [doc.text for doc in documents]
        predictions = self.model(texts,
                                max_seq_len=self.max_seq_len,
                                doc_stride=self.doc_stride)

        classified_docs: List[Document] = []

        for prediction, doc in zip(predictions, documents):
            cur_doc = doc
            cur_doc.meta["classification"] = prediction
            classified_docs.append(cur_doc)

        return classified_docs

    def predict_batch(self, query_doc_list: List[dict],  batch_size: Optional[int] = None):

        raise NotImplementedError("Batch prediction not yet available in TransformersClassifier.")
