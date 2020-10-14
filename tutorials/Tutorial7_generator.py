from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, RagSequenceForGeneration
import torch

from haystack import Document
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever

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

docs_with_emb = document_store.get_all_documents(index="test_dpr")

res = retriever.retrieve(query="Which philosopher attacked Schopenhauer?", top_k=1)

# Generator part

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-base")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-base")

input_dict = tokenizer.prepare_seq2seq_batch("Which philosopher attacked Schopenhauer?", return_tensors="pt")
print("input_dict: ", input_dict)

doc_dict = tokenizer.prepare_seq2seq_batch(res[0].text, return_tensors="pt")

question_hidden_states = model.question_encoder(input_dict["input_ids"])[0]

#doc_scores = torch.bmm(question_hidden_states.unsqueeze(1),
#                       res[0].embedding).squeeze(1)

outputs = model(context_input_ids=doc_dict["input_ids"],
                context_attention_mask=doc_dict["attention_mask"],
                decoder_input_ids=input_dict["labels"],
                doc_scores=res[0].score
                )

generated = model.generate(input_ids=input_dict["input_ids"])
generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)

print(generated_string)
