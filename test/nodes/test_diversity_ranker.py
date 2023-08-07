from typing import List

import pytest

from haystack import Document
from haystack.nodes.ranker.diversity import DiversityRanker


# Tests that predict method returns a list of Document objects
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_returns_list_of_documents(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    query = "test query"
    documents = [Document(content="doc1"), Document(content="doc2")]
    result = ranker.predict(query=query, documents=documents)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)


#  Tests that predict method returns the correct number of documents
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_returns_correct_number_of_documents(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    query = "test query"
    documents = [Document(content="doc1"), Document(content="doc2")]
    result = ranker.predict(query=query, documents=documents, top_k=1)
    assert len(result) == 1


#  Tests that predict method returns documents in the correct order
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_returns_documents_in_correct_order(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    query = "city"
    documents = [
        Document("France"),
        Document("Germany"),
        Document("Eiffel Tower"),
        Document("Berlin"),
        Document("bananas"),
        Document("Silicon Valley"),
        Document("Brandenburg Gate"),
    ]
    result = ranker.predict(query=query, documents=documents)
    expected_order = "Berlin, bananas, Eiffel Tower, Silicon Valley, France, Brandenburg Gate, Germany"
    assert ", ".join([doc.content for doc in result]) == expected_order


#  Tests that predict_batch method returns a list of lists of Document objects
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_batch_returns_list_of_lists_of_documents(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    queries = ["test query 1", "test query 2"]
    documents = [
        [Document(content="doc1"), Document(content="doc2")],
        [Document(content="doc3"), Document(content="doc4")],
    ]
    result: List[List[Document]] = ranker.predict_batch(queries=queries, documents=documents)
    assert isinstance(result, list)
    assert all(isinstance(docs, list) for docs in result)
    assert all(isinstance(doc, Document) for docs in result for doc in docs)


#  Tests that predict_batch method returns the correct number of documents
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_batch_returns_correct_number_of_documents(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    queries = ["test query 1", "test query 2"]
    documents = [
        [Document(content="doc1"), Document(content="doc2")],
        [Document(content="doc3"), Document(content="doc4")],
    ]
    result: List[List[Document]] = ranker.predict_batch(queries=queries, documents=documents, top_k=1)
    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 1


#  Tests that predict_batch method returns documents in the correct order
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_batch_returns_documents_in_correct_order(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    queries = ["Berlin", "Paris"]
    documents = [
        [Document(content="Germany"), Document(content="Munich"), Document(content="agriculture")],
        [Document(content="France"), Document(content="Space exploration"), Document(content="Eiffel Tower")],
    ]
    result: List[List[Document]] = ranker.predict_batch(queries=queries, documents=documents)
    assert len(result) == 2

    # check the correct most diverse order are in batches
    expected_order_0 = "Germany, agriculture, Munich"
    expected_order_1 = "France, Space exploration, Eiffel Tower"
    assert ", ".join([doc.content for doc in result[0]]) == expected_order_0
    assert ", ".join([doc.content for doc in result[1]]) == expected_order_1


# Tests that predict method returns the correct number of documents for a single document
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_single_document_corner_case(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    query = "test"
    documents = [Document(content="doc1")]
    result = ranker.predict(query=query, documents=documents)
    assert len(result) == 1


#  Tests that predict method raises ValueError if query is empty
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_raises_value_error_if_query_is_empty(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    query = ""
    documents = [Document(content="doc1"), Document(content="doc2")]
    with pytest.raises(ValueError):
        ranker.predict(query=query, documents=documents)


#  Tests that predict method raises ValueError if documents is empty
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_raises_value_error_if_documents_is_empty(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    query = "test query"
    documents = []
    with pytest.raises(ValueError):
        ranker.predict(query=query, documents=documents)


#  Tests that predict_batch method raises ValueError if queries is empty
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_batch_raises_value_error_if_queries_is_empty(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    queries = []
    documents = [
        [Document(content="doc1"), Document(content="doc2")],
        [Document(content="doc3"), Document(content="doc4")],
    ]
    with pytest.raises(ValueError):
        ranker.predict_batch(queries=queries, documents=documents)


#  Tests that predict_batch method raises ValueError if documents is empty
@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_batch_raises_value_error_if_documents_is_empty(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    queries = ["test query 1", "test query 2"]
    documents = []
    with pytest.raises(ValueError):
        ranker.predict_batch(queries=queries, documents=documents)


@pytest.mark.integration
@pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
def test_predict_real_world_use_case(similarity: str):
    ranker = DiversityRanker(similarity=similarity)  # type: ignore
    query = "What are the reasons for long-standing animosities between Russia and Poland?"

    doc1 = Document(
        "One of the earliest known events in Russian-Polish history dates back to 981, when the Grand Prince of Kiev , "
        "Vladimir Svyatoslavich , seized the Cherven Cities from the Duchy of Poland . The relationship between two by "
        "that time was mostly close and cordial, as there had been no serious wars between both. In 966, Poland "
        "accepted Christianity from Rome while Kievan Rus' —the ancestor of Russia, Ukraine and Belarus—was "
        "Christianized by Constantinople. In 1054, the internal Christian divide formally split the Church into "
        "the Catholic and Orthodox branches separating the Poles from the Eastern Slavs."
    )

    doc2 = Document(
        "Since the fall of the Soviet Union , with Lithuania , Ukraine and Belarus regaining independence, the "
        "Polish–Russian border has mostly been replaced by borders with the respective countries, but there still "
        "is a 210 km long border between Poland and the Kaliningrad Oblast"
    )

    doc3 = Document(
        "As part of Poland's plans to become fully energy independent from Russia within the next years, Piotr "
        "Wozniak, president of state-controlled oil and gas company PGNiG , stated in February 2019: 'The strategy of "
        "the company is just to forget about Eastern suppliers and especially about Gazprom .'[53] In 2020, the "
        "Stockholm Arbitral Tribunal ruled that PGNiG's long-term contract gas price with Gazprom linked to oil prices "
        "should be changed to approximate the Western European gas market price, backdated to 1 November 2014 when "
        "PGNiG requested a price review under the contract. Gazprom had to refund about $1.5 billion to PGNiG."
    )

    doc4 = Document(
        "Both Poland and Russia had accused each other for their historical revisionism . Russia has repeatedly "
        "accused Poland for not honoring Soviet Red Army soldiers fallen in World War II for Poland, notably in "
        "2017, in which Poland was thought on 'attempting to impose its own version of history' after Moscow was "
        "not allowed to join an international effort to renovate a World War II museum at Sobibór , site of a "
        "notorious Sobibor extermination camp."
    )

    doc5 = Document(
        "President of Russia Vladimir Putin and Prime Minister of Poland Leszek Miller in 2002 Modern Polish–Russian "
        "relations begin with the fall of communism – 1989 in Poland ( Solidarity and the Polish Round Table "
        "Agreement ) and 1991 in Russia ( dissolution of the Soviet Union ). With a new democratic government after "
        "the 1989 elections , Poland regained full sovereignty, [2] and what was the Soviet Union, became 15 newly "
        "independent states , including the Russian Federation . Relations between modern Poland and Russia suffer "
        "from constant ups and downs."
    )

    doc6 = Document(
        "Soviet influence in Poland finally ended with the Round Table Agreement of 1989 guaranteeing free elections "
        "in Poland, the Revolutions of 1989 against Soviet-sponsored Communist governments in the Eastern Bloc , and "
        "finally the formal dissolution of the Warsaw Pact."
    )

    doc7 = Document(
        "Dmitry Medvedev and then Polish Prime Minister Donald Tusk , 6 December 2010 BBC News reported that one of "
        "the main effects of the 2010 Polish Air Force Tu-154 crash would be the impact it has on Russian-Polish "
        "relations. [38] It was thought if the inquiry into the crash were not transparent, it would increase "
        "suspicions toward Russia in Poland."
    )

    doc8 = Document(
        "Soviet control over the Polish People's Republic lessened after Stalin's death and Gomułka's Thaw , and "
        "ceased completely after the fall of the communist government in Poland in late 1989, although the "
        "Soviet-Russian Northern Group of Forces did not leave Polish soil until 1993. The continuing Soviet military "
        "presence allowed the Soviet Union to heavily influence Polish politics."
    )

    documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]
    result = ranker.predict(query=query, documents=documents)
    expected_order = [doc5, doc7, doc3, doc1, doc4, doc2, doc6, doc8]
    assert result == expected_order


@pytest.mark.integration
def test_diversity_ranker_with_top_k():
    # Tests that predict method returns the correct order of documents
    ranker = DiversityRanker(similarity="cosine", top_k=1)
    query = "test"
    documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
    result = ranker.predict(query=query, documents=documents)
    assert len(result) == 1


@pytest.mark.integration
def test_diversity_ranker_with_top_k_edge():
    # Tests that predict method returns the correct order of documents for edge cases
    ranker = DiversityRanker(similarity="cosine", top_k=5)
    query = "test"
    documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
    result = ranker.predict(query=query, documents=documents)
    assert len(result) == 3

    # negative top_k should return empty list
    ranker = DiversityRanker(similarity="cosine", top_k=-5)
    query = "test"
    documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
    result = ranker.predict(query=query, documents=documents)
    assert len(result) == 0

    # we know None is ignored in slice notation, but let's make sure it works
    ranker = DiversityRanker(similarity="cosine", top_k=None)
    query = "test"
    documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
    result = ranker.predict(query=query, documents=documents)
    assert len(result) == 3
