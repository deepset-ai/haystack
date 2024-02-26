import pytest

from haystack import ComponentError, Document
from haystack.components.rankers import DiversityRanker
from haystack.utils import ComponentDevice
from haystack.utils.auth import Secret


class TestDiversityRanker:
    def test_init(self):
        component = DiversityRanker()
        assert component.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert component.top_k == 10
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.similarity == "dot_product"
        assert component.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert component.query_prefix == ""
        assert component.document_prefix == ""
        assert component.query_suffix == ""
        assert component.document_suffix == ""
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"

    def test_init_with_custom_init_parameters(self):
        component = DiversityRanker(
            model="sentence-transformers/msmarco-distilbert-base-v4",
            top_k=5,
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            similarity="cosine",
            query_prefix="query:",
            document_prefix="document:",
            query_suffix="query suffix",
            document_suffix="document suffix",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="--",
        )
        assert component.model_name_or_path == "sentence-transformers/msmarco-distilbert-base-v4"
        assert component.top_k == 5
        assert component.device == ComponentDevice.from_str("cuda:0")
        assert component.similarity == "cosine"
        assert component.token == Secret.from_token("fake-api-token")
        assert component.query_prefix == "query:"
        assert component.document_prefix == "document:"
        assert component.query_suffix == "query suffix"
        assert component.document_suffix == "document suffix"
        assert component.meta_fields_to_embed == ["meta_field"]
        assert component.embedding_separator == "--"

    def test_to_and_from_dict(self):
        component = DiversityRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.diversity.DiversityRanker",
            "init_parameters": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k": 10,
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "similarity": "dot_product",
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "query_prefix": "",
                "document_prefix": "",
                "query_suffix": "",
                "document_suffix": "",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

        ranker = DiversityRanker.from_dict(data)

        assert ranker.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert ranker.top_k == 10
        assert ranker.device == ComponentDevice.resolve_device(None)
        assert ranker.similarity == "dot_product"
        assert ranker.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert ranker.query_prefix == ""
        assert ranker.document_prefix == ""
        assert ranker.query_suffix == ""
        assert ranker.document_suffix == ""
        assert ranker.meta_fields_to_embed == []
        assert ranker.embedding_separator == "\n"

    def test_to_and_from_dict_with_custom_init_parameters(self):
        component = DiversityRanker(
            model="sentence-transformers/msmarco-distilbert-base-v4",
            top_k=5,
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            similarity="cosine",
            query_prefix="query:",
            document_prefix="document:",
            query_suffix="query suffix",
            document_suffix="document suffix",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="--",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.rankers.diversity.DiversityRanker",
            "init_parameters": {
                "model": "sentence-transformers/msmarco-distilbert-base-v4",
                "top_k": 5,
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "similarity": "cosine",
                "query_prefix": "query:",
                "document_prefix": "document:",
                "query_suffix": "query suffix",
                "document_suffix": "document suffix",
                "meta_fields_to_embed": ["meta_field"],
                "embedding_separator": "--",
            },
        }

        ranker = DiversityRanker.from_dict(data)

        assert ranker.model_name_or_path == "sentence-transformers/msmarco-distilbert-base-v4"
        assert ranker.top_k == 5
        assert ranker.device == ComponentDevice.from_str("cuda:0")
        assert ranker.similarity == "cosine"
        assert ranker.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert ranker.query_prefix == "query:"
        assert ranker.document_prefix == "document:"
        assert ranker.query_suffix == "query suffix"
        assert ranker.document_suffix == "document suffix"
        assert ranker.meta_fields_to_embed == ["meta_field"]
        assert ranker.embedding_separator == "--"

    def test_run_incorrect_similarity(self):
        """
        Tests that run method raises ValueError if similarity is incorrect
        """
        with pytest.raises(ValueError):
            DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity="incorrect")

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_without_warm_up(self, similarity):
        """
        Tests that run method raises ComponentError if model is not warmed up
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", top_k=1, similarity=similarity)
        documents = [Document(content="doc1"), Document(content="doc2")]

        with pytest.raises(ComponentError):
            ranker.run(query="test query", documents=documents)

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_empty_query(self, similarity):
        """
        Test that ranker can be run with an empty query.
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", top_k=3, similarity=similarity)
        ranker.warm_up()
        documents = [Document(content="doc1"), Document(content="doc2")]

        result = ranker.run(query="", documents=documents)
        ranked_docs = result["documents"]

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == 2
        assert all(isinstance(doc, Document) for doc in ranked_docs)

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_top_k(self, similarity):
        """
        Test that run method returns the correct number of documents for different top_k values passed at
        initialization and runtime.
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=3)
        ranker.warm_up()
        query = "test query"
        documents = [
            Document(content="doc1"),
            Document(content="doc2"),
            Document(content="doc3"),
            Document(content="doc4"),
        ]

        result = ranker.run(query=query, documents=documents)
        ranked_docs = result["documents"]

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == 3
        assert all(isinstance(doc, Document) for doc in ranked_docs)

        # Passing a different top_k at runtime
        result = ranker.run(query=query, documents=documents, top_k=2)
        ranked_docs = result["documents"]

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == 2
        assert all(isinstance(doc, Document) for doc in ranked_docs)

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_diversity_ranker_negative_top_k(self, similarity):
        """
        Tests that run method raises an error for negative top-k.
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=10)
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]

        # Setting top_k at runtime
        with pytest.raises(ValueError):
            ranker.run(query=query, documents=documents, top_k=-5)

        # Setting top_k at init
        with pytest.raises(ValueError):
            DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=-5)

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_diversity_ranker_top_k_is_none(self, similarity):
        """
        Tests that run method returns the correct order of documents for top-k set to None.
        """
        # Setting top_k to None at init should raise error
        with pytest.raises(ValueError):
            DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=None)

        # Setting top_k to None is ignored during runtime, it should use top_k set at init.
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=2)
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        result = ranker.run(query=query, documents=documents, top_k=None)

        assert len(result["documents"]) == 2

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_with_less_documents_than_top_k(self, similarity):
        """
        Tests that run method returns the correct number of documents for top_k values greater than number of documents.
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=5)
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        result = ranker.run(query=query, documents=documents)

        assert len(result["documents"]) == 3

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_single_document_corner_case(self, similarity):
        """
        Tests that run method returns the correct number of documents for a single document
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity)
        ranker.warm_up()
        query = "test"
        documents = [Document(content="doc1")]
        result = ranker.run(query=query, documents=documents)

        assert len(result["documents"]) == 1

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_no_documents_provided(self, similarity):
        """
        Test that run method returns an empty list if no documents are supplied.
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity)
        ranker.warm_up()
        query = "test query"
        documents = []
        results = ranker.run(query=query, documents=documents)

        assert len(results["documents"]) == 0

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run(self, similarity):
        """
        Tests that run method returns documents in the correct order
        """
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity)
        ranker.warm_up()
        query = "city"
        documents = [
            Document(content="France"),
            Document(content="Germany"),
            Document(content="Eiffel Tower"),
            Document(content="Berlin"),
            Document(content="Bananas"),
            Document(content="Silicon Valley"),
            Document(content="Brandenburg Gate"),
        ]
        result = ranker.run(query=query, documents=documents)
        ranked_docs = result["documents"]
        ranked_order = ", ".join([doc.content for doc in ranked_docs])
        expected_order = "Berlin, Bananas, Eiffel Tower, Silicon Valley, France, Brandenburg Gate, Germany"

        assert ranked_order == expected_order

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_real_world_use_case(self, similarity):
        ranker = DiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity)
        ranker.warm_up()
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
            "Polish Russian border has mostly been replaced by borders with the respective countries, but there still "
            "is a 210 km long border between Poland and the Kaliningrad Oblast"
        )
        doc3 = Document(
            "As part of Poland's plans to become fully energy independent from Russia within the next years, Piotr "
            "Wozniak, president of state-controlled oil and gas company PGNiG , stated in February 2019: 'The strategy of "
            "the company is just to forget about Eastern suppliers and especially about Gazprom .'[53] In 2020, the "
            "Stockholm Arbitrary Tribunal ruled that PGNiG's long-term contract gas price with Gazprom linked to oil prices "
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
            "President of Russia Vladimir Putin and Prime Minister of Poland Leszek Miller in 2002 Modern Polish Russian "
            "relations begin with the fall of communism in1989 in Poland ( Solidarity and the Polish Round Table "
            "Agreement ) and 1991 in Russia ( dissolution of the Soviet Union ). With a new democratic government after "
            "the 1989 elections , Poland regained full sovereignty, [2] and what was the Soviet Union, became 15 newly "
            "independent states , including the Russian Federation . Relations between modern Poland and Russia suffer "
            "from constant ups and downs."
        )
        doc6 = Document(
            "Soviet influence in Poland finally ended with the Round Table Agreement of 1989 guaranteeing free elections "
            "in Poland, the Revolutions of 1989 against Soviet-sponsored Communist governments in the Eastern Block , and "
            "finally the formal dissolution of the Warsaw Pact."
        )
        doc7 = Document(
            "Dmitry Medvedev and then Polish Prime Minister Donald Tusk , 6 December 2010 BBC News reported that one of "
            "the main effects of the 2010 Polish Air Force Tu-154 crash would be the impact it has on Russian-Polish "
            "relations. [38] It was thought if the inquiry into the crash were not transparent, it would increase "
            "suspicions toward Russia in Poland."
        )
        doc8 = Document(
            "Soviet control over the Polish People's Republic lessened after Stalin's death and Gomulka's Thaw , and "
            "ceased completely after the fall of the communist government in Poland in late 1989, although the "
            "Soviet-Russian Northern Group of Forces did not leave Polish soil until 1993. The continuing Soviet military "
            "presence allowed the Soviet Union to heavily influence Polish politics."
        )

        documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]
        result = ranker.run(query=query, documents=documents)
        expected_order = [doc5, doc7, doc3, doc1, doc4, doc2, doc6, doc8]
        expected_content = " ".join([doc.content or "" for doc in expected_order])
        result_content = " ".join([doc.content or "" for doc in result["documents"]])

        # Check the order of ranked documents by comparing the content of the ranked documents
        assert result_content == expected_content
