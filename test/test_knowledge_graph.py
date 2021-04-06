from pathlib import Path

from haystack.graph_retriever import Text2SparqlRetriever
from haystack.knowledge_graph import GraphDBKnowledgeGraph


def test_graph_retrieval(retriever_with_docs, document_store_with_docs):
    # doc_dir = "../data/tutorial10_knowledge_graph/"
    # s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/tutorial10_knowledge_graph.zip"
    # fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # doc_dir = "../saved_models/tutorial10_knowledge_graph/"
    # s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-models/research/lcquad_wikidata.zip"
    # fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    kg = GraphDBKnowledgeGraph(index="lcquad_full_wikidata")
    kg.create_index(config_path=Path("../data/tutorial10_knowledge_graph/repo-config.ttl"))
    kg.import_from_ttl_file(index="lcquad_full_wikidata",
                            path=Path("../data/tutorial10_knowledge_graph/tutorial10_knowledge_graph.ttl"))
    triple = {'p': {'type': 'uri', 'value': 'http://www.wikidata.org/prop/direct/P31'}, 's': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q3486420'}, 'o': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q1002697'}}
    triples = kg.get_all_triples()
    assert len(triples) > 0
    assert triple in triples

    # Define prefixes for names of resources so that we can use shorter resource names in queries
    prefixes = """PREFIX bd: <http://www.bigdata.com/rdf#>
        PREFIX cc: <http://creativecommons.org/ns#>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX geo: <http://www.opengis.net/ont/geosparql#>
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX p: <http://www.wikidata.org/prop/>
        PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
        PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
        PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
        PREFIX pr: <http://www.wikidata.org/prop/reference/>
        PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
        PREFIX prov: <http://www.w3.org/ns/prov#>
        PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
        PREFIX ps: <http://www.wikidata.org/prop/statement/>
        PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
        PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX schema: <http://schema.org/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdata: <http://www.wikidata.org/wiki/Special:EntityData/>
        PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
        PREFIX wdref: <http://www.wikidata.org/reference/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/>
        PREFIX wdv: <http://www.wikidata.org/value/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        """
    kg.prefixes = prefixes

    kgqa_retriever = Text2SparqlRetriever(knowledge_graph=kg,
                                          model_name_or_path="../saved_models/tutorial10_knowledge_graph/lcquad_full_wikidata")

    question_text = "What periodical literature does Delta Air Lines use as a mouthpiece?"
    result = kgqa_retriever.retrieve(question_text=question_text)
    assert result[0] == {'answer': '', 'meta': {'model': 'Text2SparqlRetriever', 'sparql_query': ''}}

    result = kgqa_retriever._query_kg(query="select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . }")
    assert result[0][0] == "http://www.wikidata.org/entity/Q3486420"

    result = kgqa_retriever._query_kg(
        query="select distinct ?obj where { <http://www.wikidata.org/entity/Q188920> <http://www.wikidata.org/prop/direct/P2813> ?obj . }")
    assert result[0][0] == "http://www.wikidata.org/entity/Q3486420"
