import json
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import numpy as np


prefix_to_url = {"bd": "http://www.bigdata.com/rdf#",
"cc": "http://creativecommons.org/ns#",
"dct": "http://purl.org/dc/terms/",
"geo": "http://www.opengis.net/ont/geosparql#",
"ontolex": "http://www.w3.org/ns/lemon/ontolex#",
"owl": "http://www.w3.org/2002/07/owl#",
"p": "http://www.wikidata.org/prop/",
"pq": "http://www.wikidata.org/prop/qualifier/",
"pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
"pqv": "http://www.wikidata.org/prop/qualifier/value/",
"pr": "http://www.wikidata.org/prop/reference/",
"prn": "http://www.wikidata.org/prop/reference/value-normalized/",
"prov": "http://www.w3.org/ns/prov#",
"prv": "http://www.wikidata.org/prop/reference/value/",
"ps": "http://www.wikidata.org/prop/statement/",
"psn": "http://www.wikidata.org/prop/statement/value-normalized/",
"psv": "http://www.wikidata.org/prop/statement/value/",
"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
"rdfs": "http://www.w3.org/2000/01/rdf-schema#",
"schema": "http://schema.org/",
"skos": "http://www.w3.org/2004/02/skos/core#",
"wd": "http://www.wikidata.org/entity/",
"wdata": "http://www.wikidata.org/wiki/Special:EntityData/",
"wdno": "http://www.wikidata.org/prop/novalue/",
"wdref": "http://www.wikidata.org/reference/",
"wds": "http://www.wikidata.org/entity/statement/",
"wdt": "http://www.wikidata.org/prop/direct/",
"wdtn": "http://www.wikidata.org/prop/direct-normalized/",
"wdv": "http://www.wikidata.org/value/",
"wikibase": "http://wikiba.se/ontology#",
"xsd": "http://www.w3.org/2001/XMLSchema#"}

url_to_prefix = {v: k for k, v in prefix_to_url.items()}


def translate(x):
    if not x.startswith("http://"):
        return x
    x = x.split("/")
    url = x[:-1]
    url = "/".join(url)
    if url+"/" in url_to_prefix:
        result = url_to_prefix[url+"/"]+":"+x[-1]
        #result = result.split("-")[0]
        return result
    else:
        return np.NaN

def extract_identifiers(filename):
    data = json.load(open(filename, "r"))
    identifiers = set()
    for question in data:
        question_text = question["sparql_wikidata"]
        question_text = question_text.replace("}", " } ").replace("{", " { ")
        for token in question_text.split():
            if ":" in token:
                identifiers.add(token)
    return identifiers


identifiers = extract_identifiers(filename="../../data/train.json")
identifiers.update(extract_identifiers(filename="../../data/test.json"))
identifiers = list(identifiers)
print(f"Extracted {len(identifiers)} distinct identifiers") # Extracted 27417 distinct identifiers
json.dump(identifiers, open("../../data/train_and_test_identifiers.json", "w"))


sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
triples = pd.DataFrame()

def load_triples(sparql, identifiers):
    triples = pd.DataFrame()
    for count, identifier in enumerate(identifiers):
        if count % 100 == 0:
            print(f"Processed {count} out of {len(identifiers)}")
        sparql.setQuery(f"""
        SELECT ?s ?p
        WHERE
        {{
            ?s ?p {identifier} .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        if len(results_df) > 0:
            results_df['o.value'] = identifier
            results_df['s.value'] = results_df['s.value'].apply(lambda x: translate(x))
            results_df['p.value'] = results_df['p.value'].apply(lambda x: translate(x))
            triples = triples.append(results_df[['s.value', 'p.value', 'o.value']])
            triples = triples.drop_duplicates()
            triples = triples.dropna()
    for identifier in identifiers:
        sparql.setQuery(f"""
        SELECT ?p ?o
        WHERE
        {{
            {identifier} ?p ?o .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        if len(results_df) > 0:
            results_df['s.value'] = identifier
            results_df['o.value'] = results_df['o.value'].apply(lambda x: translate(x))
            results_df['p.value'] = results_df['p.value'].apply(lambda x: translate(x))
            triples = triples.append(results_df[['s.value', 'p.value', 'o.value']])
            triples = triples.drop_duplicates()
            triples = triples.dropna()
    for identifier in identifiers:
        sparql.setQuery(f"""
        SELECT ?s ?o
        WHERE
        {{
            ?s {identifier} ?o .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        results_df = pd.io.json.json_normalize(results['results']['bindings'])
        if len(results_df) > 0:
            results_df['p.value'] = identifier
            results_df['s.value'] = results_df['s.value'].apply(lambda x: translate(x))
            results_df['o.value'] = results_df['o.value'].apply(lambda x: translate(x))
            triples = triples.append(results_df[['s.value', 'p.value', 'o.value']])
            triples = triples.drop_duplicates()
            triples = triples.dropna()
    return triples


triples = load_triples(sparql, identifiers[:5])
triples = triples.drop_duplicates()
triples = triples.dropna()
print(len(triples))
print(triples)
triples.to_csv("../../data/lcquad_triples.txt", index=False, sep=" ")
