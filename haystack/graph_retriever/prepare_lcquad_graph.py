import json
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import numpy as np

prefixes = """@prefix bd: <http://www.bigdata.com/rdf#> .
@prefix cc: <http://creativecommons.org/ns#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix geo: <http://www.opengis.net/ont/geosparql#> .
@prefix ontolex: <http://www.w3.org/ns/lemon/ontolex#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix p: <http://www.wikidata.org/prop/> .
@prefix pq: <http://www.wikidata.org/prop/qualifier/> .
@prefix pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/> .
@prefix pqv: <http://www.wikidata.org/prop/qualifier/value/> .
@prefix pr: <http://www.wikidata.org/prop/reference/> .
@prefix prn: <http://www.wikidata.org/prop/reference/value-normalized/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix prv: <http://www.wikidata.org/prop/reference/value/> .
@prefix ps: <http://www.wikidata.org/prop/statement/> .
@prefix psn: <http://www.wikidata.org/prop/statement/value-normalized/> .
@prefix psv: <http://www.wikidata.org/prop/statement/value/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix wd: <http://www.wikidata.org/entity/> .
@prefix wdata: <http://www.wikidata.org/wiki/Special:EntityData/> .
@prefix wdno: <http://www.wikidata.org/prop/novalue/> .
@prefix wdref: <http://www.wikidata.org/reference/> .
@prefix wds: <http://www.wikidata.org/entity/statement/> .
@prefix wdt: <http://www.wikidata.org/prop/direct/> .
@prefix wdtn: <http://www.wikidata.org/prop/direct-normalized/> .
@prefix wdv: <http://www.wikidata.org/value/> .
@prefix wikibase: <http://wikiba.se/ontology#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

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


identifiers = extract_identifiers(filename="../../data/test.json")
#identifiers.update(extract_identifiers(filename="../../data/train.json"))
identifiers = list(identifiers)
print(f"Extracted {len(identifiers)} distinct identifiers") # Extracted 27417 distinct identifiers
json.dump(identifiers, open("../../data/identifiers.json", "w"))


sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setReturnFormat(JSON)
triples = pd.DataFrame()

def load_describing_triples(sparql, identifiers):
    triples = pd.DataFrame()
    for count, identifier in enumerate(identifiers):
        if count % 10 == 0:
            print(f"Processed {count} out of {len(identifiers)}")
            triples = triples.drop_duplicates()
            triples = triples.dropna()
        try:
            sparql.setQuery(f"""DESCRIBE {identifier}""")
            results = sparql.query().convert()

            results_df = pd.io.json.json_normalize(results['results']['bindings'])
            if len(results_df) > 0:
                triples = triples.append(results_df[['subject.value', 'predicate.value', 'object.value']])
        except Exception:
            continue
    return triples

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


#triples = load_describing_triples(sparql, identifiers[:5])
triples = load_describing_triples(sparql, identifiers)
triples = triples.drop_duplicates()
triples = triples.dropna()
print(len(triples))

with open('../../data/tutorial10_knowledge_graph_triples.ttl', 'w') as the_file:
    the_file.write(prefixes)
    for index, row in triples.iterrows():
        the_file.write(row['subject.value']+" "+row['predicate.value']+" "+row['object.value']+" .\n")
#triples.to_csv("../../data/lcquad_triples.txt", index=False, sep=" ")
