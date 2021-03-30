import json
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

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

# todo
#   triples = pd.dataframe(...
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

for identifier in identifiers[:2]:
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

    print(results_df[['s.value', 'p.value']].head())
