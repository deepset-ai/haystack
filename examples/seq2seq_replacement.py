from haystack import Document
from haystack.nodes import PromptNode, PromptTemplate

p = PromptNode("vblagoje/bart_lfqa")

# Start by defining a question/query
query = "Why does water heated to room temperature feel colder than the air around it?"

# Given the question above, suppose the documents below were found in some document store
documents = [
    "when the skin is completely wet. The body continuously loses water by...",
    "at greater pressures. There is an ambiguity, however, as to the meaning of the terms 'heating' and 'cooling'...",
    "are not in a relation of thermal equilibrium, heat will flow from the hotter to the colder, by whatever pathway...",
    "air condition and moving along a line of constant enthalpy toward a state of higher humidity. A simple example ...",
    "Thermal contact conductance. In physics, thermal contact conductance is the study of heat conduction between solid ...",
]


# Manually concatenate the question and support documents into BART input
# conditioned_doc = "<P> " + " <P> ".join([d for d in documents])
# query_and_docs = "question: {} context: {}".format(query, conditioned_doc)

# Or use the PromptTemplate as shown here
pt = PromptTemplate("lfqa", "question: {query} context: {join(documents, delimiter='<P>')}")

res = p.prompt(prompt_template=pt, query=query, documents=[Document(d) for d in documents])

print(res)
