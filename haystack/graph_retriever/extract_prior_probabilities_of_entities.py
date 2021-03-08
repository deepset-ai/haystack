import re
from collections import Counter
import io
from operator import itemgetter
import json

entity_alias_count = Counter()
alias_to_entity = {}
alias_to_entity_and_prob = {}
entity_to_frequency = Counter()
with io.open("harrypotter_pages_current.xml",'r',encoding='utf-8') as wiki_txt:
	text = wiki_txt.read()
	regex = re.compile(r'\[\[([^\|\]\[:]+?)\|([^\|\]\[:]+?)\]\]')
	all_matches = re.findall(regex,text)
	for alias_match in all_matches:
		entity_to_frequency[alias_match[0]] += 1
		entity_alias_count[(alias_match[0], alias_match[1])] += 1
		if alias_match[1] not in alias_to_entity:
			alias_to_entity[alias_match[1]] = set([alias_match[0]])
		else: 
			alias_to_entity[alias_match[1]].add(alias_match[0])


for alias in alias_to_entity:
	total_occurrences = 0
	for entity in alias_to_entity[alias]:
		total_occurrences += entity_alias_count[(entity,alias)]
	for entity in alias_to_entity[alias]:
		if entity_alias_count[(entity,alias)] > 5:
			if not alias in alias_to_entity_and_prob:
				alias_to_entity_and_prob[alias] = []
			alias_to_entity_and_prob[alias].append((entity,entity_alias_count[(entity,alias)]/total_occurrences))
		else:
			print(f"alias {alias} not linked to entity {entity}")


#alias_to_entity_and_prob
# for every alias: choose most likely entity
# collect for each entity the chosen aliases
# select only the most frequent alias
entity_to_aliases = {}
for alias in alias_to_entity_and_prob.keys():
	most_likely_entity = max(alias_to_entity_and_prob[alias], key=itemgetter(1))[0]
	if entity_alias_count[(most_likely_entity, alias)] > 1:
		if most_likely_entity in entity_to_aliases:
			t_alias = entity_to_aliases[most_likely_entity]
			if entity_alias_count[(most_likely_entity,t_alias)] < entity_alias_count[(most_likely_entity,alias)]:
				entity_to_aliases[most_likely_entity] = alias
		else:
			entity_to_aliases[most_likely_entity] = alias

print(len(entity_to_aliases))
json.dump(entity_to_aliases, open("entity_to_most_frequent_alias.json", "w"))
print(alias_to_entity_and_prob["Harry"])
json.dump(alias_to_entity_and_prob, open("alias_to_entity_and_prob.json", "w"))
print(entity_to_frequency["Harry"])
json.dump(entity_to_frequency, open("entity_to_frequency.json", "w"))
#print(entity_alias_count.most_common(100))



