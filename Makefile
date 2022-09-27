all: elastic_search_up give_es_some_time populate_indexes rest_api_up give_api_some_time run_query run_faq_query run_rulebook_query

elastic_search_up:
	docker-compose up -d elasticsearch

give_es_some_time:
	sleep 30

populate_indexes:
	python document_indexing/populate_document_store.py

rest_api_up:
	docker-compose up -d haystack-api

give_api_some_time:
	sleep 120

run_faq_query:
	curl -H 'Content-Type: application/json' -H 'Accept: application/json' -d '{"query": "What is the objective of the game?", "params": {"CustomClassifier": {"index": "faq"}, "FaqRetriever": {"top_k": 5, "index":"faq"}}}' http://127.0.0.1:8000/query

run_rulebook_query:
	curl -H 'Content-Type: application/json' -H 'Accept: application/json' -d '{"query": "What is the objective of the game?", "params": {"CustomClassifier": {"index": "rulebook"}, "ExtrReader": {"top_k": 1}, "ExtrRetriever": {"top_k": 5, "index":"rulebook"}}}' http://127.0.0.1:8000/query
