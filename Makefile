all: elastic_search_up index_documents give_indexing_some_time rest_api_up run_query

elastic_search_up:
	docker-compose up -d elasticsearch

index_documents:
	python document_indexing/populate_document_store.py

give_indexing_some_time:
	sleep 20

rest_api_up:
	docker-compose up -d haystack-api

run_query:
	 curl -H 'Content-Type: application/json' -H 'Accept: application/json' -d '{"query": "What is the objective of the game?"}' http://127.0.0.1:8000/query
