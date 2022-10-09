all: elastic_search_up give_es_some_time populate_indexes rest_api_up give_api_some_time ui_up

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

ui_up:
	#note: ui runs at http://localhost:8501/
	docker-compose up -d ui
