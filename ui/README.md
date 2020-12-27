## Demo UI

This is a minimal UI that can spin up to test Haystack for your prototypes. It's based on streamlit. 


## Usage

### Option 1: Local
Execute in this folder:
```
streamlit run webapp.py
```

Requirements: This expects a running Haystack REST API at `http://localhost:8000`

### Option 2: Container

Just run
```
docker-compose up -d
``` 
in the root folder of the Haystack repository. This will start three containers (Elasticsearch, Haystack API, Haystack UI).
You can find the UI at `http://localhost:8501`