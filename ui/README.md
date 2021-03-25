## Demo UI

This is a minimal UI that can spin up to test Haystack for your prototypes. It's based on streamlit and is very easy to extend for your purposes. 

![Screenshot](https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/_src/img/streamlit_ui_screenshot.png)

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

## Evaluation Mode

The evaluation mode leverages the feedback REST API endpoint of haystack. The user has the options "Wrong answer", "Wrong answer and wrong passage" and "Wrong answer and wrong passage" to give feedback. 

To enter the evaluation mode, select the checkbox "Evaluation mode" in the sidebar. The UI will load the predefined questions from the file `eval_lables_examles`. The file needs to be prefilled with your data. This way, the user will get a random question from the set and can give his feedback with the buttons below the questions. To load a new questions, click the button "Get random question".

The feedback can be exported with the API endpoint `export-doc-qa-feedback`.

![Screenshot](https://raw.githubusercontent.com/deepset-ai/haystack/master/docs/_src/img/streamlit_ui_screenshot_eval_mode.png)