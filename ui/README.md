## Demo UI

This is a minimal UI that can spin up to test Haystack for your prototypes. It's based on streamlit and is very easy to extend for your purposes. 

![Screenshot](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/streamlit_ui_screenshot.png)

## Usage

### Get started with Haystack

The UI interacts with the Haystack REST API. To get started with Haystack please visit the [README](https://github.com/deepset-ai/haystack/tree/main#key-components) or checko out our [tutorials](https://haystack.deepset.ai/tutorials/first-qa-system).

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

In order to use the UI in evaluation mode, you need an ElasticSearch instance with pre-indexed files and the Haystack REST API. You can set the environment up via docker images. For ElasticSearch, you can check out our [documentation](https://haystack.deepset.ai/usage/document-store#initialisation) and for setting up the REST API this [link](https://github.com/deepset-ai/haystack/blob/main/README.md#7-rest-api).

To enter the evaluation mode, select the checkbox "Evaluation mode" in the sidebar. The UI will load the predefined questions from the file [`eval_labels_examples`](https://raw.githubusercontent.com/deepset-ai/haystack/main/ui/eval_labels_example.csv). The file needs to be prefilled with your data. This way, the user will get a random question from the set and can give his feedback with the buttons below the questions. To load a new question, click the button "Get random question". 

The file just needs to have two columns separated by semicolon. You can add more columns but the UI will ignore them. Every line represents a questions answer pair. The columns with the questions needs to be named “Question Text” and the answer column “Answer” so that they can be loaded correctly. Currently, the easiest way to create the file is manually by adding question answer pairs. 

The feedback can be exported with the API endpoint `export-doc-qa-feedback`. To learn more about finetuning a model with user feedback, please check out our [docs](https://haystack.deepset.ai/usage/domain-adaptation#user-feedback).

![Screenshot](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/streamlit_ui_screenshot_eval_mode.png)