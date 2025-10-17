---
title: Evaluators
id: evaluators-api
description: Evaluate your pipelines or individual components.
---

<a id="answer_exact_match"></a>

# Module answer\_exact\_match

<a id="answer_exact_match.AnswerExactMatchEvaluator"></a>

## AnswerExactMatchEvaluator

An answer exact match evaluator class.

The evaluator that checks if the predicted answers matches any of the ground truth answers exactly.
The result is a number from 0.0 to 1.0, it represents the proportion of predicted answers
that matched one of the ground truth answers.
There can be multiple ground truth answers and multiple predicted answers as input.


Usage example:
```python
from haystack.components.evaluators import AnswerExactMatchEvaluator

evaluator = AnswerExactMatchEvaluator()
result = evaluator.run(
    ground_truth_answers=["Berlin", "Paris"],
    predicted_answers=["Berlin", "Lyon"],
)

print(result["individual_scores"])
# [1, 0]
print(result["score"])
# 0.5
```

<a id="answer_exact_match.AnswerExactMatchEvaluator.run"></a>

#### AnswerExactMatchEvaluator.run

```python
@component.output_types(individual_scores=list[int], score=float)
def run(ground_truth_answers: list[str],
        predicted_answers: list[str]) -> dict[str, Any]
```

Run the AnswerExactMatchEvaluator on the given inputs.

The `ground_truth_answers` and `retrieved_answers` must have the same length.

**Arguments**:

- `ground_truth_answers`: A list of expected answers.
- `predicted_answers`: A list of predicted answers.

**Returns**:

A dictionary with the following outputs:
- `individual_scores` - A list of 0s and 1s, where 1 means that the predicted answer matched one of the
    ground truth.
- `score` - A number from 0.0 to 1.0 that represents the proportion of questions where any predicted
             answer matched one of the ground truth answers.

<a id="context_relevance"></a>

# Module context\_relevance

<a id="context_relevance.ContextRelevanceEvaluator"></a>

## ContextRelevanceEvaluator

Evaluator that checks if a provided context is relevant to the question.

An LLM breaks up a context into multiple statements and checks whether each statement
is relevant for answering a question.
The score for each context is either binary score of 1 or 0, where 1 indicates that the context is relevant
to the question and 0 indicates that the context is not relevant.
The evaluator also provides the relevant statements from the context and an average score over all the provided
input questions contexts pairs.

Usage example:
```python
from haystack.components.evaluators import ContextRelevanceEvaluator

questions = ["Who created the Python language?", "Why does Java needs a JVM?", "Is C++ better than Python?"]
contexts = [
    [(
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming "
        "language. Its design philosophy emphasizes code readability, and its language constructs aim to help "
        "programmers write clear, logical code for both small and large-scale software projects."
    )],
    [(
        "Java is a high-level, class-based, object-oriented programming language that is designed to have as few "
        "implementation dependencies as possible. The JVM has two primary functions: to allow Java programs to run"
        "on any device or operating system (known as the 'write once, run anywhere' principle), and to manage and"
        "optimize program memory."
    )],
    [(
        "C++ is a general-purpose programming language created by Bjarne Stroustrup as an extension of the C "
        "programming language."
    )],
]

evaluator = ContextRelevanceEvaluator()
result = evaluator.run(questions=questions, contexts=contexts)
print(result["score"])
# 0.67
print(result["individual_scores"])
# [1,1,0]
print(result["results"])
# [{
#   'relevant_statements': ['Python, created by Guido van Rossum in the late 1980s.'],
#    'score': 1.0
#  },
#  {
#   'relevant_statements': ['The JVM has two primary functions: to allow Java programs to run on any device or
#                           operating system (known as the "write once, run anywhere" principle), and to manage and
#                           optimize program memory'],
#   'score': 1.0
#  },
#  {
#   'relevant_statements': [],
#   'score': 0.0
#  }]
```

<a id="context_relevance.ContextRelevanceEvaluator.__init__"></a>

#### ContextRelevanceEvaluator.\_\_init\_\_

```python
def __init__(examples: Optional[list[dict[str, Any]]] = None,
             progress_bar: bool = True,
             raise_on_failure: bool = True,
             chat_generator: Optional[ChatGenerator] = None)
```

Creates an instance of ContextRelevanceEvaluator.

If no LLM is specified using the `chat_generator` parameter, the component will use OpenAI in JSON mode.

**Arguments**:

- `examples`: Optional few-shot examples conforming to the expected input and output format of ContextRelevanceEvaluator.
Default examples will be used if none are provided.
Each example must be a dictionary with keys "inputs" and "outputs".
"inputs" must be a dictionary with keys "questions" and "contexts".
"outputs" must be a dictionary with "relevant_statements".
Expected format:
```python
[{
    "inputs": {
        "questions": "What is the capital of Italy?", "contexts": ["Rome is the capital of Italy."],
    },
    "outputs": {
        "relevant_statements": ["Rome is the capital of Italy."],
    },
}]
```
- `progress_bar`: Whether to show a progress bar during the evaluation.
- `raise_on_failure`: Whether to raise an exception if the API call fails.
- `chat_generator`: a ChatGenerator instance which represents the LLM.
In order for the component to work, the LLM should be configured to return a JSON object. For example,
when using the OpenAIChatGenerator, you should pass `{"response_format": {"type": "json_object"}}` in the
`generation_kwargs`.

<a id="context_relevance.ContextRelevanceEvaluator.run"></a>

#### ContextRelevanceEvaluator.run

```python
@component.output_types(score=float, results=list[dict[str, Any]])
def run(**inputs) -> dict[str, Any]
```

Run the LLM evaluator.

**Arguments**:

- `questions`: A list of questions.
- `contexts`: A list of lists of contexts. Each list of contexts corresponds to one question.

**Returns**:

A dictionary with the following outputs:
- `score`: Mean context relevance score over all the provided input questions.
- `results`: A list of dictionaries with `relevant_statements` and `score` for each input context.

<a id="context_relevance.ContextRelevanceEvaluator.to_dict"></a>

#### ContextRelevanceEvaluator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

A dictionary with serialized data.

<a id="context_relevance.ContextRelevanceEvaluator.from_dict"></a>

#### ContextRelevanceEvaluator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "ContextRelevanceEvaluator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="context_relevance.ContextRelevanceEvaluator.validate_init_parameters"></a>

#### ContextRelevanceEvaluator.validate\_init\_parameters

```python
@staticmethod
def validate_init_parameters(inputs: list[tuple[str, type[list]]],
                             outputs: list[str], examples: list[dict[str,
                                                                     Any]])
```

Validate the init parameters.

**Arguments**:

- `inputs`: The inputs to validate.
- `outputs`: The outputs to validate.
- `examples`: The examples to validate.

**Raises**:

- `ValueError`: If the inputs are not a list of tuples with a string and a type of list.
If the outputs are not a list of strings.
If the examples are not a list of dictionaries.
If any example does not have keys "inputs" and "outputs" with values that are dictionaries with string keys.

<a id="context_relevance.ContextRelevanceEvaluator.prepare_template"></a>

#### ContextRelevanceEvaluator.prepare\_template

```python
def prepare_template() -> str
```

Prepare the prompt template.

Combine instructions, inputs, outputs, and examples into one prompt template with the following format:
Instructions:
`<instructions>`

Generate the response in JSON format with the following keys:
`<list of output keys>`
Consider the instructions and the examples below to determine those values.

Examples:
`<examples>`

Inputs:
`<inputs>`
Outputs:

**Returns**:

The prompt template.

<a id="context_relevance.ContextRelevanceEvaluator.validate_input_parameters"></a>

#### ContextRelevanceEvaluator.validate\_input\_parameters

```python
@staticmethod
def validate_input_parameters(expected: dict[str, Any],
                              received: dict[str, Any]) -> None
```

Validate the input parameters.

**Arguments**:

- `expected`: The expected input parameters.
- `received`: The received input parameters.

**Raises**:

- `ValueError`: If not all expected inputs are present in the received inputs
If the received inputs are not lists or have different lengths

<a id="context_relevance.ContextRelevanceEvaluator.is_valid_json_and_has_expected_keys"></a>

#### ContextRelevanceEvaluator.is\_valid\_json\_and\_has\_expected\_keys

```python
def is_valid_json_and_has_expected_keys(expected: list[str],
                                        received: str) -> bool
```

Output must be a valid JSON with the expected keys.

**Arguments**:

- `expected`: Names of expected outputs
- `received`: Names of received outputs

**Raises**:

- `ValueError`: If the output is not a valid JSON with the expected keys:
- with `raise_on_failure` set to True a ValueError is raised.
- with `raise_on_failure` set to False a warning is issued and False is returned.

**Returns**:

True if the received output is a valid JSON with the expected keys, False otherwise.

<a id="document_map"></a>

# Module document\_map

<a id="document_map.DocumentMAPEvaluator"></a>

## DocumentMAPEvaluator

A Mean Average Precision (MAP) evaluator for documents.

Evaluator that calculates the mean average precision of the retrieved documents, a metric
that measures how high retrieved documents are ranked.
Each question can have multiple ground truth documents and multiple retrieved documents.

`DocumentMAPEvaluator` doesn't normalize its inputs, the `DocumentCleaner` component
should be used to clean and normalize the documents before passing them to this evaluator.

Usage example:
```python
from haystack import Document
from haystack.components.evaluators import DocumentMAPEvaluator

evaluator = DocumentMAPEvaluator()
result = evaluator.run(
    ground_truth_documents=[
        [Document(content="France")],
        [Document(content="9th century"), Document(content="9th")],
    ],
    retrieved_documents=[
        [Document(content="France")],
        [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
    ],
)

print(result["individual_scores"])
# [1.0, 0.8333333333333333]
print(result["score"])
# 0.9166666666666666
```

<a id="document_map.DocumentMAPEvaluator.run"></a>

#### DocumentMAPEvaluator.run

```python
@component.output_types(score=float, individual_scores=list[float])
def run(ground_truth_documents: list[list[Document]],
        retrieved_documents: list[list[Document]]) -> dict[str, Any]
```

Run the DocumentMAPEvaluator on the given inputs.

All lists must have the same length.

**Arguments**:

- `ground_truth_documents`: A list of expected documents for each question.
- `retrieved_documents`: A list of retrieved documents for each question.

**Returns**:

A dictionary with the following outputs:
- `score` - The average of calculated scores.
- `individual_scores` - A list of numbers from 0.0 to 1.0 that represents how high retrieved documents
    are ranked.

<a id="document_mrr"></a>

# Module document\_mrr

<a id="document_mrr.DocumentMRREvaluator"></a>

## DocumentMRREvaluator

Evaluator that calculates the mean reciprocal rank of the retrieved documents.

MRR measures how high the first retrieved document is ranked.
Each question can have multiple ground truth documents and multiple retrieved documents.

`DocumentMRREvaluator` doesn't normalize its inputs, the `DocumentCleaner` component
should be used to clean and normalize the documents before passing them to this evaluator.

Usage example:
```python
from haystack import Document
from haystack.components.evaluators import DocumentMRREvaluator

evaluator = DocumentMRREvaluator()
result = evaluator.run(
    ground_truth_documents=[
        [Document(content="France")],
        [Document(content="9th century"), Document(content="9th")],
    ],
    retrieved_documents=[
        [Document(content="France")],
        [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
    ],
)
print(result["individual_scores"])
# [1.0, 1.0]
print(result["score"])
# 1.0
```

<a id="document_mrr.DocumentMRREvaluator.run"></a>

#### DocumentMRREvaluator.run

```python
@component.output_types(score=float, individual_scores=list[float])
def run(ground_truth_documents: list[list[Document]],
        retrieved_documents: list[list[Document]]) -> dict[str, Any]
```

Run the DocumentMRREvaluator on the given inputs.

`ground_truth_documents` and `retrieved_documents` must have the same length.

**Arguments**:

- `ground_truth_documents`: A list of expected documents for each question.
- `retrieved_documents`: A list of retrieved documents for each question.

**Returns**:

A dictionary with the following outputs:
- `score` - The average of calculated scores.
- `individual_scores` - A list of numbers from 0.0 to 1.0 that represents how high the first retrieved
    document is ranked.

<a id="document_ndcg"></a>

# Module document\_ndcg

<a id="document_ndcg.DocumentNDCGEvaluator"></a>

## DocumentNDCGEvaluator

Evaluator that calculates the normalized discounted cumulative gain (NDCG) of retrieved documents.

Each question can have multiple ground truth documents and multiple retrieved documents.
If the ground truth documents have relevance scores, the NDCG calculation uses these scores.
Otherwise, it assumes binary relevance of all ground truth documents.

Usage example:
```python
from haystack import Document
from haystack.components.evaluators import DocumentNDCGEvaluator

evaluator = DocumentNDCGEvaluator()
result = evaluator.run(
    ground_truth_documents=[[Document(content="France", score=1.0), Document(content="Paris", score=0.5)]],
    retrieved_documents=[[Document(content="France"), Document(content="Germany"), Document(content="Paris")]],
)
print(result["individual_scores"])
# [0.8869]
print(result["score"])
# 0.8869
```

<a id="document_ndcg.DocumentNDCGEvaluator.run"></a>

#### DocumentNDCGEvaluator.run

```python
@component.output_types(score=float, individual_scores=list[float])
def run(ground_truth_documents: list[list[Document]],
        retrieved_documents: list[list[Document]]) -> dict[str, Any]
```

Run the DocumentNDCGEvaluator on the given inputs.

`ground_truth_documents` and `retrieved_documents` must have the same length.
The list items within `ground_truth_documents` and `retrieved_documents` can differ in length.

**Arguments**:

- `ground_truth_documents`: Lists of expected documents, one list per question. Binary relevance is used if documents have no scores.
- `retrieved_documents`: Lists of retrieved documents, one list per question.

**Returns**:

A dictionary with the following outputs:
- `score` - The average of calculated scores.
- `individual_scores` - A list of numbers from 0.0 to 1.0 that represents the NDCG for each question.

<a id="document_ndcg.DocumentNDCGEvaluator.validate_inputs"></a>

#### DocumentNDCGEvaluator.validate\_inputs

```python
@staticmethod
def validate_inputs(gt_docs: list[list[Document]],
                    ret_docs: list[list[Document]])
```

Validate the input parameters.

**Arguments**:

- `gt_docs`: The ground_truth_documents to validate.
- `ret_docs`: The retrieved_documents to validate.

**Raises**:

- `ValueError`: If the ground_truth_documents or the retrieved_documents are an empty a list.
If the length of ground_truth_documents and retrieved_documents differs.
If any list of documents in ground_truth_documents contains a mix of documents with and without a score.

<a id="document_ndcg.DocumentNDCGEvaluator.calculate_dcg"></a>

#### DocumentNDCGEvaluator.calculate\_dcg

```python
@staticmethod
def calculate_dcg(gt_docs: list[Document], ret_docs: list[Document]) -> float
```

Calculate the discounted cumulative gain (DCG) of the retrieved documents.

**Arguments**:

- `gt_docs`: The ground truth documents.
- `ret_docs`: The retrieved documents.

**Returns**:

The discounted cumulative gain (DCG) of the retrieved
documents based on the ground truth documents.

<a id="document_ndcg.DocumentNDCGEvaluator.calculate_idcg"></a>

#### DocumentNDCGEvaluator.calculate\_idcg

```python
@staticmethod
def calculate_idcg(gt_docs: list[Document]) -> float
```

Calculate the ideal discounted cumulative gain (IDCG) of the ground truth documents.

**Arguments**:

- `gt_docs`: The ground truth documents.

**Returns**:

The ideal discounted cumulative gain (IDCG) of the ground truth documents.

<a id="document_recall"></a>

# Module document\_recall

<a id="document_recall.RecallMode"></a>

## RecallMode

Enum for the mode to use for calculating the recall score.

<a id="document_recall.RecallMode.from_str"></a>

#### RecallMode.from\_str

```python
@staticmethod
def from_str(string: str) -> "RecallMode"
```

Convert a string to a RecallMode enum.

<a id="document_recall.DocumentRecallEvaluator"></a>

## DocumentRecallEvaluator

Evaluator that calculates the Recall score for a list of documents.

Returns both a list of scores for each question and the average.
There can be multiple ground truth documents and multiple predicted documents as input.

Usage example:
```python
from haystack import Document
from haystack.components.evaluators import DocumentRecallEvaluator

evaluator = DocumentRecallEvaluator()
result = evaluator.run(
    ground_truth_documents=[
        [Document(content="France")],
        [Document(content="9th century"), Document(content="9th")],
    ],
    retrieved_documents=[
        [Document(content="France")],
        [Document(content="9th century"), Document(content="10th century"), Document(content="9th")],
    ],
)
print(result["individual_scores"])
# [1.0, 1.0]
print(result["score"])
# 1.0
```

<a id="document_recall.DocumentRecallEvaluator.__init__"></a>

#### DocumentRecallEvaluator.\_\_init\_\_

```python
def __init__(mode: Union[str, RecallMode] = RecallMode.SINGLE_HIT)
```

Create a DocumentRecallEvaluator component.

**Arguments**:

- `mode`: Mode to use for calculating the recall score.

<a id="document_recall.DocumentRecallEvaluator.run"></a>

#### DocumentRecallEvaluator.run

```python
@component.output_types(score=float, individual_scores=list[float])
def run(ground_truth_documents: list[list[Document]],
        retrieved_documents: list[list[Document]]) -> dict[str, Any]
```

Run the DocumentRecallEvaluator on the given inputs.

`ground_truth_documents` and `retrieved_documents` must have the same length.

**Arguments**:

- `ground_truth_documents`: A list of expected documents for each question.
- `retrieved_documents`: A list of retrieved documents for each question.
A dictionary with the following outputs:
- `score` - The average of calculated scores.
- `individual_scores` - A list of numbers from 0.0 to 1.0 that represents the proportion of matching
    documents retrieved. If the mode is `single_hit`, the individual scores are 0 or 1.

<a id="document_recall.DocumentRecallEvaluator.to_dict"></a>

#### DocumentRecallEvaluator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns**:

Dictionary with serialized data.

<a id="faithfulness"></a>

# Module faithfulness

<a id="faithfulness.FaithfulnessEvaluator"></a>

## FaithfulnessEvaluator

Evaluator that checks if a generated answer can be inferred from the provided contexts.

An LLM separates the answer into multiple statements and checks whether the statement can be inferred from the
context or not. The final score for the full answer is a number from 0.0 to 1.0. It represents the proportion of
statements that can be inferred from the provided contexts.

Usage example:
```python
from haystack.components.evaluators import FaithfulnessEvaluator

questions = ["Who created the Python language?"]
contexts = [
    [(
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming "
        "language. Its design philosophy emphasizes code readability, and its language constructs aim to help "
        "programmers write clear, logical code for both small and large-scale software projects."
    )],
]
predicted_answers = [
    "Python is a high-level general-purpose programming language that was created by George Lucas."
]
evaluator = FaithfulnessEvaluator()
result = evaluator.run(questions=questions, contexts=contexts, predicted_answers=predicted_answers)

print(result["individual_scores"])
# [0.5]
print(result["score"])
# 0.5
print(result["results"])
# [{'statements': ['Python is a high-level general-purpose programming language.',
'Python was created by George Lucas.'], 'statement_scores': [1, 0], 'score': 0.5}]
```

<a id="faithfulness.FaithfulnessEvaluator.__init__"></a>

#### FaithfulnessEvaluator.\_\_init\_\_

```python
def __init__(examples: Optional[list[dict[str, Any]]] = None,
             progress_bar: bool = True,
             raise_on_failure: bool = True,
             chat_generator: Optional[ChatGenerator] = None)
```

Creates an instance of FaithfulnessEvaluator.

If no LLM is specified using the `chat_generator` parameter, the component will use OpenAI in JSON mode.

**Arguments**:

- `examples`: Optional few-shot examples conforming to the expected input and output format of FaithfulnessEvaluator.
Default examples will be used if none are provided.
Each example must be a dictionary with keys "inputs" and "outputs".
"inputs" must be a dictionary with keys "questions", "contexts", and "predicted_answers".
"outputs" must be a dictionary with "statements" and "statement_scores".
Expected format:
```python
[{
    "inputs": {
        "questions": "What is the capital of Italy?", "contexts": ["Rome is the capital of Italy."],
        "predicted_answers": "Rome is the capital of Italy with more than 4 million inhabitants.",
    },
    "outputs": {
        "statements": ["Rome is the capital of Italy.", "Rome has more than 4 million inhabitants."],
        "statement_scores": [1, 0],
    },
}]
```
- `progress_bar`: Whether to show a progress bar during the evaluation.
- `raise_on_failure`: Whether to raise an exception if the API call fails.
- `chat_generator`: a ChatGenerator instance which represents the LLM.
In order for the component to work, the LLM should be configured to return a JSON object. For example,
when using the OpenAIChatGenerator, you should pass `{"response_format": {"type": "json_object"}}` in the
`generation_kwargs`.

<a id="faithfulness.FaithfulnessEvaluator.run"></a>

#### FaithfulnessEvaluator.run

```python
@component.output_types(individual_scores=list[int],
                        score=float,
                        results=list[dict[str, Any]])
def run(**inputs) -> dict[str, Any]
```

Run the LLM evaluator.

**Arguments**:

- `questions`: A list of questions.
- `contexts`: A nested list of contexts that correspond to the questions.
- `predicted_answers`: A list of predicted answers.

**Returns**:

A dictionary with the following outputs:
- `score`: Mean faithfulness score over all the provided input answers.
- `individual_scores`: A list of faithfulness scores for each input answer.
- `results`: A list of dictionaries with `statements` and `statement_scores` for each input answer.

<a id="faithfulness.FaithfulnessEvaluator.to_dict"></a>

#### FaithfulnessEvaluator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

A dictionary with serialized data.

<a id="faithfulness.FaithfulnessEvaluator.from_dict"></a>

#### FaithfulnessEvaluator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "FaithfulnessEvaluator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="faithfulness.FaithfulnessEvaluator.validate_init_parameters"></a>

#### FaithfulnessEvaluator.validate\_init\_parameters

```python
@staticmethod
def validate_init_parameters(inputs: list[tuple[str, type[list]]],
                             outputs: list[str], examples: list[dict[str,
                                                                     Any]])
```

Validate the init parameters.

**Arguments**:

- `inputs`: The inputs to validate.
- `outputs`: The outputs to validate.
- `examples`: The examples to validate.

**Raises**:

- `ValueError`: If the inputs are not a list of tuples with a string and a type of list.
If the outputs are not a list of strings.
If the examples are not a list of dictionaries.
If any example does not have keys "inputs" and "outputs" with values that are dictionaries with string keys.

<a id="faithfulness.FaithfulnessEvaluator.prepare_template"></a>

#### FaithfulnessEvaluator.prepare\_template

```python
def prepare_template() -> str
```

Prepare the prompt template.

Combine instructions, inputs, outputs, and examples into one prompt template with the following format:
Instructions:
`<instructions>`

Generate the response in JSON format with the following keys:
`<list of output keys>`
Consider the instructions and the examples below to determine those values.

Examples:
`<examples>`

Inputs:
`<inputs>`
Outputs:

**Returns**:

The prompt template.

<a id="faithfulness.FaithfulnessEvaluator.validate_input_parameters"></a>

#### FaithfulnessEvaluator.validate\_input\_parameters

```python
@staticmethod
def validate_input_parameters(expected: dict[str, Any],
                              received: dict[str, Any]) -> None
```

Validate the input parameters.

**Arguments**:

- `expected`: The expected input parameters.
- `received`: The received input parameters.

**Raises**:

- `ValueError`: If not all expected inputs are present in the received inputs
If the received inputs are not lists or have different lengths

<a id="faithfulness.FaithfulnessEvaluator.is_valid_json_and_has_expected_keys"></a>

#### FaithfulnessEvaluator.is\_valid\_json\_and\_has\_expected\_keys

```python
def is_valid_json_and_has_expected_keys(expected: list[str],
                                        received: str) -> bool
```

Output must be a valid JSON with the expected keys.

**Arguments**:

- `expected`: Names of expected outputs
- `received`: Names of received outputs

**Raises**:

- `ValueError`: If the output is not a valid JSON with the expected keys:
- with `raise_on_failure` set to True a ValueError is raised.
- with `raise_on_failure` set to False a warning is issued and False is returned.

**Returns**:

True if the received output is a valid JSON with the expected keys, False otherwise.

<a id="llm_evaluator"></a>

# Module llm\_evaluator

<a id="llm_evaluator.LLMEvaluator"></a>

## LLMEvaluator

Uses an LLM to evaluate inputs based on a prompt containing instructions and examples.

The default API requires an OpenAI API key to be provided as an environment variable "OPENAI_API_KEY".
The inputs are lists that are user-defined depending on the desired metric.
The output is a dictionary with a key `results` containing a list of evaluation results.
Each result is a dictionary with user-defined keys and values of either 0 for FALSE or 1 for TRUE respectively.

Usage example:
```python
from haystack.components.evaluators import LLMEvaluator
evaluator = LLMEvaluator(
    instructions="Is this answer problematic for children?",
    inputs=[("predicted_answers", list[str])],
    outputs=["score"],
    examples=[
        {"inputs": {"predicted_answers": "Damn, this is straight outta hell!!!"}, "outputs": {"score": 1}},
        {"inputs": {"predicted_answers": "Football is the most popular sport."}, "outputs": {"score": 0}},
    ],
)
predicted_answers = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]
results = evaluator.run(predicted_answers=predicted_answers)
print(results)
# {'results': [{'score': 0}, {'score': 0}]}
```

<a id="llm_evaluator.LLMEvaluator.__init__"></a>

#### LLMEvaluator.\_\_init\_\_

```python
def __init__(instructions: str,
             inputs: list[tuple[str, type[list]]],
             outputs: list[str],
             examples: list[dict[str, Any]],
             progress_bar: bool = True,
             *,
             raise_on_failure: bool = True,
             chat_generator: Optional[ChatGenerator] = None)
```

Creates an instance of LLMEvaluator.

If no LLM is specified using the `chat_generator` parameter, the component will use OpenAI in JSON mode.

**Arguments**:

- `instructions`: The prompt instructions to use for evaluation.
Should be a question about the inputs that can be answered with yes or no.
- `inputs`: The inputs that the component expects as incoming connections and that it evaluates.
Each input is a tuple of an input name and input type. Input types must be lists.
- `outputs`: Output names of the evaluation results. They correspond to keys in the output dictionary.
- `examples`: Few-shot examples conforming to the expected input and output format as defined in the `inputs` and
`outputs` parameters.
Each example is a dictionary with keys "inputs" and "outputs"
They contain the input and output as dictionaries respectively.
- `raise_on_failure`: If True, the component will raise an exception on an unsuccessful API call.
- `progress_bar`: Whether to show a progress bar during the evaluation.
- `chat_generator`: a ChatGenerator instance which represents the LLM.
In order for the component to work, the LLM should be configured to return a JSON object. For example,
when using the OpenAIChatGenerator, you should pass `{"response_format": {"type": "json_object"}}` in the
`generation_kwargs`.

<a id="llm_evaluator.LLMEvaluator.validate_init_parameters"></a>

#### LLMEvaluator.validate\_init\_parameters

```python
@staticmethod
def validate_init_parameters(inputs: list[tuple[str, type[list]]],
                             outputs: list[str], examples: list[dict[str,
                                                                     Any]])
```

Validate the init parameters.

**Arguments**:

- `inputs`: The inputs to validate.
- `outputs`: The outputs to validate.
- `examples`: The examples to validate.

**Raises**:

- `ValueError`: If the inputs are not a list of tuples with a string and a type of list.
If the outputs are not a list of strings.
If the examples are not a list of dictionaries.
If any example does not have keys "inputs" and "outputs" with values that are dictionaries with string keys.

<a id="llm_evaluator.LLMEvaluator.run"></a>

#### LLMEvaluator.run

```python
@component.output_types(results=list[dict[str, Any]])
def run(**inputs) -> dict[str, Any]
```

Run the LLM evaluator.

**Arguments**:

- `inputs`: The input values to evaluate. The keys are the input names and the values are lists of input values.

**Raises**:

- `ValueError`: Only in the case that  `raise_on_failure` is set to True and the received inputs are not lists or have
different lengths, or if the output is not a valid JSON or doesn't contain the expected keys.

**Returns**:

A dictionary with a `results` entry that contains a list of results.
Each result is a dictionary containing the keys as defined in the `outputs` parameter of the LLMEvaluator
and the evaluation results as the values. If an exception occurs for a particular input value, the result
will be `None` for that entry.
If the API is "openai" and the response contains a "meta" key, the metadata from OpenAI will be included
in the output dictionary, under the key "meta".

<a id="llm_evaluator.LLMEvaluator.prepare_template"></a>

#### LLMEvaluator.prepare\_template

```python
def prepare_template() -> str
```

Prepare the prompt template.

Combine instructions, inputs, outputs, and examples into one prompt template with the following format:
Instructions:
`<instructions>`

Generate the response in JSON format with the following keys:
`<list of output keys>`
Consider the instructions and the examples below to determine those values.

Examples:
`<examples>`

Inputs:
`<inputs>`
Outputs:

**Returns**:

The prompt template.

<a id="llm_evaluator.LLMEvaluator.to_dict"></a>

#### LLMEvaluator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="llm_evaluator.LLMEvaluator.from_dict"></a>

#### LLMEvaluator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "LLMEvaluator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="llm_evaluator.LLMEvaluator.validate_input_parameters"></a>

#### LLMEvaluator.validate\_input\_parameters

```python
@staticmethod
def validate_input_parameters(expected: dict[str, Any],
                              received: dict[str, Any]) -> None
```

Validate the input parameters.

**Arguments**:

- `expected`: The expected input parameters.
- `received`: The received input parameters.

**Raises**:

- `ValueError`: If not all expected inputs are present in the received inputs
If the received inputs are not lists or have different lengths

<a id="llm_evaluator.LLMEvaluator.is_valid_json_and_has_expected_keys"></a>

#### LLMEvaluator.is\_valid\_json\_and\_has\_expected\_keys

```python
def is_valid_json_and_has_expected_keys(expected: list[str],
                                        received: str) -> bool
```

Output must be a valid JSON with the expected keys.

**Arguments**:

- `expected`: Names of expected outputs
- `received`: Names of received outputs

**Raises**:

- `ValueError`: If the output is not a valid JSON with the expected keys:
- with `raise_on_failure` set to True a ValueError is raised.
- with `raise_on_failure` set to False a warning is issued and False is returned.

**Returns**:

True if the received output is a valid JSON with the expected keys, False otherwise.

<a id="sas_evaluator"></a>

# Module sas\_evaluator

<a id="sas_evaluator.SASEvaluator"></a>

## SASEvaluator

SASEvaluator computes the Semantic Answer Similarity (SAS) between a list of predictions and a one of ground truths.

It's usually used in Retrieval Augmented Generation (RAG) pipelines to evaluate the quality of the generated
answers. The SAS is computed using a pre-trained model from the Hugging Face model hub. The model can be either a
Bi-Encoder or a Cross-Encoder. The choice of the model is based on the `model` parameter.

Usage example:
```python
from haystack.components.evaluators.sas_evaluator import SASEvaluator

evaluator = SASEvaluator(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
evaluator.warm_up()
ground_truths = [
    "A construction budget of US $2.3 billion",
    "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
    "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
]
predictions = [
    "A construction budget of US $2.3 billion",
    "The Eiffel Tower, completed in 1889, symbolizes Paris's cultural magnificence.",
    "The Meiji Restoration in 1868 transformed Japan into a modernized world power.",
]
result = evaluator.run(
    ground_truths_answers=ground_truths, predicted_answers=predictions
)

print(result["score"])
# 0.9999673763910929

print(result["individual_scores"])
# [0.9999765157699585, 0.999968409538269, 0.9999572038650513]
```

<a id="sas_evaluator.SASEvaluator.__init__"></a>

#### SASEvaluator.\_\_init\_\_

```python
def __init__(
    model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    batch_size: int = 32,
    device: Optional[ComponentDevice] = None,
    token: Secret = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"],
                                        strict=False))
```

Creates a new instance of SASEvaluator.

**Arguments**:

- `model`: SentenceTransformers semantic textual similarity model, should be path or string pointing to a downloadable
model.
- `batch_size`: Number of prediction-label pairs to encode at once.
- `device`: The device on which the model is loaded. If `None`, the default device is automatically selected.
- `token`: The Hugging Face token for HTTP bearer authorization.
You can find your HF token in your [account settings](https://huggingface.co/settings/tokens)

<a id="sas_evaluator.SASEvaluator.to_dict"></a>

#### SASEvaluator.to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serialize this component to a dictionary.

**Returns**:

The serialized component as a dictionary.

<a id="sas_evaluator.SASEvaluator.from_dict"></a>

#### SASEvaluator.from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "SASEvaluator"
```

Deserialize this component from a dictionary.

**Arguments**:

- `data`: The dictionary representation of this component.

**Returns**:

The deserialized component instance.

<a id="sas_evaluator.SASEvaluator.warm_up"></a>

#### SASEvaluator.warm\_up

```python
def warm_up()
```

Initializes the component.

<a id="sas_evaluator.SASEvaluator.run"></a>

#### SASEvaluator.run

```python
@component.output_types(score=float, individual_scores=list[float])
def run(ground_truth_answers: list[str],
        predicted_answers: list[str]) -> dict[str, Any]
```

SASEvaluator component run method.

Run the SASEvaluator to compute the Semantic Answer Similarity (SAS) between a list of predicted answers
and a list of ground truth answers. Both must be list of strings of same length.

**Arguments**:

- `ground_truth_answers`: A list of expected answers for each question.
- `predicted_answers`: A list of generated answers for each question.

**Returns**:

A dictionary with the following outputs:
- `score`: Mean SAS score over all the predictions/ground-truth pairs.
- `individual_scores`: A list of similarity scores for each prediction/ground-truth pair.
