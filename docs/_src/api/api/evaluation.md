<a name="eval"></a>
# Module eval

<a name="eval.EvalRetriever"></a>
## EvalRetriever Objects

```python
class EvalRetriever()
```

This is a pipeline node that should be placed after a Retriever in order to assess its performance. Performance
metrics are stored in this class and updated as each sample passes through it. To view the results of the evaluation,
call EvalRetriever.print()

<a name="eval.EvalRetriever.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(debug=False, open_domain=True)
```

**Arguments**:

- `open_domain`: When True, a document is considered correctly retrieved so long as the answer string can be found within it.
When False, correct retrieval is evaluated based on document_id.
:type open_domain: bool
- `debug`: When True, a record of each sample and its evaluation will be stored in EvalRetriever.log
:type debug: bool

<a name="eval.EvalRetriever.run"></a>
#### run

```python
 | run(documents, labels: dict, **kwargs)
```

Run this node on one sample and its labels

<a name="eval.EvalRetriever.print"></a>
#### print

```python
 | print()
```

Print the evaluation results

<a name="eval.EvalReader"></a>
## EvalReader Objects

```python
class EvalReader()
```

This is a pipeline node that should be placed after a Reader in order to assess the performance of the Reader
individually or to assess the extractive QA performance of the whole pipeline. Performance metrics are stored in
this class and updated as each sample passes through it. To view the results of the evaluation, call EvalReader.print()

<a name="eval.EvalReader.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(skip_incorrect_retrieval=True, open_domain=True, debug=False)
```

**Arguments**:

- `skip_incorrect_retrieval`: When set to True, this eval will ignore the cases where the retriever returned no correct documents
:type skip_incorrect_retrieval: bool
- `open_domain`: When True, extracted answers are evaluated purely on string similarity rather than the position of the extracted answer
:type open_domain: bool
- `debug`: When True, a record of each sample and its evaluation will be stored in EvalReader.log
:type debug: bool

<a name="eval.EvalReader.run"></a>
#### run

```python
 | run(labels, answers, **kwargs)
```

Run this node on one sample and its labels

<a name="eval.EvalReader.print"></a>
#### print

```python
 | print(mode)
```

Print the evaluation results

