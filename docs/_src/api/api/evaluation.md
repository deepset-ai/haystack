<a name="eval"></a>
# Module eval

<a name="eval.EvalRetriever"></a>
## EvalRetriever Objects

```python
class EvalRetriever()
```

This is a pipeline node that should be placed after a Retriever in order to assess its performance. Performance
metrics are stored in this class and updated as each sample passes through it. To view the results of the evaluation,
call EvalRetriever.print(). Note that results from this Node may differ from that when calling Retriever.eval()
since that is a closed domain evaluation. Have a look at our evaluation tutorial for more info about
open vs closed domain eval (https://haystack.deepset.ai/docs/latest/tutorial5md).

<a name="eval.EvalRetriever.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(debug: bool = False, open_domain: bool = True)
```

**Arguments**:

- `open_domain`: When True, a document is considered correctly retrieved so long as the answer string can be found within it.
                    When False, correct retrieval is evaluated based on document_id.
- `debug`: When True, a record of each sample and its evaluation will be stored in EvalRetriever.log

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
this class and updated as each sample passes through it. To view the results of the evaluation, call EvalReader.print().
Note that results from this Node may differ from that when calling Reader.eval()
since that is a closed domain evaluation. Have a look at our evaluation tutorial for more info about
open vs closed domain eval (https://haystack.deepset.ai/docs/latest/tutorial5md).

<a name="eval.EvalReader.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(skip_incorrect_retrieval: bool = True, open_domain: bool = True, debug: bool = False)
```

**Arguments**:

- `skip_incorrect_retrieval`: When set to True, this eval will ignore the cases where the retriever returned no correct documents
- `open_domain`: When True, extracted answers are evaluated purely on string similarity rather than the position of the extracted answer
- `debug`: When True, a record of each sample and its evaluation will be stored in EvalReader.log

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

