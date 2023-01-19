<a name="question_generator"></a>
# Module question\_generator

<a name="question_generator.QuestionGenerator"></a>
## QuestionGenerator

```python
class QuestionGenerator(BaseComponent)
```

The Question Generator takes only a document as input and outputs questions that it thinks can be
answered by this document. In our current implementation, input texts are split into chunks of 50 words
with a 10 word overlap. This is because the default model `valhalla/t5-base-e2e-qg` seems to generate only
about 3 questions per passage regardless of length. Our approach prioritizes the creation of more questions
over processing efficiency (T5 is able to digest much more than 50 words at once). The returned questions
generally come in an order dictated by the order of their answers i.e. early questions in the list generally
come from earlier in the document.

<a name="question_generator.QuestionGenerator.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model_name_or_path="valhalla/t5-base-e2e-qg", model_version=None, num_beams=4, max_length=256, no_repeat_ngram_size=3, length_penalty=1.5, early_stopping=True, split_length=50, split_overlap=10, use_gpu=True, prompt="generate questions:")
```

Uses the valhalla/t5-base-e2e-qg model by default. This class supports any question generation model that is
implemented as a Seq2SeqLM in HuggingFace Transformers. Note that this style of question generation (where the only input
is a document) is sometimes referred to as end-to-end question generation. Answer-supervised question
generation is not currently supported.

**Arguments**:

- `model_name_or_path`: Directory of a saved model or the name of a public model e.g. "valhalla/t5-base-e2e-qg".
                           See https://huggingface.co/models for full list of available models.
- `model_version`: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
- `use_gpu`: Whether to use GPU or the CPU. Falls back on CPU if no GPU is available.
