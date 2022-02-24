<a id="question_generator"></a>

# Module question\_generator

<a id="question_generator.QuestionGenerator"></a>

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

