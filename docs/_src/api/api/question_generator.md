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

<a id="question_generator.QuestionGenerator.__init__"></a>

#### QuestionGenerator.\_\_init\_\_

```python
def __init__(model_name_or_path: str = "valhalla/t5-base-e2e-qg",
             model_version: Optional[str] = None,
             num_beams: int = 4,
             max_length: int = 256,
             no_repeat_ngram_size: int = 3,
             length_penalty: float = 1.5,
             early_stopping: bool = True,
             split_length: int = 50,
             split_overlap: int = 10,
             use_gpu: bool = True,
             prompt: str = "generate questions:",
             num_queries_per_doc: int = 1,
             sep_token: str = "<sep>",
             batch_size: int = 16,
             progress_bar: bool = True,
             use_auth_token: Optional[Union[str, bool]] = None,
             devices: Optional[List[Union[str, torch.device]]] = None)
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
- `batch_size`: Number of documents to process at a time.
- `progress_bar`: Whether to show a tqdm progress bar or not.
- `use_auth_token`: The API token used to download private models from Huggingface.
If this parameter is set to `True`, then the token generated when running
`transformers-cli login` (stored in ~/.huggingface) will be used.
Additional information can be found here
https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained
- `devices`: List of torch devices (e.g. cuda, cpu, mps) to limit inference to specific devices.
A list containing torch device objects and/or strings is supported (For example
[torch.device('cuda:0'), "mps", "cuda:1"]). When specifying `use_gpu=False` the devices
parameter is not used and a single cpu device is used for inference.

<a id="question_generator.QuestionGenerator.generate_batch"></a>

#### QuestionGenerator.generate\_batch

```python
def generate_batch(
    texts: Union[List[str], List[List[str]]],
    batch_size: Optional[int] = None
) -> Union[List[List[str]], List[List[List[str]]]]
```

Generates questions for a list of strings or a list of lists of strings.

**Arguments**:

- `texts`: List of str or list of list of str.
- `batch_size`: Number of texts to process at a time.
