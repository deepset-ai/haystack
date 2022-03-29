<!---
title: "Question Generator"
metaTitle: "Question Generator"
metaDescription: ""
slug: "/docs/question_generator"
date: "2020-11-05"
id: "questiongeneratormd"
--->

# Question Generator

<div class="recommendation">

**Running examples**

Have a look at our [tutorial notebook](/docs/latest/tutorial13md))) if you'd like to start trying out Question Generation straight away!

</div>

The Question Generation module is used to generate SQuAD style questions on a given document.

This module is useful when it comes to labelling in a new domain. It can be used to generate questions quickly for an
annotator to answer. If used in conjunction with a trained Reader model, you can automatically generate question answer
pairs. High impact annotations can then be created if a human annotator looks over these pairs and corrects the incorrect predictions.

Question generation is also a good way to make large documents more navigable. Generated questions can 
quickly give the user a sense of what information is contained within the document, thus acting as a kind of summarization.

To initialize a question generator, simply call:

```python
from haystack.question_generator import QuestionGenerator

question_generator = QuestionGenerator()
```

This loads the [`valhalla/t5-base-e2e-qg`](https://huggingface.co/valhalla/t5-base-e2e-qg) model by default which is a T5 model trained on SQuAD for question generation.

To run the node in isolation, simply use the `generate()` method:

```python
result = question_generator.generate(text="Nirvana was an American rock band formed in Aberdeen, Washington in 1987.")
```

Otherwise, the node can be used in a pipeline where its `run()` method will called.
