- Start Date: 2022-12-04
- Proposal PR: https://github.com/deepset-ai/haystack/pull/3665
- Github Issue: https://github.com/deepset-ai/haystack/issues/3306
- Example Colab notebook: https://colab.research.google.com/drive/1bOIxb8cnpTrpMtTSBArY9FJlL59Ar4K_?usp=sharing

 # Summary

- The PromptNode class is the central abstraction in Haystack's large language model (LLM) support.
  Such a component supports multiple NLP tasks out-of-the-box. PromptNode allows users to
  perform multiple tasks, such as summarization, question answering, question generation etc. using a single,
  unified model within the Haystack framework.


- One of the benefits of PromptNode is that it allows users to define and add additional tasks
  that are supported by the model. This can enable users to extend the capabilities of the model and
  to use it for a wider range of NLP tasks within the Haystack ecosystem.


  # Basic example

  PromptNode is instantiated with the underlying LLM model and prompted by using natural language:

  ``` python
	  from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base")
      pn("What is the capital of Germany?")

      ----------------------------------------------------------------------------
      ['berlin']
  ```

  # Motivation


- The use of large language models (LLMs) has become increasingly popular in natural language
  processing (NLP) due to their ability to capture complex and nuanced patterns in language.
  PromptNode allows users to leverage the power of LLMs in the Haystack ecosystem, and
  to perform multiple NLP tasks using a single, unified model. This provides a flexible and efficient
  tool for NLP in Haystack, and can enable users to improve the performance and reliability of their applications.


- Modern LLM support hundreds if not thousands of tasks. Aside from PromptNode we'll define prompt templates for
  dozen or so most popular NLP tasks and allow users to register prompt templates for additional tasks. The
  extensible and modular approach would allow users to extend the capabilities of the model and to use
  it for a wider range of NLP tasks within the Haystack ecosystem. Prompt engineers would define templates
  for each NLP task and register them with the PromptNode. The burden of defining the best templates for each task
  would be on the prompt engineers and not on the users.


- The use of templates to define NLP tasks can make it easier for users to use PromptNode, as
  they do not need to know the details of how the model works or how to define tasks for it. This can
  reduce the learning curve and make it easier for users to get started with PromptNode and
  to leverage the power of LLMs in Haystack.


- The extensible and modular approach of PromptNode allows users to easily add support for
  additional templates, even on-the-fly, which can enable them to extend the capabilities of the model and to use it for
  a wider range of NLP tasks. This can provide users with more flexibility and control over the model,
  and can enable them to tailor it to their specific needs and applications.

  # Detailed design

- The PromptNode class is the most important abstraction in Haystack's large language model (LLM) support.
  In addition to PromptNode class, we'll also define a set of prompt templates for the most popular NLP tasks.


- NLP prompt templates will be represented by `PromptTemplate` class.

  ``` python
      class PromptTemplate(BaseTemplate):

          name: str
          prompt_text: str
          input_variables: List[str]

  ````

  PromptNode would, out-of-the-box, support 10-20 default NLP tasks defined by PromptTemplate instances. However, it would
  allow registering additional templates with PromptNode.


- The prompt templates for default tasks (question-answering,question-generation, summarization etc.) could be examined by the user
  using `get_prompt_templates_names` class method of the PromptNode. For example:

  ``` python
	  from haystack.nodes.llm import PromptNode
      PromptNode.get_prompt_templates_names()

      ----------------------------------------------------------------------------
      ['question-answering',
       'question-generation',
       'conditioned-question-generation',
       'summarization',
       'question-answering-check']
  ```


- PromptNode supports natural language prompting (using `prompt` method) by specifying prompt template method parameter. For example:

  ``` python
      from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base")
	  pn.prompt("question-generation", documents=["Berlin is the capital of Germany."])

      ----------------------------------------------------------------------------
      ['What is the capital of Germany?']
  ```

- PromptNode supports selecting a particular default template for a certain task (e.g. question-generation) and then subsequently
  using the selected template until user changes the current template. For example:

  ``` python
	  qa = pn.use_prompt_template("deepset/question-generation-v2")
      qa(documents=["Berlin is the capital of Germany."])

      ----------------------------------------------------------------------------
      ['What is the capital of Germany?']
  ```

- The addition of new prompt templates is supported by the `add_prompt_template` method. For example:

  ``` python
      from haystack.nodes.llm import PromptNode
      PromptNode.add_prompt_template(PromptTemplate(name="sentiment-analysis",
                              prompt_text="Please give a sentiment for this context. Answer with positive, "
                              "negative or neutral. Context: $documents; Answer:",
                              input_variables=["documents"]))
      PromptNode.get_prompt_templates_names()

      ----------------------------------------------------------------------------
      ['question-answering',
       'question-generation',
       'conditioned-question-generation',
       'summarization',
       'question-answering-check',
       'sentiment-analysis']
  ```

- Users can inspect registered prompt templates with two class methods: `get_prompt_templates_names` and `get_prompt_templates`. The first
  method, as we have seen, simply lists the names of the supported templates while the second method returns the list of `PromptTemplate`
  instances, in readable format, allowing users to inspect the actual prompt template used and the templates input parameters.

  ``` python
      from haystack.nodes.llm import PromptNode
      PromptNode.get_prompt_templates()

      ----------------------------------------------------------------------------
      [PromptTemplate(name="sentiment-analysis",
                              prompt_text="Please give a sentiment for this context. Answer with positive, "
                              "negative or neutral. Context: $documents; Answer:",
                              input_variables=["documents"], ...]
  ```



- However, aside from existing templates, users should also be able to use "on-the-fly" templates without registering them first. For example:

  ``` python
      from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base")
      prompt_template = PromptTemplate(name="sentiment-analysis",
                          prompt_text="Please give a sentiment for this context. "
                          "Answer with positive, negative or neutral. Context: $documents; Answer:",
                          input_variables=["documents"])
      pn.prompt(prompt_template, documents=["I really enjoyed the recent movie."])

      ----------------------------------------------------------------------------
      ['positive']
  ```
  This, "on-the-fly" approach might be handy if users want to simply try stuff out


- Therefore, the most central API method of the PromptNode class would be the `prompt` method with the following signature:
  ``` python
    def prompt(self, prompt_template: Union[str, PromptTemplate] = None, *args, **kwargs) -> List[str]:
  ```


- PromptNode class `__init__` constructor, aside from the `model_name_or_path` parameter would also have a
  `prompt_template` parameter which would serve as the current and default template of the PromptNode.

- ``` python
    def __init__(self, model_name_or_path: str = "google/flan-t5-base", prompt_template: Union[str, PromptTemplate] = None):
  ```

  If the `prompt_template` is not specified in the `PromptNode` init method then user is required to specify the
  template in the prompt method:

  ``` python
      from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base")
      pn.prompt("question-generation", documents=["Berlin is the capital of Germany."])

      ----------------------------------------------------------------------------
      ['What is the capital of Germany?']
  ```

  Otherwise, when the `PromptNode` is initialized with a prompt template user can invoke the `PromptNode` directly

  ``` python
      from haystack.nodes.llm import PromptNode
      pn = PromptNode(model_name_or_path="google/flan-t5-base", prompt_template="question-generation")
      pn(documents=["Berlin is the capital of Germany."])

      ----------------------------------------------------------------------------
      ['What is the capital of Germany?']
  ```

- Template parameters verification

  All template input parameters will be verified to match the template definition and the corresponding runtime
  parameters for the input variables will be checked for type and value. For example:

  ``` python
      from haystack.nodes.llm import PromptNode
      on = PromptNode(model_name_or_path="google/flan-t5-base")
      on.prompt("question-generation", some_unknown_param=["Berlin is the capital of Germany."])

      ----------------------------------------------------------------------------
      ValueError                                Traceback (most recent call last)
      <ipython-input-16-369cca52e960> in <module>
            1 # tasks parameters are checked
      ----> 2 sa(some_param=[Document("Berlin is the capital of Germany.")])

      2 frames
      /usr/local/lib/python3.8/dist-packages/haystack/nodes/llm/multi_task.py in __call__(self, *args, **kwargs)
           34         if set(template_dict.keys()) != set(self.input_variables):
           35             available_params = set(list(template_dict.keys()) + list(set(kwargs.keys())))
      ---> 36             raise ValueError(f"Expected prompt params {self.input_variables} but got {list(available_params)}")
           37
           38         template_dict["prompt_template"] = self.prompt_text

      ValueError: Expected prompt params ['documents'] but got ['some_unknown_param']
  ```

- Pipelines

  Even though we can use PromptNode directly its real power lies in using pipelines and Haystack. For example, we
  can retrieve documents from the document store using the query and then inject the retrieved documents into documents
  as a parameter to the selected PromptNode template. For example:

  ``` python
      from haystack.pipelines import PromptNode
      top_k = 3
      query = "Who are the parents of Arya Stark?"
      retriever = EmbeddingRetriever(...)
      pn = PromptNode(model_name_or_path="google/flan-t5-base", prompt_template="question-answering")

      pipe = Pipeline()
      pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
      pipe.add_node(component=pn, name="prompt_node", inputs=["Retriever"])

      output = pipe.run(query=query,
                        params={"Retriever": {"top_k": top_k}},
                        questions=[query for n in range(0, top_k)],
                        #documents parameter we need for this task will be automatically populated by the retriever
                        )

      output["results"]
  ```

 - However, we are still not utilizing the full power of Haystack pipelines. What if we could use more than
   one PromptNode in the pipeline? Perhaps we could first retrieve documents from the retriever, pass it
   to first PromptNode that will generate questions from these documents, and then add a
   second PromptNode component that will answer those generated questions given the documents as the
   context. Here is how we can do exactly that:

    ``` python
        top_k = 3
        query = "Who are the parents of Arya Stark?"
        retriever = EmbeddingRetriever(...)
        model = PromptModel(model_name_or_path="google/flan-t5-small")

        qg = PromptNode(prompt_template="question-generation", prompt_model=model, output_variable="questions")
        qa = PromptNode(prompt_template="question-answering", prompt_model=model)

        pipe = Pipeline()
        pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
        pipe.add_node(component=qg, name="qg", inputs=["Retriever"])
        pipe.add_node(component=qa, name="qa", inputs=["qg"])

        result = pipe.run(query=query)

        print(result["results"])

    ```

    The above example shows how we can use PromptNode in a pipeline and how we can bind the results of one
    PromptNode to the input of another PromptNode. The `output_variable` parameter used in the constructor of the
    question answering node, and assigned value `questions` indicates that variable `questions` could be resolved by any
    downstream pipeline node. In this particular case, the output of the question generation node will be passed to downstream
    nodes, and answering node will use that `questions` variable to inject its stored value into the `questions` parameter of
    the question answering template.

    A careful reader of this proposal will also notice that we are using the same PromptModel instance for both PromptNodes.
    This is done mainly for reuse as the PromptModel instance could be a locally run LLM and we don't want to load it
    multiple times.

    As LLMs are very resource intensive we can also envision a scenario where we would like to use a remote LLM service.
    In such cases we can use multiple instances of a PromptNode in a pipeline directly thus bypassing PromptModel altogether.


 - Pipeline YAML config file

   Let's recreate the above pipeline using a YAML config file and a declarative way of defining a pipeline.

    ```yaml

      components:

      # can go in pipeline
      - name: prompt_node
        params:
          prompt_template: template
          model_name_or_path: model
          output_variable: "questions"
        type: PromptNode

      # can go in pipeline
      - name: prompt_node_2
        params:
          prompt_template: "question-answering"
          model_name_or_path: deepset/model-name
        type: PromptNode

      # not in pipeline - only needed if you're reusing the model across multiple PromptNode in a pipeline
      # and hidden from users in the Python beginner world
      - name: model
        params:
          model_name_or_path: google/flan-t5-xl
        type: PromptModel

      # not in pipeline
      - name: template
        params:
          name: "question-generation-v2"
          prompt_text: "Given the following $documents, please generate a question. Question:"
          input_variables: documents
        type: PromptTemplate

      pipelines:
        - name: question-generation-answering-pipeline
          nodes:
            - name: EmbeddingRetriever
              inputs: [Query]
            - name: prompt_node
              inputs: [EmbeddingRetriever]
            - name: prompt_node_2
              inputs: [prompt_node]
    ```
    First of all, notice how we reuse the resource heavy PromptModel instance across multiple PromptNode instances. And
    although we could have used already registered `question-generation` prompt template, we decided to define a new one
    called `question-generation-v2` and as such set it as the default template for the first PromptNode. We also defined
    the output of the first PromptNode as `questions` and used that variable in the second PromptNode.

    In conclusion, we can see that the YAML config file is a mirror image of the previous code centric pipeline
    example and also a very powerful way of defining a pipeline.



  - Default tasks/prompts to be added to PromptNode

    [Muffin]:
      - Summarization
      - Natural Language Inference
      - Multiple-Choice QA
      - Translation
      - Sentiment Analysis
      - Extractive QA
      - Structured Data to Text
      - Coreference Resolution
      - Code Repair
      - Code Error Generation
      - Dialogue Context Generation
      - Closed-Book QA
      - Next Sentence Prediction
      - Paraphrasing Identification
      - Conversational Question Answering
      - Topic Classification
      - Mathematical QA
      - Dialog Next Turn Prediction
      - Grammatical Acceptability
      - Punctuation fixing

    [T0-SF]:
      - Adversarial QA
      - Question Generation
      - Commonsense Reasoning
      - Title Generation
      - Dialogue Turn Prediction
      - Predict Span Indices
      - Context Generation

    [NIV2]:
      - Program Execution
      - Text Matching
      - Toxic Language Detection
      - Cause Effect Classification
      - Information Extraction
      - Textual Entailment
      - Wrong Candidate Generation
      - Named Entity Recognition
      - Commonsense Classification
      - Fill-in-the-blank
      - Text Completion
      - Sentence Composition
      - Question Understanding

    [CoT Reasoning]:
      - Explanation Generation
      - Generate Question And Answer
      - Grade School Math Word Problems
      - Algebraic Question Answering
      - Common Sense Reasoning Over Entities
      - Common Sense Reasoning For QA
      - Passage Based Question Answering
      - Sense-Making And Explanation

  # Drawbacks
- One potential drawback of PromptNode is that it may require a significant amount of computational resources
  to use. This may limit its use in applications or environments where there are constraints on the available hardware
  or software resources.


- Due to current pipeline design limitations PromptTemplate has to be a subclass of BaseComponent. This might slightly
  confuse some users who are already familiar with Haystack components. We will mitigate this issue in subsequent releases
  as we refactor the pipeline design. All in all, PromptTemplate will be a thin class with minimal inheritance signature from some base class.



  # Alternatives

- One alternative to PromptNode is to continue to use separate models for each NLP task in Haystack. This
  can enable users to tailor the model to the specific requirements of each task, and to potentially improve the
  performance of the model for that task by additional fine-tuning or model adaptation via GPL. However, using separate
  models may require these complex and computationally intensive training and deployment processes, and may not be as
  efficient or flexible as using a single, unified model.

  # Adoption strategy
- This is not a breaking change proposal and we should implement it immediately.

  # How do we teach this?
- This change would require change in documentation.
- We can provide examples of how to use PromptNode in Haystack pipelines via tutorials.
- Docs and tutorials need to be updated
