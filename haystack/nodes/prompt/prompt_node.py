from collections import defaultdict
import copy
import logging
import re
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import yaml

from haystack.nodes.base import BaseComponent

from haystack.schema import Document, MultiLabel
from haystack.telemetry import send_event
from haystack.nodes.prompt.shapers import BaseOutputParser
from haystack.nodes.prompt.prompt_model import PromptModel
from haystack.nodes.prompt.prompt_template import PromptTemplate, get_predefined_prompt_templates

logger = logging.getLogger(__name__)


class PromptNode(BaseComponent):
    """
    The PromptNode class is the central abstraction in Haystack's large language model (LLM) support. PromptNode
    supports multiple NLP tasks out of the box. You can use it to perform tasks such as
    summarization, question answering, question generation, and more, using a single, unified model within the Haystack
    framework.

    One of the benefits of PromptNode is that you can use it to define and add additional prompt templates
     the model supports. Defining additional prompt templates makes it possible to extend the model's capabilities
    and use it for a broader range of NLP tasks in Haystack. Prompt engineers define templates
    for each NLP task and register them with PromptNode. The burden of defining templates for each task rests on
    the prompt engineers, not the users.

    Using an instance of the PromptModel class, you can create multiple PromptNodes that share the same model, saving
    the memory and time required to load the model multiple times.

    PromptNode also supports multiple model invocation layers:
    - Hugging Face transformers (all text2text-generation models)
    - OpenAI InstructGPT models
    - Azure OpenAI InstructGPT models

    But you're not limited to the models listed above, as you can register
    additional custom model invocation layers.

    We recommend using LLMs fine-tuned on a collection of datasets phrased as instructions, otherwise we find that the
    LLM does not "follow" prompt instructions well. The list of instruction-following models increases every month,
    and the current list includes: Flan, OpenAI InstructGPT, opt-iml, bloomz, and mt0 models.

    For more details, see [PromptNode](https://docs.haystack.deepset.ai/docs/prompt_node).
    """

    outgoing_edges: int = 1

    def __init__(
        self,
        model_name_or_path: Union[str, PromptModel] = "google/flan-t5-base",
        default_prompt_template: Optional[Union[str, PromptTemplate]] = None,
        output_variable: Optional[str] = None,
        max_length: Optional[int] = 100,
        api_key: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        use_gpu: Optional[bool] = None,
        devices: Optional[List[Union[str, torch.device]]] = None,
        stop_words: Optional[List[str]] = None,
        top_k: int = 1,
        debug: Optional[bool] = False,
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Creates a PromptNode instance.

        :param model_name_or_path: The name of the model to use or an instance of the PromptModel.
        :param default_prompt_template: The default prompt template to use for the model.
        :param output_variable: The name of the output variable in which you want to store the inference results.
            If not set, PromptNode uses PromptTemplate's output_variable. If PromptTemplate's output_variable is not set, the default name is `results`.
        :param max_length: The maximum number of tokens the generated text output can have.
        :param api_key: The API key to use for the model.
        :param use_auth_token: The authentication token to use for the model.
        :param use_gpu: Whether to use GPU or not.
        :param devices: The devices to use for the model.
        :param top_k: The number of independently generated texts to return per prompt. For example, if you set top_k=3, the model will generate three answers to the query.
        :param stop_words: Stops text generation if any of the stop words is generated.
        :param model_kwargs: Additional keyword arguments passed when loading the model specified in `model_name_or_path`.
        :param debug: Whether to include the used prompts as debug information in the output under the key _debug.

        Note that Azure OpenAI InstructGPT models require two additional parameters: azure_base_url (the URL for the
        Azure OpenAI API endpoint, usually in the form `https://<your-endpoint>.openai.azure.com') and
        azure_deployment_name (the name of the Azure OpenAI API deployment).
        You should specify these parameters in the `model_kwargs` dictionary.

        """
        send_event(
            event_name="PromptNode",
            event_properties={
                "llm.model_name_or_path": model_name_or_path,
                "llm.default_prompt_template": default_prompt_template,
            },
        )
        super().__init__()
        self.prompt_templates: Dict[str, PromptTemplate] = {pt.name: pt for pt in get_predefined_prompt_templates()}  # type: ignore
        self.default_prompt_template: Union[str, PromptTemplate, None] = default_prompt_template
        self.output_variable: Optional[str] = output_variable
        self.model_name_or_path: Union[str, PromptModel] = model_name_or_path
        self.prompt_model: PromptModel
        self.stop_words: Optional[List[str]] = stop_words
        self.top_k: int = top_k
        self.debug = debug

        if isinstance(self.default_prompt_template, str) and not self.is_supported_template(
            self.default_prompt_template
        ):
            raise ValueError(
                f"Prompt template {self.default_prompt_template} is not supported. "
                f"Select one of: {self.get_prompt_template_names()} "
                f"or register a new prompt template first using the add_prompt_template() method."
            )

        if isinstance(model_name_or_path, str):
            self.prompt_model = PromptModel(
                model_name_or_path=model_name_or_path,
                max_length=max_length,
                api_key=api_key,
                use_auth_token=use_auth_token,
                use_gpu=use_gpu,
                devices=devices,
                model_kwargs=model_kwargs,
            )
        elif isinstance(model_name_or_path, PromptModel):
            self.prompt_model = model_name_or_path
        else:
            raise ValueError("model_name_or_path must be either a string or a PromptModel object")

    def __call__(self, *args, **kwargs) -> List[Any]:
        """
        This method is invoked when the component is called directly, for example:
        ```python
            PromptNode pn = ...
            sa = pn.set_default_prompt_template("sentiment-analysis")
            sa(documents=[Document("I am in love and I feel great!")])
        ```
        """
        if "prompt_template" in kwargs:
            prompt_template = kwargs["prompt_template"]
            kwargs.pop("prompt_template")
            return self.prompt(prompt_template, *args, **kwargs)
        else:
            return self.prompt(self.default_prompt_template, *args, **kwargs)

    def prompt(self, prompt_template: Optional[Union[str, PromptTemplate]], *args, **kwargs) -> List[Any]:
        """
        Prompts the model and represents the central API for the PromptNode. It takes a prompt template,
        a list of non-keyword and keyword arguments, and returns a list of strings - the responses from the underlying model.

        If you specify the optional prompt_template parameter, it takes precedence over the default PromptTemplate for this PromptNode.

        :param prompt_template: The name or object of the optional PromptTemplate to use.
        :return: A list of strings as model responses.
        """
        results = []
        # we pop the prompt_collector kwarg to avoid passing it to the model
        prompt_collector: List[Union[str, List[Dict[str, str]]]] = kwargs.pop("prompt_collector", [])

        # kwargs override model kwargs
        kwargs = {**self._prepare_model_kwargs(), **kwargs}
        template_to_fill = self.get_prompt_template(prompt_template)
        if template_to_fill:
            # prompt template used, yield prompts from inputs args
            for prompt in template_to_fill.fill(*args, **kwargs):
                kwargs_copy = copy.copy(kwargs)
                # and pass the prepared prompt and kwargs copy to the model
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug("Prompt being sent to LLM with prompt %s and kwargs %s", prompt, kwargs_copy)
                output = self.prompt_model.invoke(prompt, **kwargs_copy)
                results.extend(output)

            kwargs["prompts"] = prompt_collector
            results = template_to_fill.post_process(results, **kwargs)
        else:
            # straightforward prompt, no templates used
            for prompt in list(args):
                kwargs_copy = copy.copy(kwargs)
                prompt = self.prompt_model._ensure_token_limit(prompt)
                prompt_collector.append(prompt)
                logger.debug("Prompt being sent to LLM with prompt %s and kwargs %s ", prompt, kwargs_copy)
                output = self.prompt_model.invoke(prompt, **kwargs_copy)
                results.extend(output)
        return results

    def add_prompt_template(self, prompt_template: PromptTemplate) -> None:
        """
        Adds a prompt template to the list of supported prompt templates.
        :param prompt_template: The PromptTemplate object to be added.
        :return: None
        """
        if prompt_template.name in self.prompt_templates:
            raise ValueError(
                f"Prompt template {prompt_template.name} already exists. "
                f"Select a different name for this prompt template."
            )

        self.prompt_templates[prompt_template.name] = prompt_template  # type: ignore

    def remove_prompt_template(self, prompt_template: str) -> PromptTemplate:
        """
        Removes a prompt template from the list of supported prompt templates.
        :param prompt_template: Name of the prompt template to be removed.
        :return: PromptTemplate object that was removed.
        """
        if prompt_template not in self.prompt_templates:
            raise ValueError(f"Prompt template {prompt_template} does not exist")

        return self.prompt_templates.pop(prompt_template)

    def set_default_prompt_template(self, prompt_template: Union[str, PromptTemplate]) -> "PromptNode":
        """
        Sets the default prompt template for the node.
        :param prompt_template: The prompt template to be set as default.
        :return: The current PromptNode object.
        """
        if not self.is_supported_template(prompt_template):
            raise ValueError(f"{prompt_template} not supported, select one of: {self.get_prompt_template_names()}")

        self.default_prompt_template = prompt_template
        return self

    def get_prompt_templates(self) -> List[PromptTemplate]:
        """
        Returns the list of supported prompt templates.
        :return: List of supported prompt templates.
        """
        return list(self.prompt_templates.values())

    def get_prompt_template_names(self) -> List[str]:
        """
        Returns the list of supported prompt template names.
        :return: List of supported prompt template names.
        """
        return list(self.prompt_templates.keys())

    def is_supported_template(self, prompt_template: Union[str, PromptTemplate]) -> bool:
        """
        Checks if a prompt template is supported.
        :param prompt_template: The prompt template to be checked.
        :return: True if the prompt template is supported, False otherwise.
        """
        template_name = prompt_template if isinstance(prompt_template, str) else prompt_template.name
        return template_name in self.prompt_templates

    def get_prompt_template(self, prompt_template: Union[str, PromptTemplate, None] = None) -> Optional[PromptTemplate]:
        """
        Resolves a prompt template.

        :param prompt_template: The prompt template to be resolved. You can choose between the following types:
            - None: Returns the default prompt template.
            - PromptTemplate: Returns the given prompt template object.
            - str: Parses the string depending on its content:
                - prompt template name: Returns the prompt template registered with the given name.
                - prompt template yaml: Returns a prompt template specified by the given YAML.
                - prompt text: Returns a copy of the default prompt template with the given prompt text.

            :return: The prompt template object.
        """
        prompt_template = prompt_template or self.default_prompt_template
        if prompt_template is None:
            return None

        if isinstance(prompt_template, PromptTemplate):
            return prompt_template

        if isinstance(prompt_template, str) and prompt_template in self.prompt_templates:
            return self.prompt_templates[prompt_template]

        # if it's not a string or looks like a prompt template name
        if not isinstance(prompt_template, str) or re.fullmatch(r"[-a-zA-Z0-9_]+", prompt_template):
            raise ValueError(
                f"{prompt_template} not supported, select one of: {self.get_prompt_template_names()} or pass a PromptTemplate instance for prompting."
            )

        if "prompt_text:" in prompt_template:
            prompt_template_parsed = yaml.safe_load(prompt_template)
            if isinstance(prompt_template_parsed, dict):
                return PromptTemplate(**prompt_template_parsed)

        # it's a prompt_text
        prompt_text = prompt_template
        output_parser: Optional[BaseOutputParser] = None
        default_prompt_template = self.get_prompt_template()
        if default_prompt_template:
            output_parser = default_prompt_template.output_parser
        return PromptTemplate(name="custom-at-query-time", prompt_text=prompt_text, output_parser=output_parser)

    def prompt_template_params(self, prompt_template: str) -> List[str]:
        """
        Returns the list of parameters for a prompt template.
        :param prompt_template: The name of the prompt template.
        :return: The list of parameters for the prompt template.
        """
        if not self.is_supported_template(prompt_template):
            raise ValueError(f"{prompt_template} not supported, select one of: {self.get_prompt_template_names()}")

        return list(self.prompt_templates[prompt_template].prompt_params)

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        invocation_context: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, str]:
        """
        Runs the PromptNode on these input parameters. Returns the output of the prompt model.
        The parameters `query`, `file_paths`, `labels`, `documents`, and `meta` are added to the invocation context
        before invoking the prompt model. PromptNode uses these variables only if they are present as
        parameters in the PromptTemplate.

        :param query: The PromptNode usually ignores the query, unless it's used as a parameter in the
        prompt template.
        :param file_paths: The PromptNode usually ignores the file paths, unless they're used as a parameter
        in the prompt template.
        :param labels: The PromptNode usually ignores the labels, unless they're used as a parameter in the
        prompt template.
        :param documents: The documents to be used for the prompt.
        :param meta: PromptNode usually ignores meta information, unless it's used as a parameter in the
        PromptTemplate.
        :param invocation_context: The invocation context to be used for the prompt.
        :param prompt_template: The prompt template to use. You can choose between the following types:
            - None: Use the default prompt template.
            - PromptTemplate: Use the given prompt template object.
            - str: Parses the string depending on its content:
                - prompt template name: Uses the prompt template registered with the given name.
                - prompt template yaml: Uses the prompt template specified by the given YAML.
                - prompt text: Uses a copy of the default prompt template with the given prompt text.
        :param generation_kwargs: The generation_kwargs are used to customize text generation for the underlying pipeline.
        """
        # prompt_collector is an empty list, it's passed to the PromptNode that will fill it with the rendered prompts,
        # so that they can be returned by `run()` as part of the pipeline's debug output.
        prompt_collector: List[str] = []

        invocation_context = invocation_context or {}
        if query and "query" not in invocation_context.keys():
            invocation_context["query"] = query

        if file_paths and "file_paths" not in invocation_context.keys():
            invocation_context["file_paths"] = file_paths

        if labels and "labels" not in invocation_context.keys():
            invocation_context["labels"] = labels

        if documents and "documents" not in invocation_context.keys():
            invocation_context["documents"] = documents

        if meta and "meta" not in invocation_context.keys():
            invocation_context["meta"] = meta

        if "prompt_template" not in invocation_context.keys():
            invocation_context["prompt_template"] = self.get_prompt_template(prompt_template)

        if generation_kwargs:
            invocation_context.update(generation_kwargs)

        results = self(prompt_collector=prompt_collector, **invocation_context)

        prompt_template_resolved: PromptTemplate = invocation_context.pop("prompt_template")
        output_variable = self.output_variable or prompt_template_resolved.output_variable or "results"
        invocation_context[output_variable] = results
        invocation_context["prompts"] = prompt_collector
        final_result: Dict[str, Any] = {output_variable: results, "invocation_context": invocation_context}

        if self.debug:
            final_result["_debug"] = {"prompts_used": prompt_collector}

        return final_result, "output_1"

    def run_batch(  # type: ignore
        self,
        queries: Optional[List[str]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        invocation_contexts: Optional[List[Dict[str, Any]]] = None,
        prompt_templates: Optional[List[Union[str, PromptTemplate]]] = None,
    ):
        """
        Runs PromptNode in batch mode.

        - If you provide a list containing a single query (or invocation context)...
            - ... and a single list of Documents, the query is applied to each Document individually.
            - ... and a list of lists of Documents, the query is applied to each list of Documents and the results
              are aggregated per Document list.

        - If you provide a list of multiple queries (or multiple invocation contexts)...
            - ... and a single list of Documents, each query (or invocation context) is applied to each Document individually.
            - ... and a list of lists of Documents, each query (or invocation context) is applied to its corresponding list of Documents
              and the results are aggregated per query-Document pair.

        - If you provide no Documents, then each query (or invocation context) is applied directly to the PromptTemplate.

        :param queries: List of queries.
        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.
        :param invocation_contexts: List of invocation contexts.
        :param prompt_templates: The prompt templates to use. You can choose between the following types:
            - None: Use the default prompt template.
            - PromptTemplate: Use the given prompt template object.
            - str: Parses the string depending on its content:
                - prompt template name: Uses the prompt template registered with the given name.
                - prompt template yaml: Uuses the prompt template specified by the given YAML.
                - prompt text: Uses a copy of the default prompt template with the given prompt text.
        """
        inputs = PromptNode._flatten_inputs(queries, documents, invocation_contexts, prompt_templates)
        all_results: Dict[str, List] = defaultdict(list)
        for query, docs, invocation_context, prompt_template in zip(
            inputs["queries"], inputs["documents"], inputs["invocation_contexts"], inputs["prompt_templates"]
        ):
            prompt_template = self.get_prompt_template(self.default_prompt_template)
            output_variable = self.output_variable or prompt_template.output_variable or "results"
            results = self.run(
                query=query, documents=docs, invocation_context=invocation_context, prompt_template=prompt_template
            )[0]
            all_results[output_variable].append(results[output_variable])
            all_results["invocation_contexts"].append(results["invocation_context"])
            if self.debug:
                all_results["_debug"].append(results["_debug"])
        return all_results, "output_1"

    def _prepare_model_kwargs(self):
        # these are the parameters from PromptNode level
        # that are passed to the prompt model invocation layer
        return {"stop_words": self.stop_words, "top_k": self.top_k}

    @staticmethod
    def _flatten_inputs(
        queries: Optional[List[str]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        invocation_contexts: Optional[List[Dict[str, Any]]] = None,
        prompt_templates: Optional[List[Union[str, PromptTemplate]]] = None,
    ) -> Dict[str, List]:
        """Flatten and copy the queries, documents, and invocation contexts into lists of equal length.

        - If you provide a list containing a single query (or invocation context)...
            - ... and a single list of Documents, the query is applied to each Document individually.
            - ... and a list of lists of Documents, the query is applied to each list of Documents and the results
              are aggregated per Document list.

        - If you provide a list of multiple queries (or multiple invocation contexts)...
            - ... and a single list of Documents, each query (or invocation context) is applied to each Document individually.
            - ... and a list of lists of Documents, each query (or invocation context) is applied to its corresponding list of Documents
              and the results are aggregated per query-Document pair.

        - If you provide no Documents, then each query (or invocation context) is applied to the PromptTemplate.

        :param queries: List of queries.
        :param documents: Single list of Documents or list of lists of Documents in which to search for the answers.
        :param invocation_contexts: List of invocation contexts.
        """
        # Check that queries, and invocation_contexts are of the same length if provided
        input_queries: List[Any]
        input_invocation_contexts: List[Any]
        input_prompt_templates: List[Any]
        if queries is not None and invocation_contexts is not None:
            if len(queries) != len(invocation_contexts):
                raise ValueError("The input variables queries and invocation_contexts should have the same length.")
            input_queries = queries
            input_invocation_contexts = invocation_contexts
        elif queries is not None and invocation_contexts is None:
            input_queries = queries
            input_invocation_contexts = [None] * len(queries)
        elif queries is None and invocation_contexts is not None:
            input_queries = [None] * len(invocation_contexts)
            input_invocation_contexts = invocation_contexts
        else:
            input_queries = [None]
            input_invocation_contexts = [None]

        if prompt_templates is not None:
            if len(prompt_templates) != len(input_queries):
                raise ValueError("The input variables prompt_templates and queries should have the same length.")
            input_prompt_templates = prompt_templates
        else:
            input_prompt_templates = [None] * len(input_queries)

        multi_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list)
        single_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], Document)

        # Docs case 1: single list of Documents
        # -> apply each query (and invocation_contexts) to all Documents
        inputs: Dict[str, List] = defaultdict(list)
        if documents is not None:
            if single_docs_list:
                for query, invocation_context, prompt_template in zip(
                    input_queries, input_invocation_contexts, input_prompt_templates
                ):
                    for doc in documents:
                        inputs["queries"].append(query)
                        inputs["invocation_contexts"].append(invocation_context)
                        inputs["documents"].append([doc])
                        inputs["prompt_templates"].append(prompt_template)
            # Docs case 2: list of lists of Documents
            # -> apply each query (and invocation_context) to corresponding list of Documents,
            # if queries contains only one query, apply it to each list of Documents
            elif multi_docs_list:
                total_queries = input_queries.copy()
                total_invocation_contexts = input_invocation_contexts.copy()
                total_prompt_templates = input_prompt_templates.copy()
                if len(total_queries) == 1 and len(total_invocation_contexts) == 1 and len(total_prompt_templates) == 1:
                    total_queries = input_queries * len(documents)
                    total_invocation_contexts = input_invocation_contexts * len(documents)
                    total_prompt_templates = input_prompt_templates * len(documents)
                if (
                    len(total_queries) != len(documents)
                    or len(total_invocation_contexts) != len(documents)
                    or len(total_prompt_templates) != len(documents)
                ):
                    raise ValueError("Number of queries must be equal to number of provided Document lists.")
                for query, invocation_context, prompt_template, cur_docs in zip(
                    total_queries, total_invocation_contexts, total_prompt_templates, documents
                ):
                    inputs["queries"].append(query)
                    inputs["invocation_contexts"].append(invocation_context)
                    inputs["documents"].append(cur_docs)
                    inputs["prompt_templates"].append(prompt_template)
        elif queries is not None or invocation_contexts is not None or prompt_templates is not None:
            for query, invocation_context, prompt_template in zip(
                input_queries, input_invocation_contexts, input_prompt_templates
            ):
                inputs["queries"].append(query)
                inputs["invocation_contexts"].append(invocation_context)
                inputs["documents"].append([None])
                inputs["prompt_templates"].append(prompt_template)
        return inputs
