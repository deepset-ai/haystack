# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union, Literal

from haystack import component, logging
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


@component
class TokenizerAwarePromptBuilder(PromptBuilder):
    """
    A PromptBuilder that is aware of the number of tokens in the prompt and can truncate the documents to fit the model's context window.
    This component is useful for RAG pipelines where the number of documents can vary and cause the prompt to exceed the model's context window.
    """

    def __init__(
        self,
        template: str,
        tokenizer,
        max_length: int = 512,
        required_variables: Optional[Union[List[str], Literal["*"]]] = None,
        variables: Optional[List[str]] = None,
    ):
        """
        :param template: A Jinja2 template for the prompt.
        :param tokenizer: A tokenizer that is used to count the number of tokens in the prompt.
        :param max_length: The maximum number of tokens allowed in the prompt.
        """
        self._template_string = template
        self._variables = variables
        self._required_variables = required_variables
        self.required_variables = required_variables or []

        try:
            from jinja2.sandbox import SandboxedEnvironment
            from haystack.utils import Jinja2TimeExtension
            self._env = SandboxedEnvironment(extensions=[Jinja2TimeExtension])
        except ImportError:
            self._env = SandboxedEnvironment()

        self.template = self._env.from_string(template)

        if not variables:
            from jinja2 import meta
            ast = self._env.parse(template)
            template_variables = meta.find_undeclared_variables(ast)
            variables = list(template_variables)
        self.variables = variables or []

        from haystack import component
        from typing import Any
        for var in self.variables:
            if self.required_variables == "*" or var in self.required_variables:
                component.set_input_type(self, var, Any)
            else:
                component.set_input_type(self, var, Any, "")

        self.tokenizer = tokenizer
        self.max_length = max_length

    @component.output_types(prompt=str)
    def run(self, template: Optional[str] = None, template_variables: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Renders the prompt template with the provided variables, making sure that the prompt does not exceed the max_length.
        """
        kwargs = kwargs or {}
        template_variables = template_variables or {}
        template_variables_combined = {**kwargs, **template_variables}
        self._validate_variables(set(template_variables_combined.keys()))

        compiled_template = self.template
        if template is not None:
            compiled_template = self._env.from_string(template)

        if "documents" in template_variables_combined:
            documents = template_variables_combined.pop("documents")
            
            # render the template without the documents to calculate the number of tokens of the boilerplate
            prompt_without_docs = compiled_template.render({**template_variables_combined})
            tokens_without_docs = self.tokenizer.encode(prompt_without_docs)
            
            # calculate the remaining tokens for the documents
            remaining_tokens = self.max_length - len(tokens_without_docs)
            
            # truncate the documents to fit the remaining tokens
            truncated_docs = []
            for doc in documents:
                doc_tokens = self.tokenizer.encode(doc.content)
                if remaining_tokens - len(doc_tokens) >= 0:
                    truncated_docs.append(doc)
                    remaining_tokens -= len(doc_tokens)
                else:
                    # if the document is too long, we truncate it
                    truncated_content = self.tokenizer.decode(doc_tokens[:remaining_tokens])
                    truncated_docs.append(Document(content=truncated_content, meta=doc.meta))
                    break # stop adding documents
            
            template_variables_combined["documents"] = truncated_docs

        result = compiled_template.render(template_variables_combined)
        return {"prompt": result}