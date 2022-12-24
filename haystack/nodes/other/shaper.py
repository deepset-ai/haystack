import copy
import inspect
from collections import deque
from typing import Optional, Union, List, Dict, Any, Tuple, Callable
from types import MappingProxyType

from haystack.nodes.base import BaseComponent
from haystack.schema import Document, MultiLabel


class Shaper(BaseComponent):
    outgoing_edges = 1

    def __init__(self, inputs: Dict[str, Any]):
        super().__init__()
        self.inputs = inputs

        concat: Callable[[List[str]], str] = lambda docs, delimiter=" ": delimiter.join(docs)  # type: ignore
        concat_docs: Callable[[List[Document]], str] = lambda docs, delimiter=" ", num_tokens=-1: delimiter.join(  # type: ignore
            [d.content for d in docs]
        )[
            :num_tokens
        ]
        tmp_registry: Dict[str, Callable] = {
            "len": len,
            "expand": (lambda expand_target, size: [expand_target] * size),
            "expand:size": lambda documents: len(documents),  # pylint: disable=unnecessary-lambda
            "concat": concat,
            "concat_docs": concat_docs,
        }
        # see https://adamj.eu/tech/2022/01/05/how-to-make-immutable-dict-in-python/
        self.function_registry = MappingProxyType(tmp_registry)

    def invoke(self, parent, node: Dict[str, Any], invocation_context: Dict[str, Any]):
        #
        # If the node is a function invocation, invoke it with the provided arguments
        # For example, in the YAML snippet below, the function len is invoked with the argument documents,
        # the result is stored in the variable size, and then used in the next function invocation expand
        # ---
        #     query:
        #       func: expand
        #       output: questions
        #       params:
        #         expand_target: query
        #         size:
        #           func: len
        #           params:
        #             - documents
        #
        if "func" in node:
            self.invoke_function(parent, node, invocation_context)

        elif "output" in node and parent in invocation_context:
            # Assign the value of the input variable from the invocation context to the output variable, store the
            # output variable in the invocation context, and pass it to the next pipeline node
            # ____
            # - name: shaper
            #  params:
            #    inputs:
            #        query:
            #          output: questions
            invocation_context[node["output"]] = copy.deepcopy(invocation_context[parent])
        else:
            params_is_parent = "params" == parent
            root_node = parent is None and all(key for key in node.keys() if key in invocation_context)
            # if not a function, valid output, or params definition then this must be a root node in the inputs
            # otherwise, something is wrong, raise an exception
            if not params_is_parent and not root_node:
                raise ValueError(
                    f"Invalid input configuration. The input configuration should be a function invocation, or "
                    f"an assignment. The input configuration is {node}"
                )

    def invoke_function(self, parent, node, invocation_context: Dict[str, Any]):
        params = node.get("params", [])
        output = node.get("output", parent)
        # use case when the function is invoked with positional arguments
        if isinstance(params, list):
            # resolve the argument values from the invocation context
            args = [invocation_context[param] for param in params if param in invocation_context]

            if len(args) != len(params):
                raise ValueError(
                    f"Invalid function arguments. The function {node['func']} "
                    f"specifies in its YAML definition {len(params)} arguments yet {len(args)} arguments "
                    f"were resolved"
                )
            # push the output variable name to the invocation stack
            if output not in invocation_context:
                invocation_context["invocation_stack"].append(output)
            # and finally invoke the function with the resolved arguments
            invocation_result = self.invoke_function_with_args(parent, node, invocation_context, *args)

        # use case when the function is invoked with keyword arguments
        elif isinstance(params, dict):
            kwargs = {}
            for key, value in params.items():
                resolved_value = invocation_context.get(key)
                if resolved_value:
                    # if the key is a variable available in the invocation context, resolve it
                    kwargs[key] = resolved_value
                else:
                    # attempt to find the value from the invocation context and assign to the keyword argument
                    kwargs[key] = value if value not in invocation_context else invocation_context[value]
                if key in invocation_context and invocation_context["invocation_stack"][-1] == key:
                    invocation_context["invocation_stack"].pop()
                    invocation_context.pop(key)
            invocation_result = self.invoke_function_with_kwargs(parent, node, invocation_context, **kwargs)
        else:
            raise ValueError(f"Invalid YAML definition {node}")

        # finally store the result in the invocation context
        invocation_context[output] = invocation_result

    def invoke_function_with_args(self, parent, node: Dict[str, Any], invocation_context: Dict[str, Any], *args):
        func = self.resolve_function(node)
        try:
            return func(*args)
        except TypeError as e:
            signature = inspect.getfullargspec(func)
            kwargs = {key: value for key, value in zip(signature.args, args)}
            if len(kwargs) < len(signature.args):
                # Handles the case when the function is invoked with positional arguments or none at all
                # In this case we need to resolve the missing arguments from the invocation context guided by the
                # function signature and the functions registered in the function registry
                # components:
                # - name: shaper
                #  params:
                #    inputs:
                #        query:
                #          func: expand
                #          output: questions
                for key in signature.args:
                    key_func = self.function_registry.get(
                        # if the function is registered with a suffix, use it to resolve the value
                        # otherwise, resolve the value of the parent from invocation context
                        # i.e. resolve query in the example above
                        node["func"] + ":" + key,
                        lambda: invocation_context.get(parent),
                    )
                    func_kwargs = {}
                    for arg in inspect.getfullargspec(key_func).args:
                        func_kwargs[arg] = invocation_context.get(arg, None)
                    kwargs[key] = key_func(**func_kwargs)
            if len(kwargs) == len(signature.args):
                return self.invoke_function_with_kwargs(parent, node, invocation_context, **kwargs)
            else:
                raise TypeError(f"Error invoking function {node['func']}, {e}")

    def invoke_function_with_kwargs(self, parent, node: Dict[str, Any], invocation_context: Dict[str, Any], **kwargs):
        func = self.resolve_function(node)
        try:
            return func(**kwargs)
        except TypeError as e:
            signature = inspect.getfullargspec(func)
            if len(kwargs) < len(signature.args):
                # Handles the case when the function is invoked with keyword arguments but some are missing.
                # In this case we need to resolve the missing keyword arguments from the invocation context guided
                # by the function signature and the functions registered in the function registry
                # components:
                # - name: shaper
                #   params:
                #     inputs:
                #         query:
                #           func: expand
                #           params:
                #             expand_target: query
                #           output: questions
                missing_keys = signature.args - kwargs.keys()
                for key in missing_keys:
                    # if the function is registered with a suffix, use it to resolve the kwargs value
                    key_func = self.resolve_function({"func": node["func"] + ":" + key})
                    func_kwargs = {}
                    for arg in inspect.getfullargspec(key_func).args:
                        func_kwargs[arg] = invocation_context.get(arg, None)
                    kwargs[key] = key_func(**func_kwargs)

            if len(kwargs) == len(signature.args) and list(kwargs.keys()) == signature.args:
                # if all the arguments are resolved, invoke the function
                return self.invoke_function_with_kwargs(parent, node, invocation_context, **kwargs)
            else:
                raise TypeError(
                    f"Error invoking function {node['func']}, arguments are {signature.args} "
                    f"but the provided arguments are {list(kwargs.keys())}. Error: {e}"
                )

    def resolve_function(self, node: Dict[str, Any]):
        function_name = node["func"]
        func = self.function_registry.get(function_name)
        if not func:
            raise ValueError(f"{function_name} not supported by Shaper. Check the function name and try again.")
        return func

    def traverse(self, parent, node, invocation_context, node_visitor):
        # traverse the inputs tree in post-order
        if isinstance(node, dict):
            for key, value in node.items():
                self.traverse(key, value, invocation_context, node_visitor)
        elif isinstance(node, list):
            for item in node:
                self.traverse(node, item, invocation_context, node_visitor)

        if isinstance(node, Dict):
            node_visitor(parent, node, invocation_context)

    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
    ) -> Tuple[Dict, str]:

        invocation_context: Dict[str, Any] = {
            "invocation_stack": deque(),
            **{
                k: v
                for k, v in {"query": query, "file_paths": file_paths, "labels": labels, "documents": documents}.items()
                if v is not None
            },
        }

        # check if the inputs are valid (i.e. in invocation context)
        invalid_vars = [key for key in self.inputs.keys() if key not in invocation_context]
        if invalid_vars:
            raise ValueError(
                f"The following variables were not found in the invocation context {invalid_vars}."
                f"Please check the inputs definition and try again."
            )

        # traverse the inputs YAML subtree of the Shaper component declaration, invoke the specified
        # functions, and store the results in the invocation context
        self.traverse(None, self.inputs, invocation_context, self.invoke)
        invocation_context.pop("invocation_stack")

        meta = meta or {}
        meta["invocation_context"] = invocation_context
        response = {
            "query": invocation_context.pop("query", query),
            "documents": invocation_context.pop("documents", documents),
            "file_paths": invocation_context.pop("file_paths", file_paths),
            "labels": invocation_context.pop("labels", labels),
            "meta": {**meta},
        }, "output_1"
        return response

    def run_batch(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        multi_query = isinstance(queries, list) and len(queries) > 0 and isinstance(queries[0], str)
        single_query = isinstance(queries, str)
        multi_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list)
        single_docs_list = isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], Document)

        results = {"queries": queries, "documents": documents, "meta": meta}
        if single_query:
            if single_docs_list:
                response, node_id = self.run(query=queries, documents=documents, meta=meta)  # type: ignore
                self.result_update_helper(results, response)
                results.update(response)
            elif multi_docs_list:
                for doc in documents:  # type: ignore
                    response, node_id = self.run(query=queries, documents=doc, meta=meta)  # type: ignore
                    self.result_update_helper(results, response)
        elif multi_query:
            for query, docs in zip(queries, documents):  # type:ignore
                response, node_id = self.run(query=query, documents=docs, meta=meta)  # type: ignore
                self.result_update_helper(results, response)
        return results, "output_1"

    @staticmethod
    def result_update_helper(parent_dict, response):
        for k, v in response.items():
            tmp_list = parent_dict.get(k, [])
            if isinstance(tmp_list, list):
                tmp_list.append(v)
            else:
                new_list = [v]
                parent_dict[k] = new_list
