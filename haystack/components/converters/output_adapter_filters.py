import json
from typing import List, Dict, Any

from haystack.dataclasses import ChatMessage, ChatRole


def change_role(messages: List[ChatMessage], role: ChatRole) -> List[ChatMessage]:
    messages[-1].role = role
    return messages


def prepare_fc_params(openai_functions_schema: Dict[str, Any]) -> Dict[str, Any]:
    if openai_functions_schema:
        return {
            "tools": [{"type": "function", "function": openai_functions_schema}],
            "tool_choice": {"type": "function", "function": {"name": openai_functions_schema["name"]}},
        }
    else:
        return {}


def tojson(string_like_json: Any) -> Dict[str, Any]:
    return json.loads(str(string_like_json))


ALL_FILTERS = {"change_role": change_role, "prepare_fc_params": prepare_fc_params, "tojson": tojson}
