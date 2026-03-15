from typing import Protocol, runtime_checkable, TYPE_CHECKING, Any, List, Optional, Dict
from dataclasses import dataclass

# Sirf wahi types rakho jo bohot zaroori hain
if TYPE_CHECKING:
    from haystack.dataclasses.chat_message import ChatMessage
    from haystack.tools.tool import Tool 

@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    modified_args: Optional[Dict[str, Any]] = None

@runtime_checkable
class GuardrailProvider(Protocol):
    def evaluate_tool_call(
        self,
        tool: "Tool", 
        tool_call_args: Dict[str, Any],
        messages: List["ChatMessage"],
        agent_state: Optional[Any] = None, # Agent type hatake 'Any' kar diya
    ) -> GuardrailResult:
        ...