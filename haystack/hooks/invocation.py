# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.agents.state.state import State
from haystack.hooks.protocol import Hook, HookPoint
from haystack.utils.async_utils import _execute_component_async


def _run_hooks(hooks: dict[HookPoint, list[Hook]], hook_point: HookPoint, state: State) -> None:
    """
    Run every hook registered for the given hook point, in list order.

    :param hooks: Hooks keyed by hook point.
    :param hook_point: The hook point whose hooks to run; hooks registered under other hook points are skipped.
    :param state: The Agent's live `State`, passed to each hook and mutated in place.
    """
    for h in hooks.get(hook_point, []):
        h.run(state)


async def _run_hooks_async(hooks: dict[HookPoint, list[Hook]], hook_point: HookPoint, state: State) -> None:
    """
    Run every hook for the given hook point, preferring `run_async` and offloading sync-only `run` hooks.

    :param hooks: Hooks keyed by hook point.
    :param hook_point: The hook point whose hooks to run; hooks registered under other hook points are skipped.
    :param state: The Agent's live `State`, passed to each hook and mutated in place.
    """
    for h in hooks.get(hook_point, []):
        await _execute_component_async(h, state=state)
