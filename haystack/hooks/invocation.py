# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.agents.state.state import State
from haystack.hooks.protocol import Hook, HookPoint


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
    Run every hook for the given hook point, awaiting `run_async` when defined and calling `run` otherwise.

    :param hooks: Hooks keyed by hook point.
    :param hook_point: The hook point whose hooks to run; hooks registered under other hook points are skipped.
    :param state: The Agent's live `State`, passed to each hook and mutated in place.
    """
    for h in hooks.get(hook_point, []):
        run_async = getattr(h, "run_async", None)
        if run_async is not None:
            await run_async(state)
        else:
            h.run(state)
