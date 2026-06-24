# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.agents.state.state import State
from haystack.hooks.protocol import Hook, HookEvent


def _run_hooks(hooks: dict[HookEvent, list[Hook]], event: HookEvent, state: State) -> None:
    """Run every hook registered for the given lifecycle event, in list order."""
    for h in hooks.get(event, []):
        h.run(state)


async def _run_hooks_async(hooks: dict[HookEvent, list[Hook]], event: HookEvent, state: State) -> None:
    """Run every hook for the given event, awaiting hooks that define `run_async` and calling `run` otherwise."""
    for h in hooks.get(event, []):
        run_async = getattr(h, "run_async", None)
        if run_async is not None:
            await run_async(state)
        else:
            h.run(state)
