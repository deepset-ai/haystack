# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillMeta:
    """
    Metadata describing a single skill.

    :param name: The skill's name, used by the agent to load it.
    :param description: A short description of when to use the skill. Shown to the agent up front.
    :param path: The skill's directory. Set by `FileSystemSkillStore`; can be `None` for other stores.
    """

    name: str
    description: str
    path: Path | None = field(default=None)
