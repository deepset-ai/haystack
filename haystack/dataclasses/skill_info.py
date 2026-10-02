# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class SkillInfo:
    """
    Lightweight metadata describing a skill.

    This is what a `SkillStore` returns when listing its skills, keeping the catalog cheap; the full skill
    content (the instructions body and bundled files) is fetched on demand.

    :param name: The skill's name, used to look it up.
    :param description: A short description of when to use the skill. Shown to the agent up front.
    """

    name: str
    description: str
