# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from canals.__about__ import __version__

from canals.component import component, Component
from canals.pipeline.pipeline import Pipeline
from canals.pipeline.save_load import (
    save_pipelines,
    load_pipelines,
    marshal_pipelines,
    unmarshal_pipelines,
)
