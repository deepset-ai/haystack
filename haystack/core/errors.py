# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

# TODO: Remove this when PipelineMaxLoops is removed
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    pass


class PipelineConnectError(PipelineError):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelineDrawingError(PipelineError):
    pass


class PipelineMaxLoops(PipelineError):
    # NOTE: This is shown also when importing PipelineMaxComponentRuns, I can't find an easy
    # way to fix this, so I will ignore that case.
    warnings.warn(
        "PipelineMaxLoops is deprecated and will be remove in version '2.7.0'; use PipelineMaxComponentRuns instead.",
        DeprecationWarning,
    )


class PipelineMaxComponentRuns(PipelineMaxLoops):
    pass


class PipelineUnmarshalError(PipelineError):
    pass


class ComponentError(Exception):
    pass


class ComponentDeserializationError(Exception):
    pass


class DeserializationError(Exception):
    pass


class SerializationError(Exception):
    pass
