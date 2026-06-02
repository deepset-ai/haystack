# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ClusterFuzzLite / OSS-Fuzz build environment for the Haystack Atheris fuzz targets.
# The base image is intentionally left as the rolling OSS-Fuzz tag: the fuzzing
# toolchain expects the latest base-builder, and OSS-Fuzz/ClusterFuzzLite manage it.
FROM gcr.io/oss-fuzz-base/base-builder-python

COPY . $SRC/haystack
WORKDIR $SRC/haystack
COPY .clusterfuzzlite/build.sh $SRC/
