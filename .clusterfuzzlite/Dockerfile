# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ClusterFuzzLite / OSS-Fuzz build environment for the Haystack Atheris fuzz targets.
# Pinned by digest for supply-chain integrity. Bump periodically (the OSS-Fuzz
# base-builder is updated frequently with toolchain fixes); resolve a fresh digest with:
#   docker buildx imagetools inspect gcr.io/oss-fuzz-base/base-builder-python:latest --format '{{.Manifest.Digest}}'
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:140e90791e768fdcbbbd9f7299e5d4d9fc8cfd1ed3c906aa1ca555c9dfe69fa6

COPY . $SRC/haystack
WORKDIR $SRC/haystack
COPY .clusterfuzzlite/build.sh $SRC/
