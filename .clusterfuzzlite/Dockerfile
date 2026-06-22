# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ClusterFuzzLite / OSS-Fuzz build environment for the Haystack Atheris fuzz targets.
# Pinned by digest for supply-chain integrity. Bump periodically (the OSS-Fuzz
# base-builder is updated frequently with toolchain fixes); resolve a fresh digest with:
#   docker buildx imagetools inspect gcr.io/oss-fuzz-base/base-builder-python:latest --format '{{.Manifest.Digest}}'
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:bdae8ffe13ebbaf3b653f0a5082d8d72e108d8cd9eed1fef1a85d8350efa3fbf

COPY . $SRC/haystack
WORKDIR $SRC/haystack
COPY .clusterfuzzlite/build.sh $SRC/
