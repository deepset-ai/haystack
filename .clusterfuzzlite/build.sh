#!/bin/bash -eu
# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Builds the Atheris fuzz targets for ClusterFuzzLite / OSS-Fuzz.
# `compile_python_fuzzer` is provided by the base-builder-python image.

pip3 install .

for harness in "$SRC"/haystack/test/fuzz/fuzz_*.py; do
  compile_python_fuzzer "$harness"
done
