#!/bin/bash -eu
# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Builds the Atheris fuzz targets for ClusterFuzzLite / OSS-Fuzz.
# `compile_python_fuzzer` is provided by the base-builder-python image.

pip3 install .

for harness in "$SRC"/haystack/test/fuzz/fuzz_*.py; do
  name=$(basename "$harness" .py)
  compile_python_fuzzer "$harness"

  # Ship a seed corpus if one exists for this harness. The runner unpacks
  # "<fuzzer>_seed_corpus.zip" next to the binary and seeds libFuzzer with it.
  corpus_dir="$SRC/haystack/test/fuzz/corpus/$name"
  if [ -d "$corpus_dir" ]; then
    zip -j "$OUT/${name}_seed_corpus.zip" "$corpus_dir"/*
  fi
done
