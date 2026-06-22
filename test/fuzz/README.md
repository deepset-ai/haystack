# Fuzz targets

[Atheris](https://github.com/google/atheris) fuzz harnesses for Haystack's
untrusted-input entry points. They are wired into CI via
[ClusterFuzzLite](https://google.github.io/clusterfuzzlite/)
(see [`.clusterfuzzlite/`](../../.clusterfuzzlite) and the
`ClusterFuzzLite PR fuzzing` workflow).

| Harness | Target | Why |
|---|---|---|
| `fuzz_pipeline_loads.py` | `Pipeline.loads` | Deserializing a serialized pipeline (YAML) is a documented attack surface. |
| `fuzz_document_from_dict.py` | `Document.from_dict` | Reconstructing a `Document` from an untrusted dict. |
| `fuzz_filters.py` | `document_matches_filter` | Evaluating an untrusted filter expression. |

Each harness catches the exceptions that are a *normal* reaction to malformed
input; anything else (a crash, unbounded recursion, a hang, or an unexpected
exception type) is reported by Atheris as a finding. The "expected" exception
lists can be tightened over time to surface more subtle bugs.

## Seed corpus

`fuzz_document_from_dict` and `fuzz_filters` parse the raw fuzzer input as JSON,
so the input domain *is* JSON text. A small seed corpus of valid inputs lives in
[`corpus/<harness>/`](corpus) to bootstrap coverage past the JSON parse — without
it a short run spends most of its budget producing inputs that don't parse. The
ClusterFuzzLite build (`.clusterfuzzlite/build.sh`) zips each `corpus/<harness>/`
into the `<harness>_seed_corpus.zip` the runner expects. Add a new valid input by
dropping a `.json` file into the matching directory.

## Run locally

```sh
pip install atheris
pip install -e .

# Fuzz for a bit (Ctrl-C to stop); -atheris_runs limits the number of inputs.
# Pass the seed corpus dir so libFuzzer starts from valid inputs.
python test/fuzz/fuzz_pipeline_loads.py -atheris_runs=100000
python test/fuzz/fuzz_document_from_dict.py test/fuzz/corpus/fuzz_document_from_dict -atheris_runs=100000
python test/fuzz/fuzz_filters.py test/fuzz/corpus/fuzz_filters -atheris_runs=100000
```

A directory argument is used as a seed corpus (and grown with new coverage); a
crashing input file reproduces a finding:

```sh
python test/fuzz/fuzz_filters.py crash-<hash>       # reproduce a crash
```

> Note: Atheris builds a native extension and is not part of the dev
> dependencies; install it on demand as shown above. `pytest` does not collect
> these files (they are named `fuzz_*.py`, not `test_*.py`).
