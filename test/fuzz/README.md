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

## Run locally

```sh
pip install atheris
pip install -e .

# Fuzz for a bit (Ctrl-C to stop); -atheris_runs limits the number of inputs.
python test/fuzz/fuzz_pipeline_loads.py -atheris_runs=100000
python test/fuzz/fuzz_document_from_dict.py -atheris_runs=100000
python test/fuzz/fuzz_filters.py -atheris_runs=100000
```

Pass a directory argument to use/grow a seed corpus, and a crashing input file
to reproduce a finding:

```sh
python test/fuzz/fuzz_pipeline_loads.py corpus/            # use corpus dir
python test/fuzz/fuzz_pipeline_loads.py crash-<hash>       # reproduce a crash
```

> Note: Atheris builds a native extension and is not part of the dev
> dependencies; install it on demand as shown above. `pytest` does not collect
> these files (they are named `fuzz_*.py`, not `test_*.py`).
