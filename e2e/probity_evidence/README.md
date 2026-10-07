# Native signed-effect contract

This end-to-end job checks native agent effects against a separately installed reader.
It helps Haystack maintainers catch changes in tool, trace and terminal behavior.

It uses published Observer source pinned to `5d9feedcfffcf441380e11e9f25a5db44d274068` and its hash locks.
The workflow selects that source automatically.
For a local run, put the selected Observer checkout beside the Haystack checkout.

```sh
cd e2e/probity_evidence
hatch run python run.py \
  --source ../../../selected-observer \
  --selection observer-source-selection.json \
  --framework-source ../.. \
  --output /tmp/fresh-haystack-evidence-contract \
  --python 3.13.3
```

The runner builds and installs the current Haystack checkout's wheel.
It checks installed source bytes against the selected Git tree and checks package dependencies.
It copies only selected, rechecked bytes into each build.
The reader has no Haystack installation.

The job derives a current-host reference contract from the frozen release profile.
It binds only `SDK_VERSION` and `SDK_SOURCE` to the selected host version and commit.
It keeps all 31 original tests and all substantive guards.
It retains original and derived module bytes, hashes and the exact two-selector diff.
Derived wheel metadata carries a distinct local version, matching producer dependency and embedded receipt.
This is a declared source transformation, not the unchanged frozen release profile.

The fixed workload runs real `Pipeline`, `Agent`, `Tool` and tracer transitions with scripted generator replies.
Its seven cases cover permission, wrong content, faults before and after durable writes, step limits and truncated replies.
Only the permitted effect reaches publication.
Two changed-input controls must refuse, including a changed invocation whose artifact digest was updated.
Original packets, decisions, refusal output, wheel hashes and process receipts remain in the job artifact.

This is a framework contract test, not a model-quality benchmark.
The producer, initial policy and keys share one operator.
Separate environments do not establish outside custody.
The repository operator owns the job, reference-source updates, policy and artifact retention after acceptance.
