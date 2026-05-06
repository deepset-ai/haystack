# Transaction Protocol for Haystack Pipelines

The Transaction Protocol (CC BY 4.0) adds idempotency, rollback, and audit guarantees to agent pipelines — solving a critical gap in production Haystack deployments where pipeline failures leave systems in unknown states.

## The Problem

Haystack pipelines are powerful but linear. When a 10-step RAG pipeline fails at step 7:

- ❌ Steps 1-6 already executed — are their side effects safe?
- ❌ No way to rollback partial work
- ❌ Retrying re-executes everything, potentially doubling API costs
- ❌ No audit trail of what happened vs what should have happened

## The Solution

The Transaction Protocol wraps pipeline execution with three guarantees:

| Guarantee | Mechanism | Haystack Fit |
|-----------|-----------|--------------|
| **Idempotency** | Transaction IDs prevent double-execution | `run()` can be called N times, only executes once |
| **Rollback** | Reversal tokens for each step | Step 7 fails → steps 1-6 reversed cleanly |
| **Audit Trail** | Signed, immutable execution record | Every pipeline run has cryptographic proof |

## Architecture

```
┌─────────────────────────────────────────┐
│           Transaction Wrapper           │
│  ┌───────┐  ┌───────┐  ┌───────────┐   │
│  │ Begin │→ │Execute│→ │Commit/Roll │   │
│  │  TXN  │  │ Steps │  │   back     │   │
│  └───────┘  └───────┘  └───────────┘   │
│       ↓         ↓             ↓         │
│  Haystack Pipeline Components           │
└─────────────────────────────────────────┘
```

### Example

```python
from haystack import Pipeline
from works_with_agents import TransactionProtocol

pipeline = Pipeline()
pipeline.add_component("retriever", ...)
pipeline.add_component("generator", ...)

txn = TransactionProtocol.wrap(pipeline)

# First run — executes normally
result = txn.run(query="What is RAG?")
# txn_id: "txn-abc123" — committed, audit trail saved

# Same txn_id — returns cached result, doesn't re-execute
result = txn.run(query="What is RAG?", txn_id="txn-abc123")

# Step fails → all prior steps rolled back
try:
    result = txn.run(query="...")
except StepFailure:
    # Pipeline state restored, no partial side effects
    pass
```

## Why It Matters for Haystack

Haystack is the leading RAG framework in production. Production means:

1. **API costs matter** — idempotency prevents accidental re-execution
2. **State integrity matters** — rollback keeps systems consistent
3. **Compliance matters** — audit trails for regulated industries (healthcare, finance, legal)

## Getting Started

The Transaction Protocol is open source:

- **Spec:** https://workswithagents.com/specs/transaction.md (CC BY 4.0)
- **Python SDK:** `pip install works-with-agents`
- **Reference implementations:** 6 languages

## Related Specs

- [Identity Protocol](https://workswithagents.com/specs/identity.md) — Ed25519 agent identity for transaction signing
- [Compliance-as-Code](https://workswithagents.com/specs/compliance-as-code.md) — Executable regulation packs
- [Trust Score](https://workswithagents.com/specs/trust-score.md) — Verifiable agent reputation
