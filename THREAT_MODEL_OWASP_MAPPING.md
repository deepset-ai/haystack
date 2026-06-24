# OWASP LLM Top 10 (2025) — Haystack Coverage Mapping

This document maps each entry in the [OWASP Top 10 for LLM Applications, version 2025](https://genai.owasp.org/llm-top-10/) to the controls, components, and operator obligations described in [`THREAT_MODEL.md`](THREAT_MODEL.md).

**Purpose.** Enterprise security reviewers need a one-page answer to the question "what does Haystack handle, and what do we have to handle?" This document is that one page. Each row identifies:

- **Haystack framework controls**: what the library does to address the risk on the operator's behalf, with file references where applicable.
- **Application developer controls (required)**: the controls that must exist in the application built on Haystack for that risk to be addressed end-to-end.
- **THREAT_MODEL.md sections**: the operator-obligations rows and STRIDE entries that elaborate on each topic.

Each row resolves to one of three coverage states:

| State | Meaning |
|---|---|
| **Shared** | Haystack provides building blocks; the application must wire them in. |
| **Framework partial** | Haystack mitigates one or more sub-risks directly; remaining sub-risks are the application's responsibility. |
| **Application-only** | Haystack provides no direct control. The application or its infrastructure owns the risk entirely. |

This is intentional. Haystack is a framework, not a platform — see `SECURITY.md` and `THREAT_MODEL.md` for the rationale.

---

## Summary table

| # | OWASP LLM Risk | Coverage | One-line owner statement |
|---|---|---|---|
| LLM01:2025 | Prompt Injection | **Shared** | Haystack provides general-purpose routing, classification, and validation building blocks that applications can use in prompt-injection defenses; the application chooses the detector and decides what content to trust. |
| LLM02:2025 | Sensitive Information Disclosure | **Shared** | Haystack provides `Secret` wrapping for serialization and pluggable tracing hooks; the application owns logging/tracing redaction, output filtering, system-prompt secrecy, and per-tenant data segregation. |
| LLM03:2025 | Supply Chain | **Framework partial** | Haystack SHA-pins GitHub Actions, runs daily Dependabot, and publishes via PyPI trusted publishing. The application is responsible for the model, dataset, and embedding-provider supply chain. |
| LLM04:2025 | Data and Model Poisoning | **Application-only** | Haystack does not curate training data or validate model integrity. The application owns dataset provenance and pre-load model integrity checks. |
| LLM05:2025 | Improper Output Handling | **Shared** | Haystack returns structured outputs and provides validator/router components; the application owns sanitization at each downstream sink. |
| LLM06:2025 | Excessive Agency | **Shared** | Haystack caps agent-loop iterations and exposes per-tool registration; the application owns tool scoping, argument validation, and confirmation policies. |
| LLM07:2025 | System Prompt Leakage | **Application-only** | Haystack passes prompt templates through; the application owns secret-free prompt design and output filtering for prompt extraction attempts. |
| LLM08:2025 | Vector and Embedding Weaknesses | **Shared** | Haystack supports structured `filters` and document-store-agnostic retrieval; the application owns per-tenant filter enforcement and embedding-source hygiene. |
| LLM09:2025 | Misinformation | **Framework partial** | Haystack supports citation patterns (e.g., `AnswerBuilder` with sources) and grounded-generation primitives; the application owns ground-truth UX and confidence-disclosure to end users. |
| LLM10:2025 | Unbounded Consumption | **Framework partial** | Haystack supports agent-loop iteration caps and timeout knobs; the application owns per-user/per-tenant rate limits and cost budgets. |

---

## Detailed mapping

### LLM01:2025 — Prompt Injection

**Risk in one line.** Crafted input — direct from a user or indirect via retrieved documents, tool outputs, or uploaded files — overrides the model's intended instructions.

**Haystack framework controls.**

- The `PromptBuilder` and `ChatPromptBuilder` components separate the system instruction from data interpolation, supporting the "structured prompt" defense pattern.
- General-purpose routing, classification, and validation building blocks (`TransformersZeroShotTextRouter`, `LLMMessagesRouter`, `ConditionalRouter`, `JsonSchemaValidator`) can be wired in front of generators so that the application can gate retrieved content using a classifier or guard model of its choice.
- Tracing hooks (OpenTelemetry, Datadog, custom tracers) make it possible to log gate verdicts alongside generations for incident review, once the operator configures redaction.

Haystack does not ship a turnkey prompt-injection detector. The application chooses the detector and wires it through the components above.

**Application developer controls (required).**

- Place a classifier (or an equivalent application-layer guard) in front of generators that consume retrieved or user-supplied content.
- Treat the LLM's response — including tool-call arguments — as untrusted data; validate before any downstream sink.
- Restrict the data the model has access to in the first place. Defense-in-depth: a prompt injection that succeeds against a model with no destructive tools attached has limited blast radius.
- Implement input-screening, output-screening, and action-screening per the [OWASP Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html).

**Cross-references in `THREAT_MODEL.md`.**

- Generators table → row "I" (prompt-injection payload exfiltrating secrets).
- Tools table → all rows (LLM output flows through tool arguments).
- Operator obligations matrix → "Sanitizing prompt-injection payloads in retrieved documents."

---

### LLM02:2025 — Sensitive Information Disclosure

**Risk in one line.** The model reveals data the application did not intend to expose — training data fragments, system prompts, prior user prompts, tenant data, secrets in retrieved documents.

**Haystack framework controls.**

- `Secret.from_token()` refuses serialization through `to_dict()`, so a token wrapped this way cannot escape via pipeline dumps. `Secret.from_env_var()` serializes the environment-variable name rather than the resolved value, so credentials live only in the runtime environment. Neither mechanism replaces application-level logging and tracing redaction.
- A pluggable tracing layer (`haystack/tracing/`, including OpenTelemetry, Datadog, and logging backends) gives the operator a single chokepoint to attach redaction and filtering before spans leave the process. Redaction itself is the operator's responsibility.

**Application developer controls (required).**

- Filter LLM outputs for PII, PHI, secrets, and tenant data before display to end users.
- Treat system prompts as potentially exposed; do not place secrets inside them.
- Segregate per-tenant data at retrieval (`filters` passed to the retriever) rather than relying on the model to "remember" tenancy boundaries.
- Apply DLP at the prompt-input boundary if employees can paste arbitrary text into the application.

**Cross-references in `THREAT_MODEL.md`.**

- Operator obligations matrix → "Securing secrets at rest," "Filtering PII in inputs and outputs," "Authorizing access to documents."
- Document stores and retrievers table → row "S" (cross-tenant retrieval).

---

### LLM03:2025 — Supply Chain

**Risk in one line.** A compromised or malicious component anywhere in the chain — foundation model, fine-tuning dataset, LoRA adapter, inference SDK, embedding model, third-party plugin — affects the application.

**Haystack framework controls.**

- Haystack SHA-pins GitHub Actions and uses daily Dependabot monitoring for the GitHub Actions, pip, npm, and Docker ecosystems supported in this repository (see `.github/dependabot.yml`).
- `actions/checkout` is invoked with `persist-credentials: false` on workflows that do not need to push, limiting the blast radius of a compromised action.
- The `haystack-ai` PyPI package is published through a GitHub Actions release workflow that uses PyPI trusted publishing (OIDC, no long-lived PyPI tokens).
- `SECURITY.md` defines a coordinated disclosure path for vulnerabilities reported in the framework itself.

Runtime Python dependencies are declared with version ranges rather than full pin-by-hash; integrity is enforced primarily by the consumer's lockfile and dependency-scanning tooling. CycloneDX SBOMs, sigstore signing, PEP 740 attestations, OpenSSF Scorecard, and CodeQL are not part of the current release or CI matrix — contributions adding any of these are welcome (see `THREAT_MODEL.md` § Defense-in-depth contributions are welcome).

**Application developer controls (required).**

- Verify the provenance of any LLM provider, embedding model, vector store, or third-party `haystack-*` integration the application loads. Third-party `haystack-*` packages have their own maintainers and threat models — see `THREAT_MODEL.md` § Scope.
- Pin model versions (`text-embedding-3-small`, not "latest") and verify integrity at load time when possible.
- Apply traditional software-supply-chain controls — SBOM, dependency scanning, source attestation — to the application that embeds Haystack.

**Cross-references in `THREAT_MODEL.md`.**

- Scope section → exclusion of third-party `haystack-*` integration packages.
- Defense-in-depth contributions are welcome section → invitation for pinned-version/integrity-check contributions in first-party components.

---

### LLM04:2025 — Data and Model Poisoning

**Risk in one line.** Training data, fine-tuning data, embedding inputs, or RAG sources are tainted so the model produces attacker-chosen outputs for attacker-chosen triggers.

**Haystack framework controls.**

- Haystack does not train models. It is consumer-side relative to this risk.
- Haystack does not curate, validate, or check the integrity of documents ingested into a `DocumentStore`.

**Application developer controls (required).**

- Apply data-provenance controls at the ingestion boundary into the `DocumentStore` — signed sources, ingestion review, eviction discipline for sources that fail review.
- Run canary evaluations against the embedded model on a known clean prompt set before deploying changes to retrieved-content sources.
- Audit the LLM provider's stated controls on fine-tuning dataset integrity; treat any uploaded fine-tuning data as the application's responsibility.

**Cross-references in `THREAT_MODEL.md`.**

- Operator obligations matrix → "Authorizing access to documents," "Logging and auditing pipeline executions."

---

### LLM05:2025 — Improper Output Handling

**Risk in one line.** LLM output is rendered, executed, or passed to a downstream sink (SQL, shell, HTML, file path, tool argument) without validation, turning the LLM into an injection vector for classical web vulnerabilities.

**Haystack framework controls.**

- Generators return structured Python objects, not pre-rendered HTML or pre-executed code. The output type makes the un-trust explicit.
- Routers, validators, and conditional-router components in `haystack/components/routers/` and `haystack/components/validators/` allow operators to gate output before it reaches a sink.
- Tool definitions describe the interface exposed to the model. They do not, by themselves, validate LLM-produced arguments at invocation time; applications must enforce authorization and validate type, range, allowlist, and business-rule constraints in the tool implementation or in an explicit pre-invocation guard.

**Application developer controls (required).**

- HTML-escape model output before rendering. Sanitize Markdown if rendered. Never `eval`, never `exec`, never string-format into a shell command.
- Validate tool arguments inside each tool's `invoke` body — type, range, allowlist — regardless of what the LLM produced.
- Parameterize database queries. Never concatenate model output into SQL or other DSLs.
- Treat file paths produced by the model as untrusted; canonicalize and bound them before file-system access.

**Cross-references in `THREAT_MODEL.md`.**

- Operator obligations matrix → "Sanitizing model output before display to end users," "Validating tool-call arguments before execution."
- Tools table → rows "T" (argument tampering) and "E" (LLM-suggested commands).

---

### LLM06:2025 — Excessive Agency

**Risk in one line.** The agent has more functionality, permissions, or autonomy than the use case requires — over-broad tool catalog, write-scope credentials on read-only tools, missing human-in-the-loop on destructive actions.

**Haystack framework controls.**

- The `Agent` runtime supports per-agent iteration caps to bound runaway loops.
- Tools are registered explicitly per-agent; there is no implicit tool discovery.
- The `ChatGenerator` `tools` parameter requires per-call explicit binding — the framework does not auto-attach tools.

**Application developer controls (required).**

- Scope each tool to the minimum capability required. A research agent should not be granted send-mail tools "just in case."
- Bind tool calls to the originating user identity at the application layer; do not let the LLM choose the identity.
- Add human-in-the-loop confirmation on destructive actions — deletes, writes, financial transactions, code execution.
- Enforce per-agent cost and rate budgets at the application layer; iteration caps are necessary but not sufficient.

**Cross-references in `THREAT_MODEL.md`.**

- Tools table → all rows.
- Generators table → row "E" (LLM-suggested tool calls executed without checks).
- Operator obligations matrix → "Validating tool-call arguments before execution," "Authenticating end users."

---

### LLM07:2025 — System Prompt Leakage

**Risk in one line.** A user extracts the system prompt and learns the application's business logic, guardrails, secrets, or routing rules.

**Haystack framework controls.**

- `PromptBuilder` and `ChatPromptBuilder` interpolate templates; they do not log the rendered prompt by default.
- `Secret.from_env_var()` keeps credentials out of the serialized pipeline definition, so YAML-stored prompts do not embed resolved tokens. This does not protect against the model echoing the system prompt at generation time.

**Application developer controls (required).**

- Treat the system prompt as potentially exposed. Do not place secrets, internal endpoints, or business-rule details that aid attackers inside it.
- Apply output filtering to detect and block prompt-extraction attempts (verbatim instruction echoes, instruction-overflow patterns).
- Design business logic to be defense-in-depth: assume the attacker has read the system prompt and ask whether the application is still safe.

**Cross-references in `THREAT_MODEL.md`.**

- Generators table → row "I" (prompt injection causing the LLM to emit secrets).

---

### LLM08:2025 — Vector and Embedding Weaknesses

**Risk in one line.** Weaknesses in how embeddings are generated, stored, retrieved, or access-controlled — cross-tenant leakage, embedding inversion to recover source content, RAG injection, missing per-user authorization.

**Haystack framework controls.**

- Retrievers accept structured `filters` and forward them to the underlying document store, supporting per-tenant retrieval restrictions when the application supplies the filter.
- First-party embedders use the operator-configured model and surface `dimensions` and other knobs at runtime; nothing is hidden.
- The `DocumentStore` protocol abstracts the backing store so the application can choose a store with the access-control properties it needs.

**Application developer controls (required).**

- Always set `filters` from the authenticated user's context on every retrieval. Cross-tenant leak is the easiest failure mode and the framework cannot enforce it for the application.
- Choose a document store backend that enforces per-document or per-row authorization at the storage layer if the application's tenancy model demands it.
- Apply hygiene to documents before insertion — strip injected instructions, scrub PII, version sources — because the embedder will faithfully embed whatever the application provides.
- Consider the embedding-inversion risk for highly sensitive corpora: an attacker with read access to embeddings can recover plausible source content.

**Cross-references in `THREAT_MODEL.md`.**

- Document stores and retrievers table → all rows.
- Operator obligations matrix → "Authorizing access to documents."

---

### LLM09:2025 — Misinformation

**Risk in one line.** The model hallucinates plausible-sounding but incorrect outputs — citations, code references, legal claims, medical claims, financial figures — that downstream consumers trust.

**Haystack framework controls.**

- `AnswerBuilder` and related components support attaching source documents to answers, supporting citation-based grounding patterns.
- Pipelines support multi-step retrieval-augmented generation, allowing the application to ground generations against verifiable sources.
- Routers and validators support post-generation checks (e.g., reject answers without sources).

**Application developer controls (required).**

- Display sources to end users so they can verify claims; do not surface unsourced model output in regulated contexts.
- Apply domain-specific validation — citation existence, calculation re-check, jurisdiction filtering — before display in legal, medical, or financial settings.
- Communicate confidence to end users honestly; do not present hallucinated content with the same UI weight as grounded content.

**Cross-references in `THREAT_MODEL.md`.**

- Generators table → row "T" (LLM output trusted as system-supplied).

---

### LLM10:2025 — Unbounded Consumption

**Risk in one line.** Without per-identity, per-tenant, per-conversation, or per-tool limits, the application incurs runaway cost, capacity exhaustion, or denial-of-service through legitimate-looking but oversized usage.

**Haystack framework controls.**

- The `Agent` runtime exposes iteration caps that bound agent loops.
- Generators expose `max_tokens`, `timeout`, and `max_retries` knobs so the operator can bound a single request's resource consumption.
- Tracing hooks emit per-call latency and token-usage data, supporting downstream cost and budget alerting.

**Application developer controls (required).**

- Apply per-user, per-tenant, and per-conversation rate limits at the application layer. Haystack will faithfully execute whatever volume the application drives.
- Apply per-call cost budgets; reject requests whose expected token cost exceeds a threshold.
- Cap the agent's tool-invocation count per session, in addition to the iteration count.
- Apply infrastructure-layer DoS controls (load balancer rate limits, WAF rules) as a backstop for application-layer logic failures.

**Cross-references in `THREAT_MODEL.md`.**

- Operator obligations matrix → "Rate-limiting end users."
- Generators table → row "D" (runaway loops in agentic patterns).
- Fetchers and Tools tables → row "D" entries.

---

## How to use this document in a compliance review

For each row of the OWASP LLM Top 10:

1. Read the **Coverage** column in the summary table. If it is **Application-only**, the framework is not in the loop — the auditor should focus on the application code and the application's runtime infrastructure.
2. If the coverage is **Shared** or **Framework partial**, read the **Application developer controls (required)** subsection in the detailed mapping. These items must be demonstrable in the application's code review, runtime configuration, or runbooks.
3. Reference the matching row of `THREAT_MODEL.md` § Operator obligations matrix for the canonical statement of who owns the control.
4. For findings or gaps, file them against the application, not against Haystack, unless the gap matches one of the framework responsibilities described in `THREAT_MODEL.md` (in which case it is in scope for `SECURITY.md` reporting).

This document, `THREAT_MODEL.md`, and `SECURITY.md` are designed to be cited together as the framework's statement of security posture.

---

## References

- [`SECURITY.md`](SECURITY.md) — vulnerability reporting policy and out-of-scope items.
- [`THREAT_MODEL.md`](THREAT_MODEL.md) — trust boundaries, operator obligations matrix, STRIDE analysis per component class.
- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/) — source for the canonical risk taxonomy used in this mapping.
- [OWASP LLM Prompt Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) — referenced under LLM01.

---

## Updates

This document is versioned with the repository and refreshed when the OWASP Top 10 for LLM Applications is republished or when Haystack's controls change materially. Significant updates are noted in release notes under the `security:` tag.
