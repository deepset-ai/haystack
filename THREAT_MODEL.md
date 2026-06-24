# Haystack Threat Model

## Purpose of this document

This document describes the threat model that Haystack is designed and maintained against. **Its primary purpose is to make the line between framework responsibility and application responsibility explicit**, so that:

- Application developers building on Haystack can identify, up front, where their code must add validation, authorization, sandboxing, or network controls.
- Security reviewers can scope their reviews of Haystack-based systems correctly, focusing effort on the boundary code rather than on the framework internals.
- Vulnerability reporters can determine whether a finding is in scope under [`SECURITY.md`](SECURITY.md) before investing time in a writeup.
- Maintainers have a shared vocabulary for evaluating proposed security changes and feature requests.

This document is **descriptive, not prescriptive**: it describes how Haystack is built and what assumptions it makes about its environment. It does not tell you how to secure your application — only what assumptions you must satisfy for Haystack to behave safely inside it.

This document complements `SECURITY.md`. Where `SECURITY.md` defines what is in scope for vulnerability reports, `THREAT_MODEL.md` explains *why* the scope is drawn where it is.

---

## Scope

This threat model covers:

- The Haystack 2.x runtime as published on PyPI (`haystack-ai`).
- First-party components shipped in this repository.
- Pipelines, agents, and the serialization formats they use (`Pipeline.dumps`/`Pipeline.loads`, `PipelineSnapshot`, `AgentSnapshot`).

This threat model does **not** cover:

- Third-party integrations distributed as separate `haystack-*` packages. Those packages have their own maintainers, threat models, and disclosure processes.
- The infrastructure on which Haystack runs (containers, hosts, network, cloud).
- The application code that wires Haystack components into a product.
- The LLM provider, vector database, document store, or any other external service Haystack talks to.

If you are running Haystack inside a managed product (for example a deepset Cloud deployment), that product has its own threat model and security posture in addition to this one.

---

## Trust boundaries

Haystack runs in a system with **four trust zones**. Every byte of data Haystack handles crosses at least one of these boundaries, and each boundary carries different assumptions about what the data may contain.

```
+--------------------------------------------------------------+
|                                                              |
|   Zone 1: OPERATOR CODE                                      |
|   - Application code, pipeline YAML, environment, secrets    |
|   - Trusted as code. Treated as source code by the operator. |
|                                                              |
|     +------------------------------------------------------+ |
|     |                                                      | |
|     |  Zone 2: HAYSTACK RUNTIME                            | |
|     |  - The Pipeline, Components, Agents in this repo     | |
|     |  - Trusts Zone 1 inputs as configuration.            | |
|     |  - Treats Zone 3 and Zone 4 outputs as data.         | |
|     |                                                      | |
|     +------------------------------------------------------+ |
|                                                              |
+--------------------------------------------------------------+
                     |                       |
                     v                       v
       +-----------------------+   +-----------------------+
       |                       |   |                       |
       |  Zone 3: LLM / MODEL  |   |  Zone 4: EXTERNAL     |
       |  - OpenAI, Anthropic, |   |  SERVICES             |
       |    Bedrock, local     |   |  - Document stores,   |
       |    inference servers  |   |    vector DBs, web    |
       |  - Outputs are        |   |    pages, APIs,       |
       |    UNTRUSTED data.    |   |    files on disk      |
       |                       |   |  - Contents are       |
       |                       |   |    UNTRUSTED data.    |
       |                       |   |                       |
       +-----------------------+   +-----------------------+
```

### Zone 1 — Operator code

This includes:

- The Python source code of the application embedding Haystack.
- Pipeline and agent definitions, including YAML files and serialized `PipelineSnapshot` / `AgentSnapshot` artifacts the operator has chosen to load or resume.
- Environment variables, configuration files, and secrets (`Secret.from_env_var`, `Secret.from_token`).
- Any callable or class passed to Haystack via `import_class_by_name`, `deserialize_callable`, custom components, or tool definitions.

**Assumption**: Everything in Zone 1 has the same trust level as the application's own source code. The operator controls it, reviews it, and applies the same change-management controls to it as they do to code.

**Consequence**: Haystack does **not** attempt to defend itself against attacker-controlled Zone 1 content. If an attacker can write to your YAML file or modify your environment variables, they already have arbitrary code execution at your application's privilege level, with or without Haystack.

### Zone 2 — Haystack runtime

The library code in this repository. Haystack:

- Trusts Zone 1 inputs as configuration (model names, URLs, prompts, component instances).
- Treats Zone 3 (LLM) and Zone 4 (external service) outputs as untrusted data. Haystack is designed so that malformed model or external-service data is handled at this boundary; a framework defect that causes unintended code execution, state disclosure, or unsafe parsing as a direct result of malformed Zone 3 or Zone 4 data is considered a security issue (see `SECURITY.md`).
- Does not, however, sanitize, filter, or interpret the *content* of Zone 3 and Zone 4 data on the operator's behalf. That is the operator's job.

### Zone 3 — LLM / model

The model provider that the operator has connected Haystack to. From Haystack's perspective:

- The model's response text, function-call arguments, and tool-call arguments are **untrusted data**, even if the model is hosted by a reputable vendor.
- Prompt injection from upstream data sources (Zone 4) means the LLM may produce text and tool arguments that an attacker controls.
- Haystack will pass LLM output to downstream components when the pipeline is configured to do so. Whether that downstream component treats the output safely is a pipeline-design question, not a framework-implementation question.

### Zone 4 — External services

Document stores, vector databases, web URLs, files on the local filesystem, third-party APIs, search engines. From Haystack's perspective:

- Contents are untrusted data. A web page returned by `LinkContentFetcher` may contain anything; a document in a vector store may have been written by an attacker.
- Haystack components that read from Zone 4 are designed to handle malformed content without escaping sandboxing or corrupting unrelated pipeline state; the content itself is passed unchanged into the rest of the pipeline for the application to interpret.

---

## Operator obligations matrix

The following table summarizes who is responsible for what at each trust boundary. **It is the most important section of this document**: if you are building on Haystack, this is the list you must work through for your application.

| Concern | Who owns it | What Haystack does | What the application must do |
|---|---|---|---|
| **Authenticating end users** | Application | Nothing | Add an authentication layer in front of the application; Haystack components run with the application's identity, not the user's. |
| **Authorizing access to documents** | Application | Stores documents as the operator inserts them | Filter documents by user identity *before* they reach a retriever, or use a document store that supports per-user filters and pass those filters in. |
| **Validating user-supplied URLs before fetching** | Application | Fetches the URL the operator passes | Validate the URL, resolve its hostname, reject private CIDRs, and pass only validated URLs to `LinkContentFetcher` or equivalent components. |
| **Validating user-supplied file paths before reading** | Application | Opens the file path the operator passes | Validate, canonicalize, and bound the path inside an allowed directory before passing it to file-reading converters. |
| **Sanitizing prompt-injection payloads in retrieved documents** | Application | Inserts the document into the prompt template as configured | Wire a classifier or guard model (operator's choice) in front of generators using Haystack's general-purpose routing and validation components, or apply equivalent guarding at the application layer. |
| **Network egress controls** | Application / infrastructure | Issues outbound HTTP requests to whatever URL it is given | Enforce egress firewall rules at the infrastructure layer (NetworkPolicy, security group, VPC). |
| **Sandboxing executed code** | Application | Calls callables the operator registered | Run Haystack inside a container, jail, or namespace whose blast radius matches the threat model. |
| **Securing secrets at rest** | Application | Wraps secrets in `Secret` to prevent accidental serialization | Use `Secret.from_env_var` (preferred) over `Secret.from_token`; ensure environment variables are sourced from a secret manager; do not commit YAML pipelines containing tokens. |
| **Protecting the pipeline definition** | Application | Loads any pipeline definition it is handed | Treat pipeline YAML and snapshots as source code: store under version control, restrict write access, never load from end-user input. |
| **Rate-limiting end users** | Application | Issues calls to LLMs and external services as the pipeline demands | Throttle at the application layer; Haystack will faithfully execute whatever volume the application drives. |
| **Logging and auditing pipeline executions** | Application | Provides tracing hooks (OpenTelemetry, custom tracers) | Wire up the tracing hooks; ship logs to a SIEM with retention appropriate to the operator's compliance regime. |
| **Validating tool-call arguments before execution** | Application | Passes tool arguments produced by the LLM to the tool's `invoke` method | Validate, type-check, and bound the arguments inside the tool implementation; never trust LLM-supplied arguments as if they came from a trusted caller. |
| **Sanitizing model output before display to end users** | Application | Returns model output as a Python object | Treat the output as untrusted markup; HTML-escape it before rendering, sanitize Markdown if rendered, never `eval` it. |
| **Filtering PII in inputs and outputs** | Application | Passes content through untouched | Apply PII detection and redaction at the application layer; Haystack does not look inside content. |

If you are doing a security review of a Haystack-based application, every "Application" row in this table is something you should be able to find evidence of in the application's code.

---

## STRIDE analysis by component class

The following sections walk through the major Haystack component classes and the threats most relevant to each. Each entry lists the threat, the boundary it sits on, and what mitigation is expected at which layer.

### Fetchers (`haystack/components/fetchers/`)

Components in this class issue outbound network requests. Examples: `LinkContentFetcher`.

| Threat (STRIDE) | Boundary | Framework responsibility | Application responsibility |
|---|---|---|---|
| **S** — A URL passed by the application points at a service the application did not intend (SSRF, metadata endpoint) | Application → Haystack | None (see `SECURITY.md` → SSRF section) | Validate the URL, resolve hostname, reject private CIDRs, enforce egress firewall. |
| **T** — A fetched response is tampered in transit | Haystack → external service | Use TLS, follow redirects safely | Use HTTPS endpoints; pin TLS roots at the OS level if required. |
| **R** — Difficult to attribute a malicious fetch back to the originating end user | Application | None | Log the originating user identity alongside the URL in the application layer. |
| **I** — A fetched response leaks more bytes than expected (memory pressure, slow-DoS) | Haystack | Expose timeout and retry configuration | Enforce response-size limits and resource budgets at the application or infrastructure layer (HTTP client wrapper, reverse proxy, container memory cap). |
| **D** — A redirect chain or malicious server causes the fetch to hang | Haystack | Sensible default timeouts; expose the knob | Tune timeouts to the application's latency budget. |
| **E** — None within Haystack itself (no privilege escalation surface) | n/a | n/a | n/a |

### Converters (`haystack/components/converters/`)

Components in this class parse files into Documents. Examples: `PyPDFToDocument`, `MSGToDocument`, `HTMLToDocument`, `OpenAPIServiceToFunctions`, `ImageFileToDocument`.

| Threat (STRIDE) | Boundary | Framework responsibility | Application responsibility |
|---|---|---|---|
| **S** — A file path passed to a converter escapes its intended directory (path traversal) | Application → Haystack | Document the assumption that paths are operator-controlled | Canonicalize and bound paths before passing them in. |
| **T** — A parsed document's metadata is later trusted as if it were system-provided | Haystack → application | Document that meta fields originate from the file contents | Treat parsed metadata as untrusted data; do not use it as a primary key, file path, or authorization decision input. |
| **R** — Pipeline cannot attribute which user supplied a particular file | Application | None | Log the source-of-truth alongside the file at ingestion. |
| **I** — A maliciously crafted file leaks server-side data (XXE, SSRF via `$ref`, file:// resolution) | Haystack | Use safe parsers by default (e.g., `defusedxml` patterns, no remote `$ref` resolution where avoidable) | Sandbox the converter at the infrastructure layer where the file source is unknown. |
| **D** — A maliciously crafted file consumes unbounded memory or CPU (zip bombs, deeply nested JSON, recursive structures) | Haystack | Use streaming parsers where practical; document size assumptions | Cap file sizes at the ingestion layer; run converters with memory and CPU limits. |
| **E** — A converter that allows arbitrary code execution from a malformed file would be a Haystack bug (in scope) | Haystack | Parse safely; no `exec`, no `pickle.load` of file contents | Report bugs of this type through `SECURITY.md`. |

### Generators (`haystack/components/generators/`)

Components in this class call LLM provider APIs. Examples: `OpenAIGenerator`, `OpenAIChatGenerator`, `AzureOpenAIGenerator`.

| Threat (STRIDE) | Boundary | Framework responsibility | Application responsibility |
|---|---|---|---|
| **S** — Generator is misconfigured to talk to an attacker-controlled "OpenAI-compatible" base URL | Application | None | Lock down `api_base_url` to a known provider; do not let end-user input choose providers. |
| **T** — LLM output is later trusted as if it were system-supplied | Haystack → application | Return LLM output as structured data; do not pre-trust it | Sanitize and validate all LLM output before using it as a tool argument, file path, URL, SQL fragment, or rendered HTML. |
| **R** — Cannot prove which end user triggered a particular generation | Application | Provide tracing hooks | Wire OpenTelemetry or custom tracers; ship request metadata alongside the generation. |
| **I** — A prompt-injection payload smuggled in via a retrieved document causes the LLM to emit secrets it has access to | Application | Provide general-purpose routing, classification, and validation building blocks that applications can use in prompt-injection defenses | Place a classifier or guard in front of generators; restrict what data the model has access to in the first place. |
| **D** — Runaway loops in agentic patterns cause cost or rate-limit issues | Haystack | Cap iterations on the `Agent` runtime | Apply per-pipeline rate limiting and cost budgets at the application layer. |
| **E** — LLM-suggested tool calls are executed without further checks | Application | Pass tool arguments through as-is | Validate tool arguments inside each tool's `invoke`; reject calls that violate the tool's safety contract. |

### Tools (`haystack/components/tools/`)

Tools wrap callables the LLM can request the runtime to invoke. The threats here are the most consequential in modern agentic systems.

| Threat (STRIDE) | Boundary | Framework responsibility | Application responsibility |
|---|---|---|---|
| **S** — The LLM produces tool arguments that impersonate a different end user | Application | None | Bind tool calls to the originating user identity at the application layer; do not let the LLM choose the identity. |
| **T** — Tool arguments are tampered between the LLM and the tool implementation | Haystack | Pass arguments unmodified | Validate, type-check, and bound the arguments inside the tool body. |
| **R** — Cannot prove which prompt led to which tool invocation | Application | Provide tracing hooks | Log the prompt, tool name, and argument JSON together. |
| **I** — A tool implementation reads data the calling user is not authorized to see | Application | None | Authorize inside the tool, using the application's user-identity context — not the LLM's claims. |
| **D** — The LLM repeatedly invokes a tool, causing cost or rate-limit issues | Application | None | Rate-limit per-user inside the application; cap iterations at the agent level. |
| **E** — A tool executes arbitrary commands derived from LLM output | Application | None | Never construct subprocess or SQL strings from LLM output without an allowlist; prefer typed APIs over shell strings. |

### Pipeline serialization (`Pipeline.loads`, `PipelineSnapshot`, `AgentSnapshot`)

Pipeline YAML and snapshots are an executable serialization format. They are intentionally extensible via dynamic class loading.

| Threat (STRIDE) | Boundary | Framework responsibility | Application responsibility |
|---|---|---|---|
| **S/T/E** — A pipeline definition or snapshot from an untrusted source executes attacker-controlled code on load | Application → Haystack | None — `Pipeline.loads` is a Zone 1 → Zone 2 transition, and Zone 1 is trusted by design (see `SECURITY.md`) | Treat pipelines and snapshots as code: version-control them, review changes, never load from end-user input. |
| **R** — Cannot tell which version of a pipeline produced a given result | Application | None | Tag pipeline versions in metadata; persist the loaded definition alongside the run. |
| **I** — A pipeline definition encodes secrets in plain text | Operator | Provide `Secret` to wrap values that should not serialize | Use `Secret.from_env_var` so credentials resolve at runtime, not at serialization time. |

### Document stores and retrievers

Examples: `InMemoryDocumentStore` plus third-party document store integrations distributed as `haystack-*` packages.

| Threat (STRIDE) | Boundary | Framework responsibility | Application responsibility |
|---|---|---|---|
| **S** — An end user retrieves documents intended for a different user (cross-tenant leak) | Application | Pass `filters` through to the underlying store | Always set `filters` from the authenticated user's context; never rely on the LLM or end-user-provided filter expressions alone. |
| **T** — A filter expression supplied by the end user is concatenated into a backend query (analogous to SQL injection in vector stores that compile filters to SQL) | Application + integration | First-party Haystack handles structured filters as typed objects | Pass structured filters, not raw query strings, into stores. Integration packages should parameterize backend queries. |
| **R** — Cannot attribute which user inserted which document | Application | None | Set meta fields at ingestion with the originating identity. |
| **I** — Retrieved documents contain prompt-injection or PII | Zone 4 → Haystack → LLM | None | Apply prompt-injection classifiers and PII filters at the application layer. |
| **D** — A retriever loads more documents than the application can handle | Haystack | Honor `top_k` and similar bounds | Choose `top_k` defensively per the LLM's context budget. |

### Web search and connectors (`haystack/components/websearch/`, `haystack/components/connectors/`)

Components in this class issue outbound network requests with credentials.

| Threat (STRIDE) | Boundary | Framework responsibility | Application responsibility |
|---|---|---|---|
| **S** — A search query crafted from LLM output causes the connector to issue a request the operator did not intend | Application | None | Validate or constrain queries that originate from LLM output. |
| **T** — Connector results are trusted as if they came from a curated source | Haystack → application | Return raw results; do not annotate as "trusted" | Treat all retrieved snippets as untrusted data (Zone 4). |
| **R** — Cannot attribute an outbound search to a user | Application | Provide tracing hooks | Wire tracing; log the originating user. |
| **I/D** — Connector credentials leak via logs | Application | `Secret.from_token()` refuses serialization through `to_dict()`; `Secret.from_env_var()` serializes the variable name rather than the resolved value. Neither replaces application-level logging and tracing redaction. | Configure logging and tracing redaction at the application layer; do not log connector kwargs verbatim; do not echo `repr()` of components with secrets attached. |
| **E** — None within Haystack | n/a | n/a | n/a |

---

## In-scope vs. out-of-scope: vocabulary

Reports that fall on the "Application responsibility" side of the matrix above are **out of scope** for Haystack's vulnerability program. This is the same policy expressed in `SECURITY.md`, restated in terms of the threat model:

> A finding is **in scope** if it requires no attacker-controlled Zone 1 input and demonstrates that Zone 2 (the Haystack runtime) fails to uphold one of the framework responsibilities listed above — that is, the framework itself crashes, leaks unintended state, or executes attacker-controlled code as a direct result of malformed Zone 3 or Zone 4 data.

> A finding is **out of scope** if it requires attacker-controlled Zone 1 input (a malicious pipeline definition, a malicious environment variable, a malicious tool registration), or if it asks the framework to take responsibility for a duty the matrix assigns to the application.

Examples (illustrative, not exhaustive):

- **In scope**: A converter component crashes with arbitrary code execution when reading a malformed PDF. (Zone 4 → Zone 2, framework fails to uphold "parse safely.")
- **In scope**: A pipeline `to_dict()` call emits a `Secret.from_token()` value as plaintext. (Framework fails to uphold its serialization-refusal contract.)
- **In scope**: A pipeline runtime bug allows one pipeline's components to exfiltrate state from a concurrently-running pipeline. (Framework fails to uphold isolation between operator-defined pipelines.)
- **Out of scope**: An operator loads a YAML pipeline from a public S3 bucket and the pipeline executes arbitrary code. (Attacker controlled Zone 1.)
- **Out of scope**: `LinkContentFetcher` fetches `http://169.254.169.254` when given that URL by the application. (Application's responsibility per matrix.)
- **Out of scope**: An attacker stores a prompt-injection payload in a document store and the LLM follows the injected instructions. (Application's responsibility per matrix.)
- **Out of scope**: A tool implementation executes a shell command interpolated from LLM output. (Application's responsibility per matrix.)

If you are unsure where a finding falls, contact `opensource-security@deepset.ai` before writing it up — we would rather triage a question than discard a report you spent days on.

---

## Defense-in-depth contributions are welcome

The "Application responsibility" column is not a refusal to consider security improvements at the framework layer. It is a statement about *whose duty* a control is. There is room for Haystack to make the application's job easier:

- Opt-in knobs on components (illustrative examples: a `max_response_bytes` cap on fetchers, an `allowed_schemes` allowlist) that would **default to current behavior** but make safe configuration a one-line change.
- Improved docstrings that point operators at the matrix row they should be thinking about.
- Sample pipelines in `examples/` that demonstrate the application-layer controls expected for common deployments.
- New components (classifiers, validators, sanitizers) that operators can wire into pipelines.

Contributions of this shape are welcome through normal pull requests. Frame them as operator convenience aligned with this threat model, not as the framework taking over an application duty.

---

## Updates

This document is versioned with the repository. Material changes to the trust model are announced in release notes under the `security:` or `upgrade:` tag.

For specific vulnerability reports, see [`SECURITY.md`](SECURITY.md).
