# Security Policy

## Report a Vulnerability

If you have found a security vulnerability in Haystack, please report via email to
[opensource-security@deepset.ai](mailto:opensource-security@deepset.ai).

In your message, please include:

1. Reproducible steps to trigger the vulnerability.
2. An explanation of what makes you think there is a vulnerability.
3. Any information you may have on active exploitations of the vulnerability (zero-day).
4. An explanation of why you believe the vulnerability is not out of scope. See the Out of Scope section below.

We encourage reports that are meaningful, high-impact, and reviewed by a human before submission. Fully automated or AI-generated reports submitted without human review and validation are unlikely to meet this bar and risk being declined.

## Out of Scope

Haystack is a framework intended to run inside a trusted execution environment. It assumes that the application built with it has already validated and sanitized user-supplied input before passing it to the framework. Validation and sanitization of input, for example URLs, file paths, filter expressions, and queries, are the responsibility of the application, not Haystack.

Any vulnerability that can only be triggered by passing unsanitized, attacker-controlled input to Haystack is considered out of scope. This reflects a conscious design decision after evaluating the trade-offs and risks: as a framework, Haystack cannot and should not enforce input validation on behalf of every application that uses it.

The following areas have been deliberately scoped out below and we ask that you read them carefully before submitting.

### Pipeline Serialization

Haystack pipelines in YAML are a serialization format for executable pipelines. `Pipeline.loads()`, `deserialize_callable()`, and `import_class_by_name()` are designed to load user-defined components, filters, and callables. They intentionally support dynamic imports at runtime to enable extensibility.

**Loading a pipeline from an untrusted source is unsafe by design.** This is not a hidden weakness but the expected consequence of a system that lets users bring their own code. The security responsibility lies with the operator: pipeline definitions must be treated as code, stored and transmitted with the same controls applied to source code, and never loaded from untrusted or user-controlled input without review.
Reports that demonstrate, for example, `import_class_by_name()` can import arbitrary modules given a pipeline that an operator chose to load are out of scope.

However, if you find a way to achieve arbitrary code execution that does *not* rely on an operator loading an untrusted pipeline (for example, a memory-safety issue in the parser itself, or a class of pipelines that triggers unintended behaviour in the Haystack runtime), that finding is in scope.

### SSRF via URL-fetching Components (e.g. `LinkContentFetcher`)

Components such as `LinkContentFetcher` accept URLs and fetch their content. They do not validate or restrict which hosts or IP ranges are reachable. This is intentional: restricting network access at the framework layer would break legitimate use cases and would provide only a weak, bypassable control anyway. Just as developers would not expect the [requests](https://github.com/psf/requests) library to enforce SSRF protections, Haystack components that perform HTTP requests do not gate which hosts are reachable by default.

**SSRF prevention is the responsibility of the application and the network layer.** The expected mitigations are: resolving the requested hostname before fetching and rejecting requests whose resolved IP falls within private or reserved CIDR ranges, enforcing egress firewall rules at the infrastructure level, and running Haystack in a network-isolated environment where internal services are not reachable from the component's runtime context.

We are aware that demonstrating an open fetch to `http://169.254.169.254` or similar metadata endpoints is straightforward. Reports that do so without the application-layer controls described above in place are out of scope. If you find a bypass of a network control that *is* in place, or a vulnerability in how Haystack processes the fetched content (for example, a parsing issue that leads to unintended behaviour), that finding is in scope.

### Prompt Injection via Documents or Metadata

Haystack passes documents, metadata, and other retrieved content into prompt templates at runtime. If an attacker can store prompt-injection payloads in a document store or retrieval source that a Haystack pipeline later reads and inserts into a prompt, the LLM may follow those instructions.

**Prompt injection detection and mitigation are the responsibility of the application.** Haystack provides building blocks, including classifier components and documented patterns, for application developers to implement prompt injection defences. Without such defences in place, retrieved content can reach the LLM unfiltered.

We are aware that crafting a document containing `Ignore previous instructions...` and showing it surfaces in a generated response is straightforward to demonstrate. Reports of that form are out of scope. If you find a vulnerability in a Haystack component that *bypasses* a defence a developer has wired into their pipeline, or causes unintended behaviour outside of the LLM interaction (for example, exfiltrating data through Haystack's own APIs), that finding may be in scope.

---

If you are uncertain whether a finding falls within scope, feel free to reach out before submitting a full report.

## Vulnerability Response

We aim to review your report within 5 business days where we do a preliminary analysis
to confirm that the vulnerability is plausible. Otherwise, we'll decline the report.

We won't disclose any information you share with us but we'll use it to get the issue
fixed or to coordinate a vendor response, as needed.

We'll keep you updated of the status of the issue.

Our goal is to disclose bugs as soon as possible once a user mitigation is available.
Once we get a good understanding of the vulnerability, we'll set a disclosure date after consulting the author of the report and Haystack maintainers.
