# Security Policy

## Report a Vulnerability

If you found a security vulnerability in Haystack, send a message to
[security@deepset.ai](mailto:security@deepset.ai).

In your message, please include:

1. Reproducible steps to trigger the vulnerability.
2. An explanation of what makes you think there is a vulnerability.
3. Any information you may have on active exploitations of the vulnerability (zero-day).
4. An explanation of why you believe the vulnerability is not out of scope. See the Out of Scope section below.

## Out of Scope

Haystack is a Python framework intended to run inside a trusted execution environment. It assumes that the application build on top of it has already validated and sanitized all user-supplied input before passing it to the framework. Validation and sanitization of user inputs are the responsibility of the hosting application, not Haystack.

The following classes of issues will **not** be accepted as security vulnerabilities because they can only be exploited when the surrounding application passes attacker-controlled input directly to Haystack without any prior validation:

- **Filter injection** – manipulating document store filter expressions by supplying attacker-controlled filter values. Applications must validate and sanitize filter inputs before passing them to Haystack.
- **SSRF or open redirect** – passing attacker-controlled URLs to components that fetch remote resources (e.g., URL-based document fetchers or audio transcription components). URL validation is the responsibility of the application.
- **Path traversal or arbitrary file read/write** – passing untrusted file paths or file-like objects into components that perform file I/O. The application must restrict which paths are accessible.
- **Prompt injection or prompt leakage** – manipulating natural-language content (e.g., documents, query strings, chat messages) to influence LLM behavior. These are inherent to the LLM threat landscape and must be mitigated at the application layer, via specialized Haystack pipeline components or via security-focused architectural approaches like the dual LLM pattern.
- **Denial of Service via unbounded input** – crashes or memory exhaustion caused by sending arbitrarily large or malformed payloads through an exposed API endpoint.

If you are uncertain whether a finding falls within scope, feel free to reach out before submitting a full report.

## Vulnerability Response

We'll review your report within 5 business days and we will do a preliminary analysis
to confirm that the vulnerability is plausible. Otherwise, we'll decline the report.

We won't disclose any information you share with us but we'll use it to get the issue
fixed or to coordinate a vendor response, as needed.

We'll keep you updated of the status of the issue.

Our goal is to disclose bugs as soon as possible once a user mitigation is available.
Once we get a good understanding of the vulnerability, we'll set a disclosure date after
consulting the author of the report and Haystack maintainers.
