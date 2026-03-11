# Security Policy

## Report a Vulnerability

If you found a security vulnerability in Haystack, send a message to
[security@deepset.ai](mailto:security@deepset.ai).

In your message, please include:

1. Reproducible steps to trigger the vulnerability.
2. An explanation of what makes you think there is a vulnerability.
3. Any information you may have on active exploitations of the vulnerability (zero-day).
4. An explanation of why you believe the vulnerability is not out of scope. See the Out of Scope section below.

We encourage reports that are meaningful, high-impact, and reviewed by a human before submission. Fully automated or AI-generated reports submitted without human review and validation are unlikely to meet this bar and risk being declined.

## Out of Scope

Haystack is a framework intended to run inside a trusted execution environment. It assumes that the application built with it has already validated and sanitized user-supplied input before passing it to the framework. Validation and sanitization of input, for example URLs, file paths, filter expressions, and queries, are the responsibility of the application, not Haystack.

Any vulnerability that can only be triggered by passing unsanitized, attacker-controlled input to Haystack is considered out of scope. This reflects a conscious design decision after evaluating the trade-offs and risks: as a framework, Haystack cannot and should not enforce input validation on behalf of every application that uses it.

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
