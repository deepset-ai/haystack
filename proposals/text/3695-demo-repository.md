- Start Date: 2022-12-12
- Proposal PR: https://github.com/deepset-ai/haystack/pull/3695
- Github Issue: (if available, link the issue containing the original request for this change)

# Summary

A new git repository is created to host NLP applications showcasing Haystack's features.

# Basic example

A git repository was already created on Github as an example:

https://github.com/deepset-ai/haystack-demos

# Motivation

NLP applications showcasing Haystack's capabilities can be an invaluable learning resource
for its users, but at this moment we don't fully take advantage of the only one demo we have
as a documentation source.

This proposal aims at overcoming that limitation in two ways:
- Define better requirements for a demo application so that users can learn from it.
- Make it easier to add more demo applications showcasing Haystack.

# Detailed design

Every demo has a descriptive name that will be used as its identifier.

Every demo lives in a dedicated folder named after its identifier at the root of the repo, and
provides all the resources needed to understand the code, run the application locally or deploy it
remotely on a server.

Every demo provides a README.md file containing the following information:
- A brief description of the application and what's its goal.
- Explicit mention of which NLP use case is implemented: for example "QA", or "Document Retrieval".
- Detailed instructions about how to run the application locally.
- Any hardware requirement and the limitations when not provided (for example, a GPU device).
- How to modify and test the code, and how to contribute changes.

The code of a demo application should be tested whenever possible, and at least some of the
tests should be able to run in the repo CI system. Every demo has a dedicated Workflow defined
in a file named after its identifier. The workflow runs only when files in the demo folder are
modified.

In case the CI is needed for continuous deployment, or for building artifacts, a demo can have
more than one workflow file defined, named after its identifier plus a descriptive suffix, for
example: `my_demo_identifier.yml` for tests, `my_demo_identifier_docker.yml` for building a
Docker image, `my_demo_identifier_deploy.yml` for continuous delivery.

# Drawbacks

- The code of the existing demo would be removed from Haystack and potentially become harder to
  find for existing contributors.
- The proposed design dictates a list of new requirements for a demo that will take time to
  implement.

# Alternatives

- Leave things as they are
- Implement the design proposal to a subfolder of Haystack's git repository

# Adoption strategy

Adoption will be mostly driven by communicating the changes to the community and monitoring the
traffic in the new Github repository: interacting with the existing demo will not be affected
but accessing the code would.

# How we teach this

- A link to the demo repository will be added to the web page of the [demo itself](https://haystack-demo.deepset.ai/).
- Haystack's README and documentation will mention where to find the code for the demos.
- [Haystack Home](https://haystack.deepset.ai) will host a whole section dedicated to Haystack demos
  (detailing the aforementioned section is out of scope for this proposal).

# Unresolved questions

N/A.
