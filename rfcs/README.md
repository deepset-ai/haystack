# Haystack RFCs

Most of the changes to Haystack, including bug fixes and small improvements,
are implemented through the normal Pull Request workflow, according to our
[contribution guidelines](../CONTRIBUTING.md).

Some changes though are "substantial", and those will be put through a bit
of a design process in order to produce a consensus before investing time
into the actual implementation of a new feature or a deep refactoring.

The RFC (request for comments) design process is intended to provide a
consistent and controlled path for such changes to Haystack.

We will apply the same level of rigour both to core developers RFCs and
Community RFCs. The primary difference between them is in the design phase:
core developers RFCs tend to be submitted at the end of the design process
whereas the Community RFCs tend to be submitted at the beginning as a way
to kickstart it.

## When you need to follow this process

You need to follow this process if you intend to make "substantial" changes to
Haystack, `rest_api` or `ui`, or the RFC process itself. What constitutes a
"substantial" change is evolving based on community norms and varies depending
on what part of the project you are proposing to change, but may include the following:

- A new feature that creates new API surface areas.
- A new component (Nodes, Pipelines, Document Stores).
- The removal of features that already shipped in the current minor version.
- A deep refactoring that would require new tests or introduce new dependencies
- A change complex enough that would require multiple steps to be delivered

Some changes do not require an RFC:

- Rephrasing, reorganizing, or otherwise "changing shape does not change meaning".
- Addition and removal of warnings or other error messages.
- Additions only likely to be noticed by other contributors, invisible to Haystack users.

In any case, the core developers might politely ask you to submit an RFC before merging
a new feature when they see fit.

## Before creating an RFC

A hastily-proposed RFC can hurt its chances of acceptance. Low quality proposals, proposals
for previously-rejected features, or those that don't fit into the near-term
[roadmap](https://github.com/orgs/deepset-ai/projects/3), may be rejected, which can be
demotivating for the new contributor. Laying some groundwork ahead of the RFC can make the
process smoother.

Although there is no single way to prepare for submitting an RFC, it is generally a good idea
to pursue feedback from other project developers beforehand, to ascertain that the RFC may be
desirable; having a consistent impact on the project requires concerted effort toward
consensus-building.

Preparations for writing and submitting an RFC include talking the idea over on our official
[Discord server](https://haystack.deepset.ai/community/join), discussing the topic on a Github
issue or discussion in the [Haystack repository](https://github.com/deepset-ai/haystack).

## What the process is

In short, to get a major feature added to Haystack, one usually first gets the RFC merged into
the RFC repo as a markdown file. At that point the RFC is 'active' and may be implemented with
the goal of eventual inclusion into the Haystack codebase.

- Copy `0000-template.md` to `text/0000-my-feature.md`, where 'my-feature' is descriptive. Don't assign an RFC number yet.
- Fill in the RFC. Put care into the details: RFCs that do not present convincing motivation, demonstrate understanding of the impact of the design, or are disingenuous about the drawbacks or alternatives tend to be poorly-received.
- Submit a pull request. As a pull request the RFC will receive design feedback from the larger community, and the author should be prepared to revise it in response.
- Rename the file using the PR number, e.g. from `text/0000-my-feature.md` to `text/4242-my-feature.md`.
- Build consensus and integrate feedback. RFCs that have broad support are much more likely to make progress than those that don't receive any comments.
- Eventually, the core developers will decide whether the RFC is a candidate for inclusion in Haystack. Note that a team review may take a long time, and we suggest that you ask members of the community to review it first.
- RFCs that are candidates for inclusion in Haystack will enter a "final comment period" lasting 3 calendar days. The beginning of this period will be signaled with a comment and label on the RFCs pull request.
- An RFC can be modified based upon feedback from the core developers and community. Significant modifications may trigger a new final comment period.
- An RFC may be rejected by the core developers after public discussion has settled and comments have been made summarizing the rationale for rejection. A core developer should then close the RFCs associated pull request.
- An RFC may be accepted at the close of its final comment period. A core developer will merge the RFCs associated pull request, at which point the RFC will become 'active'.

## The RFC lifecycle

Once an RFC becomes active, then authors may implement it and submit the feature as a pull request
to the Haystack repo. Becoming 'active' is not a rubber stamp, and in particular still does not mean
the feature will ultimately be merged; it does mean that the core team has agreed to it in principle
and are amenable to merging it.

Furthermore, the fact that a given RFC has been accepted and is 'active' implies nothing about what
priority is assigned to its implementation, nor whether anybody is currently working on it.

Modifications to active RFCs can be done in followup PRs. We strive to write each RFC in a manner that
it will reflect the final design of the feature; but the nature of the process means that we cannot
expect every merged RFC to actually reflect what the end result will be at the time of the next release;
therefore we try to keep each RFC document somewhat in sync with the feature as planned, tracking such
changes via followup pull requests to the document.

The author of an RFC is not obligated to implement it. Of course, the RFC author (like any other developer)
is welcome to post an implementation for review after the RFC has been accepted.

## Inspiration

React's RFC process owes its inspiration to the [React](https://github.com/reactjs/rfcs) and
[Rust](https://github.com/rust-lang/rfcs) RFC processes. We're open to change it if needed.