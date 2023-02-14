# Haystack proposals design process

Most of the changes to Haystack, including bug fixes and small improvements,
are implemented through the normal Pull Request workflow, according to our
[contribution guidelines](../CONTRIBUTING.md).

Some changes, though, are "substantial", and these are the ones we want to put through a bit
of a design process to make sure we're all on the same page before we invest the time
into the actual implementation of a new feature or a deep refactoring.

We've introduced the "Proposals design process" to provide a
consistent and controlled path for such changes to Haystack.

We will apply the same level of rigor to both core developers' and
Community's proposals. The primary difference between them is in the design phase:
core developers proposals tend to be submitted at the end of the design process
whereas the Community ones tend to be submitted at the beginning, as a way
to kickstart it.

## When do I follow the process?

Follow the process if you intend to make "substantial" changes to Haystack, `rest_api` or the process itself.  What is
defined as a "substantial" change is evolving based on community norms and on what part of the project you are proposing
to change, but it may include the following:

- A new feature that creates new API surface areas.
- A new component (Nodes, Pipelines, Document Stores).
- Removing features that already shipped in the current minor version.
- A deep refactoring that would require new tests or introduce new dependencies.
- A change that's complex enough to require multiple steps to be delivered.

Some changes don't require a proposal, for example:

- Minor bug fixes.
- Rephrasing, reorganizing, or otherwise "changing shape does not change meaning".
- Addition and removal of warnings or other error messages.
- Additions only likely to be noticed by other contributors, invisible to Haystack users.

In any case, the core developers might politely ask you to submit a proposal before merging
a new feature when they see fit.

## Before creating a proposal

Laying some groundwork ahead of the proposal can make the process smoother.

Although there is no single way to prepare for submitting a proposal, it is generally a good idea
to collect feedback from other project developers first, to make sure that the change is
is actually needed. As we're an open source community where everyone can impact the project, we all
need to make an effort to build consensus.

When you're preparing for writing and submitting a proposal, talk the idea over on our official
[Discord server](https://haystack.deepset.ai/community/join) and in a Github
issue or discussion in the [Haystack repository](https://github.com/deepset-ai/haystack).

## The process

To get a major feature added to Haystack, you first merge the proposal into the Haystack repo as a Markdown file.
At that point, the proposal can be implemented and eventually included into the codebase.

There are several people involved in the process:
- **Decision Driver**: the person creating the proposal. If the Decision Driver is not a core contributor themselves,
  one will be assigned to the PR and will take care of facilitating the process.
- **Input Givers**: anybody reviewing or commenting the PR.
- **Approvers**: the core contributors approving the PR.

During its lifecycle, a proposal can transition between the following states:
- **Review**: proposal is getting feedback.
- **Final Comment**: proposal received approval from 3 core contributors; this state must be kept for a grace period of
  3 calendar days.
- **Active**: proposal was approved and merged and can be implemented if not already.
- **Stale**: proposal didn't get any update in the last 30 days and will be closed after a grace period of 10 days.
- **Rejected**: proposal was actively rejected and the reasons explained.


To create a proposal:

1. Copy `0000-template.md` to `text/0000-my-feature.md`, where 'my-feature' is a descriptive name of the feature you're
   proposing. Don't assign an identification number yet.
2. Fill in the proposal. Pay attention to details. Proposals that present convincing motivation,
   demonstrate an understanding of the feature impact, and honestly present the drawbacks and
   alternatives tend to be received well.
3. Submit a pull request. This ensures the document receives design feedback from a larger community,
   and as the Decision Driver, you should be prepared to revise it in response.
4. Rename the file using the PR number, for example from `text/0000-my-feature.md` to `text/4242-my-feature.md`. The
   proposal is now in **Review** state.
5. Reach an agreement with the Input Givers and integrate the feedback you got. Proposals that have broad support are
   much more likely to make progress than those that don't receive any comments.
6. Now it's time for the Approvers to decide whether the proposal is a candidate for inclusion in Haystack. Note that a
   review from the core contributors may take a long time, and getting early feedback from members of the Community can
   ease the process.
7. When the proposal enters the **Final Comment** state (see above), the PR will be marked accordingly, entering a
   grace period lasting 3 calendar days during which a proposal can be modified based on feedback from core developers
   or the Community. Big changes may trigger a new final comment period.
8. Approvers may reject a proposal once the public discussion and adding comments are over, adding the reason for
   rejection. A core developer then closes the related PR. The proposal gets the **Rejected** state.
9. When the final comment period ends, the PR is merged and the proposal becomes **Active**.

## What happens next

Once a proposal becomes active, the authors are free to implement it and submit the feature as one or more pull
requests to the Haystack repo. Becoming 'active' is not a rubber stamp, and in particular still doesn't
mean the feature will ultimately be merged; it does mean that the core team has agreed to it in
principle and is open to merging it if the implementation reflects the contents of the proposal.

The fact that a given proposal has been accepted and is 'active' doesn't imply it has a priority assigned or somebody's
currently working on it.

To change an active proposal, open follow-up PRs. Our goal is to write each proposal so that
it reflects the final design of the feature, but the nature of the process means that we cannot
expect every merged proposal to actually reflect what the end result will be at the time of the next release.
That's why we try to keep each proposal document somewhat in sync with the feature as planned, tracking such
changes through follow-up pull requests to the document.

As the author of a proposal, you're not obligated to implement it. Of course, the author (like any other developer)
is welcome to post an implementation for review after the proposal has been accepted.

## Inspiration

Haystack's proposals design process process owes its inspiration to the [React](https://github.com/reactjs/rfcs) and
[Rust](https://github.com/rust-lang/rfcs) RFC processes. We're open to changing it if needed.
