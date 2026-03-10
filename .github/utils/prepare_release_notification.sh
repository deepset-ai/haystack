#!/bin/bash
# prepare_release_notification.sh - Prepare Slack notification for release outcome
#
# Requires: VERSION, RUN_URL, HAS_FAILURE, GH_TOKEN, GITHUB_REPOSITORY
# Optional: IS_FIRST_RC, MAJOR_MINOR, GITHUB_URL, PYPI_URL, DOCKER_URL, BUMP_VERSION_PR_URL
# Output: text (via GITHUB_OUTPUT or stdout)
#
# This script is used in the release.yml workflow to prepare the notification payload
# sent to Slack after a release completes (success or failure).
# Text uses Slack mrkdwn format: *bold*, <url|label> for links.

set -euo pipefail

OUTPUT_FILE="${GITHUB_OUTPUT:-/dev/stdout}"

IS_RC="false"
[[ "${VERSION}" == *"-rc"* ]] && IS_RC="true"

if [[ "${HAS_FAILURE}" == "true" ]]; then
  {
    echo "text<<EOF"
    echo ":red_circle: Release *${VERSION}* failed"
    echo "Check workflow run for details: <${RUN_URL}|View Logs>"
    echo "EOF"
  } >> "$OUTPUT_FILE"
  exit 0
fi

# Success case

TXT=":white_check_mark: Release *${VERSION}* completed successfully"

# Add artifact URLs if available
if [[ -n "${GITHUB_URL:-}" || -n "${PYPI_URL:-}" || -n "${DOCKER_URL:-}" ]]; then
  TXT+=$'\n\n:package: *Artifacts:*'
  [[ -n "${GITHUB_URL:-}" ]] && TXT+=$'\n'"- <${GITHUB_URL}|Release notes (GitHub)>"
  [[ -n "${PYPI_URL:-}" ]] && TXT+=$'\n'"- <${PYPI_URL}|PyPI>"
  [[ -n "${DOCKER_URL:-}" ]] && TXT+=$'\n'"- <${DOCKER_URL}|Docker>"
fi

# For RCs, include link to the Tests workflow run
if [[ "${IS_RC}" == "true" ]]; then
  COMMIT_SHA=$(gh api "repos/${GITHUB_REPOSITORY}/commits/${VERSION}" --jq '.sha' 2>/dev/null || echo "")
  if [[ -n "${COMMIT_SHA}" ]]; then
    TESTS_RUN=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs?head_sha=${COMMIT_SHA}" \
      --jq '.workflow_runs[] | select(.name == "Tests") | .html_url' 2>/dev/null | head -1 || echo "")
    if [[ -n "${TESTS_RUN}" ]]; then
      TXT+=$'\n\n'":test_tube: <${TESTS_RUN}|Tests>"
    fi
  fi
fi

# For first RC, include the PRs to merge from branch-off
if [[ "${IS_FIRST_RC:-}" == "true" && -n "${BUMP_VERSION_PR_URL:-}" ]]; then
  TXT+=$'\n\n'":clipboard: *PRs to merge:*"
  TXT+=$'\n'"- <${BUMP_VERSION_PR_URL}|Bump unstable version and create unstable docs>"
fi

# For RCs, request testing from Platform Engineering
if [[ "${IS_RC}" == "true" ]]; then
  TXT+=$'\n\n'"This release is marked as a Release Candidate."
  TXT+=$'\n'"Comment on this message and tag Platform Engineering to request testing on both Platform and DC custom nodes."
fi

# For final minor releases (vX.Y.0), include the docs promotion PR
if [[ "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.0$ && -n "${MAJOR_MINOR:-}" ]]; then
  PROMOTE_DOCS_PR_URL=$(gh pr list --repo "${GITHUB_REPOSITORY}" \
    --head "promote-unstable-docs-${MAJOR_MINOR}" --json url --jq '.[0].url' 2>/dev/null || echo "")
  if [[ -n "${PROMOTE_DOCS_PR_URL}" ]]; then
    TXT+=$'\n\n'":clipboard: *PRs to merge:*"
    TXT+=$'\n'"- <${PROMOTE_DOCS_PR_URL}|Promote unstable docs>"
  fi
fi

# For final releases (not RCs), include info about pushing release notes to website
if [[ "${IS_RC}" == "false" ]]; then
  TXT+=$'\n\n'":memo: After refining and finalizing release notes, push them to Haystack website:"
  TXT+=$'\n'"\`gh workflow run push_release_notes_to_website.yml -R deepset-ai/haystack -f version=${VERSION}\`"
fi

{
  echo "text<<EOF"
  echo "${TXT}"
  echo "EOF"
} >> "$OUTPUT_FILE"
