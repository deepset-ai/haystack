#!/bin/bash
# prepare_release_notification.sh - Prepare Slack notification for release outcome
#
# Requires: VERSION, RUN_URL, HAS_FAILURE, GH_TOKEN, GITHUB_REPOSITORY
# Optional: IS_RC, IS_FIRST_RC, MAJOR_MINOR, GITHUB_URL, PYPI_URL, DOCKER_URL,
#           BUMP_VERSION_PR_URL, DC_PIPELINE_TEMPLATES_PR_URL, CUSTOM_NODES_PR_URL,
#           DC_PIPELINE_IMAGES_PR_URL, GITHUB_WORKSPACE
# Output: slack_payload.json
#
# This script is used in the release.yml workflow to prepare the notification payload
# sent to Slack after a release completes (success or failure).
# Text uses Slack mrkdwn format: *bold*, <url|label> for links.

set -euo pipefail

PAYLOAD_FILE="${GITHUB_WORKSPACE:-/tmp}/slack_payload.json"

write_payload() {
  jq -n --arg text "$TXT" '{
    text: $text,
    blocks: [{ type: "section", text: { type: "mrkdwn", text: $text } }]
  }' > "$PAYLOAD_FILE"
}

if [[ "${HAS_FAILURE}" == "true" ]]; then
  TXT=":red_circle: Release *${VERSION}* failed"
  TXT+=$'\n'"Check workflow run for details: <${RUN_URL}|View Logs>"
  write_payload
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
if [[ "${IS_RC:-}" == "true" ]]; then
  COMMIT_SHA=$(gh api "repos/${GITHUB_REPOSITORY}/commits/${VERSION}" --jq '.sha' 2>/dev/null || echo "")
  if [[ -n "${COMMIT_SHA}" ]]; then
    TESTS_RUN=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs?head_sha=${COMMIT_SHA}" \
      --jq '.workflow_runs[] | select(.name == "Tests") | .html_url' 2>/dev/null | head -1 || echo "")
    if [[ -n "${TESTS_RUN}" ]]; then
      TXT+=$'\n\n'":test_tube: <${TESTS_RUN}|Haystack Tests>"
    fi
  fi
fi

# For first RC, include the PRs to merge from branch-off
if [[ "${IS_FIRST_RC:-}" == "true" && -n "${BUMP_VERSION_PR_URL:-}" ]]; then
  TXT+=$'\n\n'":clipboard: *PRs to merge:*"
  TXT+=$'\n'"- <${BUMP_VERSION_PR_URL}|Bump unstable version and create unstable docs>"
fi

# For RCs, include Platform test PRs
if [[ "${IS_RC:-}" == "true" ]]; then
  PLATFORM_PRS=""
  [[ -n "${DC_PIPELINE_TEMPLATES_PR_URL:-}" ]] && PLATFORM_PRS+=$'\n'"- <${DC_PIPELINE_TEMPLATES_PR_URL}|dc-pipeline-templates>"
  [[ -n "${DC_CUSTOM_NODES_PR_URL:-}" ]] && PLATFORM_PRS+=$'\n'"- <${DC_CUSTOM_NODES_PR_URL}|deepset-cloud-custom-nodes>"
  [[ -n "${DC_PIPELINE_IMAGES_PR_URL:-}" ]] && PLATFORM_PRS+=$'\n'"- <${DC_PIPELINE_IMAGES_PR_URL}|dc-pipeline-images>"
  if [[ -n "${PLATFORM_PRS}" ]]; then
    TXT+=$'\n\n'":factory: *Test PRs opened on Platform:*${PLATFORM_PRS}"
  fi
fi

# For RCs, request Platform Engineering taking over testing
if [[ "${IS_RC:-}" == "true" ]]; then
  TXT+=$'\n\n'"This release is marked as a Release Candidate."
  TXT+=$'\n'"Notify #deepset-platform-engineering channel on Slack that the RC is available"
  TXT+=" and that Platform tests pass/fail, linking the PRs opened on the Platform repositories."
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
if [[ "${IS_RC:-}" != "true" ]]; then
  TXT+=$'\n\n'":memo: After refining and finalizing release notes, push them to Haystack website:"
  TXT+=$'\n'"\`gh workflow run push_release_notes_to_website.yml -R deepset-ai/haystack -f version=${VERSION}\`"
fi

write_payload
