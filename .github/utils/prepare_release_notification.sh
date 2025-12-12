#!/bin/bash
# prepare_release_notification.sh - Prepare Datadog notification for release outcome
#
# Requires: VERSION, RUN_URL, HAS_FAILURE, GH_TOKEN, GITHUB_REPOSITORY
# Optional: IS_FIRST_RC, MAJOR_MINOR, GITHUB_URL, PYPI_URL, DOCKER_URL, BUMP_VERSION_PR_URL
# Output: alert_type, title, event_text, released_rc (via GITHUB_OUTPUT or stdout)
#
# This script is used in the release.yml workflow to prepare the notification payload
# sent to Datadog after a release completes (success or failure).

set -euo pipefail

OUTPUT_FILE="${GITHUB_OUTPUT:-/dev/stdout}"

IS_RC="false"
[[ "${VERSION}" == *"-rc"* ]] && IS_RC="true"

if [[ "${HAS_FAILURE}" == "true" ]]; then
  {
    echo "alert_type=error"
    echo "title=Release ${VERSION} failed"
    echo "event_text=Release ${VERSION} failed. Check workflow run for details: ${RUN_URL}"
    echo "released_rc=false"
  } >> "$OUTPUT_FILE"
  exit 0
fi

# Success case

TXT="Release ${VERSION} completed successfully\n"

# Add artifact URLs if available
if [[ -n "${GITHUB_URL:-}" || -n "${PYPI_URL:-}" || -n "${DOCKER_URL:-}" ]]; then
  TXT+="\n📦 Artifacts:\n"
  [[ -n "${GITHUB_URL:-}" ]] && TXT+="- Release notes (GitHub): ${GITHUB_URL}\n"
  [[ -n "${PYPI_URL:-}" ]] && TXT+="- PyPI: ${PYPI_URL}\n"
  [[ -n "${DOCKER_URL:-}" ]] && TXT+="- Docker: ${DOCKER_URL}\n"
fi

# For RCs, include link to the Tests workflow run
if [[ "${IS_RC}" == "true" ]]; then
  COMMIT_SHA=$(gh api "repos/${GITHUB_REPOSITORY}/commits/${VERSION}" --jq '.sha' 2>/dev/null || echo "")
  if [[ -n "${COMMIT_SHA}" ]]; then
    TESTS_RUN=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs?head_sha=${COMMIT_SHA}" \
      --jq '.workflow_runs[] | select(.name == "Tests") | .html_url' 2>/dev/null | head -1 || echo "")
    if [[ -n "${TESTS_RUN}" ]]; then
      TXT+="\n🧪 Tests: ${TESTS_RUN}\n"
    fi
  fi
fi

# For first RC, include the PRs to merge from branch-off
if [[ "${IS_FIRST_RC:-}" == "true" && -n "${BUMP_VERSION_PR_URL:-}" ]]; then
  TXT+="\n📋 PRs to merge:\n"
  TXT+="- Bump unstable version and create unstable docs: ${BUMP_VERSION_PR_URL}\n"
fi

# For final minor releases (vX.Y.0), include the docs promotion PR
if [[ "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.0$ && -n "${MAJOR_MINOR:-}" ]]; then
  PROMOTE_DOCS_PR_URL=$(gh pr list --repo "${GITHUB_REPOSITORY}" \
    --head "promote-unstable-docs-${MAJOR_MINOR}" --json url --jq '.[0].url' 2>/dev/null || echo "")
  if [[ -n "${PROMOTE_DOCS_PR_URL}" ]]; then
    TXT+="\n📋 PRs to merge:\n"
    TXT+="- Promote unstable docs: ${PROMOTE_DOCS_PR_URL}\n"
  fi
fi

# For final releases (not RCs), include info about pushing release notes to website
if [[ "${IS_RC}" == "false" ]]; then
  TXT+="\n📝 After refining and finalizing release notes, push them to Haystack website:\n"
  TXT+="gh workflow run push_release_notes_to_website.yml -R deepset-ai/haystack -f version=${VERSION}\n"
fi

{
  echo "alert_type=success"
  echo "title=Release ${VERSION} completed successfully"
  echo "event_text=${TXT}"
  echo "released_rc=${IS_RC}"
} >> "$OUTPUT_FILE"
