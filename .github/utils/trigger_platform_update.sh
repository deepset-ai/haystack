#!/bin/bash
# trigger_platform_update.sh - Trigger a platform repo's Haystack bump workflow and wait for the PR
#
# Usage: ./trigger_platform_update.sh <repo> <workflow_file> <version>
# Requires: GH_TOKEN environment variable with actions:write on <repo>
# Output: Writes pr_url to $GITHUB_OUTPUT if set, otherwise to stdout.
#         pr_url is empty when the version was already pinned and the workflow skipped the PR.
#
# Example:
#   ./trigger_platform_update.sh deepset-ai/haystack-runtime update-package-version.yaml 2.99.0-rc1
#
# This script is used in the release.yml workflow to dispatch a platform repo's update workflow,
# which bumps the Haystack pin in that repo and opens a PR from branch "bump/hs<version>".


# With the default values, we wait up to 30 seconds for the dispatched run to appear
# and up to 5 minutes for it to complete
FIND_MAX_ATTEMPTS="${FIND_MAX_ATTEMPTS:-6}"
FIND_SLEEP_SECONDS="${FIND_SLEEP_SECONDS:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"

set -euo pipefail

if [[ -z "${GH_TOKEN:-}" ]]; then
    echo "❌ GH_TOKEN must be set"
    exit 1
fi

REPO="$1"
WORKFLOW_FILE="$2"
VERSION="$3"

# --- Dispatch the update workflow ---

TRIGGER_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gh workflow run "${WORKFLOW_FILE}" -R "${REPO}" -f haystack_version="${VERSION}"
echo "✅ Dispatched ${WORKFLOW_FILE} on ${REPO} with haystack_version=${VERSION}"

# --- Find the dispatched run (workflow_dispatch returns no run id) ---

echo "⏳ Waiting for the run to appear"
RUN_ID=""
for ((i=1; i<=FIND_MAX_ATTEMPTS; i++)); do
    sleep "${FIND_SLEEP_SECONDS}"
    RUN_ID=$(gh run list -R "${REPO}" --workflow="${WORKFLOW_FILE}" \
        --created ">=${TRIGGER_TIME}" --json databaseId \
        --jq '.[0].databaseId // empty' 2>/dev/null || true)
    [[ -n "${RUN_ID}" ]] && break
    echo "   Attempt $i/${FIND_MAX_ATTEMPTS}: not started yet..."
done

if [[ -z "${RUN_ID}" ]]; then
    echo "❌ Dispatched run never appeared on ${REPO}"
    exit 1
fi
echo "✅ Found run: https://github.com/${REPO}/actions/runs/${RUN_ID}"

# --- Wait for the run to complete ---

echo "⏳ Waiting for the run to complete"
STATUS=""
CONCLUSION=""
for ((i=1; i<=MAX_ATTEMPTS; i++)); do
    result=$(gh run view "${RUN_ID}" -R "${REPO}" --json status,conclusion 2>/dev/null || echo "")
    if [[ -n "${result}" ]]; then
        STATUS=$(echo "${result}" | jq -r '.status')
        CONCLUSION=$(echo "${result}" | jq -r '.conclusion')
        [[ "${STATUS}" == "completed" ]] && break
    fi
    echo "   Attempt $i/${MAX_ATTEMPTS}: ${STATUS:-unknown}..."
    sleep "${SLEEP_SECONDS}"
done

if [[ "${STATUS}" != "completed" ]]; then
    echo "❌ Run did not complete within $((MAX_ATTEMPTS * SLEEP_SECONDS / 60)) minutes"
    exit 1
fi
if [[ "${CONCLUSION}" != "success" ]]; then
    echo "❌ Run failed: ${CONCLUSION}"
    exit 1
fi
echo "✅ Run completed successfully"

# --- Look up the PR ---

# The branch name is version-specific, so an open PR on it is the right one
# even when a re-run updated an existing PR instead of creating it
PR_URL=$(gh pr list -R "${REPO}" --head "bump/hs${VERSION}" --state open \
    --json url --jq '.[0].url // empty')

if [[ -n "${PR_URL}" ]]; then
    echo "✅ Found PR: ${PR_URL}"
else
    echo "ℹ️  No PR opened: ${VERSION} is already pinned on ${REPO}"
fi

# --- Output to GITHUB_OUTPUT (or stdout for local testing) ---

OUTPUT_FILE="${GITHUB_OUTPUT:-/dev/stdout}"
echo "pr_url=${PR_URL}" >> "${OUTPUT_FILE}"
