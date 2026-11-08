#!/bin/bash
# wait_for_platform_pr.sh - Wait for the PR opened by a platform repo's Haystack bump workflow
#
# Usage: ./wait_for_platform_pr.sh <repo> <version>
# Requires: GH_TOKEN environment variable
# Output: Writes pr_url to $GITHUB_OUTPUT if set, otherwise to stdout
#
# Example:
#   ./wait_for_platform_pr.sh deepset-ai/haystack-runtime 2.99.0-rc1
#
# This script is used in the release.yml workflow after dispatching a platform repo's update
# workflow, which opens a PR from branch "bump/hs<version>".


# With the default values, we wait for 5 minutes
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"

set -euo pipefail

if [[ -z "${GH_TOKEN:-}" ]]; then
    echo "❌ GH_TOKEN must be set"
    exit 1
fi

REPO="$1"
VERSION="$2"
BRANCH="bump/hs${VERSION}"

echo "⏳ Waiting for a PR from branch ${BRANCH} on ${REPO}"

for ((i=1; i<=MAX_ATTEMPTS; i++)); do
    PR_URL=$(gh pr list -R "${REPO}" --head "${BRANCH}" --state open \
        --json url --jq '.[0].url // empty' 2>/dev/null || true)
    if [[ -n "${PR_URL}" ]]; then
        echo "✅ Found PR: ${PR_URL}"
        OUTPUT_FILE="${GITHUB_OUTPUT:-/dev/stdout}"
        echo "pr_url=${PR_URL}" >> "${OUTPUT_FILE}"
        exit 0
    fi
    echo "   Attempt $i/${MAX_ATTEMPTS}: PR not found yet..."
    sleep "${SLEEP_SECONDS}"
done

echo "❌ PR not found. Something might have failed: check https://github.com/${REPO}/actions"
exit 1
