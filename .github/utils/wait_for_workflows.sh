#!/bin/bash
# wait_for_workflows.sh - Wait for tag-triggered workflows to complete
#
# Usage: ./wait_for_workflows.sh <tag> <workflow_name1> [workflow_name2] ...
# Requires: GH_TOKEN and GITHUB_REPOSITORY environment variables
#
# Example:
#   ./wait_for_workflows.sh v2.19.0 "Project release on PyPi" "Docker image release"
#
# This script is used in the release.yml workflow to wait for the workflows triggered by a specific release tag to
# successfully complete.


# With the current values, we wait for 20 minutes
MAX_ATTEMPTS=40
SLEEP_SECONDS=30

set -euo pipefail

TAG="$1"
shift
WORKFLOWS=("$@")


# Get commit SHA from tag
TAG_SHA=$(git rev-list -n 1 "${TAG}" 2>/dev/null) || {
    echo "❌ Tag ${TAG} not found"
    exit 1
}

echo "Tag ${TAG} (commit: ${TAG_SHA:0:7})"
echo ""

wait_for_workflow() {
    local name="$1"
    echo "⏳ Waiting for: $name"

    for ((i=1; i<=MAX_ATTEMPTS; i++)); do
        jq_filter="[.workflow_runs[] | select(.head_sha == \"${TAG_SHA}\" and .name == \"${name}\")]
            | sort_by(.created_at) | last"

        result=$(gh api "repos/${GITHUB_REPOSITORY}/actions/runs" \
            --jq "$jq_filter" 2>/dev/null || echo "")

        if [[ -z "$result" ]]; then
            echo "   Attempt $i/$MAX_ATTEMPTS: not started yet..."
            sleep $SLEEP_SECONDS
            continue
        fi

        status=$(echo "$result" | jq -r '.status')
        conclusion=$(echo "$result" | jq -r '.conclusion')

        if [[ "$status" == "completed" ]]; then
            if [[ "$conclusion" == "success" ]]; then
                echo "✅ $name completed"
                return 0
            else
                echo "❌ $name failed: $conclusion"
                return 1
            fi
        fi

        echo "   Attempt $i/$MAX_ATTEMPTS: $status..."
        sleep $SLEEP_SECONDS
    done

    echo "❌ $name: timeout after $((MAX_ATTEMPTS * SLEEP_SECONDS / 60)) minutes"
    return 1
}

for workflow in "${WORKFLOWS[@]}"; do
    wait_for_workflow "$workflow" || exit 1
done

echo ""
echo "✅ All workflows completed"
