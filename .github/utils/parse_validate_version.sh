#!/bin/bash
# parse_validate_version.sh - Parse and validate version for release
#
# Usage: ./parse_validate_version.sh <version>
# Output: Writes to $GITHUB_OUTPUT if set, otherwise to stdout
#
# Example:
#   ./parse_validate_version.sh v2.99.0-rc1
#
# This script is used in the release.yml workflow to parse and validate the version to be released.
# Covers several checks to prevent accidental releases of incorrect versions.

set -euo pipefail

# --- Helpers ---

fail() {
    echo ""
    echo -e "❌ $1"
    echo ""
    exit 1
}

ok() {
    echo "✅ $1"
}

tag_exists() {
    git tag -l "$1" | grep -q "^$1$"
}

branch_exists() {
    git ls-remote --heads origin "$1" | grep -q "$1"
}

# --- Parse and validate version ---

VERSION="${1#v}"  # Strip 'v' prefix

echo ""
echo "ℹ️  Validating: ${1}"
echo ""

if [[ ! "${VERSION}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-rc([0-9]+))?$ ]]; then
    fail "Invalid version format: $1\n\n"\
"Expected format: vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCH-rcN\n"\
"Examples: v2.99.0-rc1, v2.99.0, v2.99.1-rc1"
fi
ok "Version format is valid"

MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"
PATCH="${BASH_REMATCH[3]}"
RC_NUM="${BASH_REMATCH[5]:-0}"

if [[ "${RC_NUM}" == "0" && "${VERSION}" == *"-rc0" ]]; then
    fail "Cannot release rc0\n\n"\
"rc0 is an internal marker created automatically during branch-off.\n"\
"Release candidates start at rc1."
fi

MAJOR_MINOR="${MAJOR}.${MINOR}"
RELEASE_BRANCH="v${MAJOR_MINOR}.x"
TAG="v${VERSION}"

IS_FIRST_RC="false"
if [[ "${PATCH}" == "0" && "${RC_NUM}" == "1" ]]; then
    IS_FIRST_RC="true"
fi

# 1. Tag must not already exist
if tag_exists "${TAG}"; then
    fail "Version ${TAG} was already released\n\n"\
"Each version can only be released once.\n"\
"To publish changes, release the next RC or patch version."
fi
ok "Tag ${TAG} does not exist"

# 2. Checks based on release type
if [[ "${IS_FIRST_RC}" == "true" ]]; then
    # First RC of minor: branch must NOT exist yet
    if branch_exists "${RELEASE_BRANCH}"; then
        fail "Branch ${RELEASE_BRANCH} already exists\n\n"\
"The first RC of a minor (e.g., v${MAJOR_MINOR}.0-rc1) creates the release branch.\n"\
"Since the branch exists, this minor was likely already started.\n"\
"Did you mean to release the next RC (rc2, rc3...) or a patch (v${MAJOR_MINOR}.1-rc1)?"
    fi
    ok "Branch ${RELEASE_BRANCH} does not exist"

    # First RC of minor: VERSION.txt must contain rc0
    EXPECTED="${MAJOR_MINOR}.0-rc0"
    ACTUAL=$(cat VERSION.txt)
    if [[ "${ACTUAL}" != "${EXPECTED}" ]]; then
        ACTUAL_MINOR=$(echo "${ACTUAL}" | cut -d. -f1,2)
        fail "Cannot release v${MAJOR_MINOR}.0-rc1 from this branch\n\n"\
"The main branch is prepared for version ${ACTUAL_MINOR}, not ${MAJOR_MINOR}.\n"\
"Check that you're releasing the correct version."
    fi
    ok "VERSION.txt = ${EXPECTED}"

else
    # Not first RC: branch MUST exist
    if ! branch_exists "${RELEASE_BRANCH}"; then
        if [[ "${PATCH}" == "0" ]]; then
            fail "Branch ${RELEASE_BRANCH} does not exist\n\n"\
"For subsequent RCs (rc2, rc3...), the release branch must already exist.\n"\
"Release the first RC (v${MAJOR_MINOR}.0-rc1) first to create the branch."
        else
            fail "Branch ${RELEASE_BRANCH} does not exist\n\n"\
"For patch releases, the release branch must already exist.\n"\
"The minor version (v${MAJOR_MINOR}.0) must be released before any patches."
        fi
    fi
    ok "Branch ${RELEASE_BRANCH} exists"

    # Subsequent RC (rc2, rc3...): previous RC must exist
    if [[ "${RC_NUM}" -gt 1 ]]; then
        PREV_RC_NUM=$((RC_NUM - 1))
        PREV_TAG="v${MAJOR_MINOR}.${PATCH}-rc${PREV_RC_NUM}"
        if ! tag_exists "${PREV_TAG}"; then
            fail "Cannot release v${MAJOR_MINOR}.${PATCH}-rc${RC_NUM}\n\n"\
"Previous RC (${PREV_TAG}) was not found.\n"\
"RC versions must be sequential. Release rc${PREV_RC_NUM} first."
        fi
        ok "Previous tag ${PREV_TAG} exists"
    fi

    # Final release: at least one RC must exist
    if [[ "${RC_NUM}" == "0" ]]; then
        RC_TAGS=$(git tag -l "v${MAJOR_MINOR}.${PATCH}-rc*" | grep -v "\-rc0$" || true)
        if [[ -z "${RC_TAGS}" ]]; then
            fail "Cannot release stable version v${MAJOR_MINOR}.${PATCH}\n\n"\
"No release candidate found for this version.\n"\
"Stable releases require at least one RC first (e.g., v${MAJOR_MINOR}.${PATCH}-rc1)."
        fi
        LAST_RC=$(echo "${RC_TAGS}" | sort -V | tail -n1)
        ok "Found RC: ${LAST_RC}"

        # Check Tests workflow passed (only if credentials available)
        if [[ -n "${GH_TOKEN:-}" && -n "${GITHUB_REPOSITORY:-}" ]]; then
            RC_SHA=$(git rev-list -n 1 "${LAST_RC}")
            RESULT=$(gh api "/repos/${GITHUB_REPOSITORY}/actions/runs?head_sha=${RC_SHA}&status=success" \
                --jq '.workflow_runs[] | select(.name == "Tests") | .conclusion' 2>/dev/null || true)
            if [[ -z "${RESULT}" ]]; then
                fail "Cannot release stable version v${MAJOR_MINOR}.${PATCH}\n\n"\
"Tests did not pass on the last RC (${LAST_RC}).\n"\
"Wait for tests to complete, or release a new RC with fixes."
            fi
            ok "Tests passed on ${LAST_RC}"
        fi
    fi
fi

echo ""
ok "All validations passed!"
echo ""

# --- Output to GITHUB_OUTPUT (or stdout for local testing) ---

OUTPUT_FILE="${GITHUB_OUTPUT:-/dev/stdout}"

{
    echo "version=${VERSION}"
    echo "major_minor=${MAJOR_MINOR}"
    echo "release_branch=${RELEASE_BRANCH}"
    echo "is_first_rc=${IS_FIRST_RC}"
} >> "${OUTPUT_FILE}"
