#!/usr/bin/env bash
# Run the course public tests against this local folder (fast iteration).
# Usage: ./run_public_tests.sh <path-to-course-repo>/projects/project1/grading_tests
# The "github link format" test always fails with a local path; that is expected.
set -euo pipefail
TESTS_DIR="${1:-../grading_tests}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$TESTS_DIR"
pytest --github_link "$REPO_DIR" .
