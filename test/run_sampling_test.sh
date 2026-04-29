#!/bin/bash

# Backward-compatible entry point for the legacy direct sampling smoke test.
# The upstream main branch removed this workflow from the current test matrix,
# but keeping this wrapper preserves the old command for users and reviewers.

bash test/run_sampling_legacy_test.sh
