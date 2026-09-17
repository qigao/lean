#!/usr/bin/env bash
set -euo pipefail

timeout --kill-after=10s 240s lake env lean \
  -DmaxErrors=1 -DstderrAsMessages=false \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
