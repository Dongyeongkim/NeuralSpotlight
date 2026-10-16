#!/bin/bash
# Launches build.noindex/NeuralSpotlight.app against the testbed with a scratch index, models folder and settings under
# .dev/ and a separate defaults suite: never real folders, never the real index, model or settings. Extra launch
# arguments pass through, e.g.
#   scripts/dev-run.sh -debugQuery "zebra" -debugSnapshot /tmp/panel.png -debugQuit YES
# ROOTS=a:b:c overrides the folders (keep them inside the testbed); ROOTS= (empty) launches with none, as a fresh
# install would. INDEX, MODELS and SUITE override the rest; APP=/Applications/NeuralSpotlight.app runs an installed copy the
# same way. The app's log is <index>/app.log.
set -euo pipefail
cd "$(dirname "$0")/.."
TB="${TESTBED:-$HOME/NeuralSpotlightTestbed}"
INDEX="${INDEX:-$PWD/.dev/app-index}"
MODELS="${MODELS:-$PWD/.dev/models}"
SUITE="${SUITE:-com.neuralspotlight.dev}"

# Use the model manifest rather than copying an obsolete runtime's files. The app resumes any missing Core ML
# downloads itself; to provision before launch: .build/debug/neuralspotlight download-model --to .dev/models
mkdir -p "$MODELS"

# Only an earlier run of this same copy is stopped: an installed NeuralSpotlight in use keeps running.
APP="${APP:-build.noindex/NeuralSpotlight.app}"
EXE="$(cd "$APP" && pwd -P)/Contents/MacOS/NeuralSpotlight"
for pid in $(pgrep -x NeuralSpotlight); do
  [[ "$(ps -o comm= -p "$pid")" == "$EXE" ]] && kill "$pid" && sleep 0.5
done
ARGS=(-indexDir "$INDEX" -modelsDir "$MODELS" -defaultsSuite "$SUITE")
if [[ -z "${ROOTS+set}" ]]; then
  ARGS+=(-roots "$TB/real:$TB/synthetic:$TB/traps")
elif [[ -n "$ROOTS" ]]; then
  ARGS+=(-roots "$ROOTS")
fi
open -n "$APP" --args "${ARGS[@]}" "$@"
