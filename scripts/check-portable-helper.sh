#!/bin/bash
# Checks a Linux/Windows helper without native model assets or a configured external worker.
# A native-only operation must fail explicitly; metadata tools must still run without a model.
set -euo pipefail
helper=${1:?path to the installed neuralspotlight helper}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
unset NEURALSPOTLIGHT_WORKER_DIR NEURALSPOTLIGHT_EMBEDDING_BACKEND
"$helper" --help > "$work/help"
"$helper" status --index "$work/index" > "$work/status"
grep -q 'configure NEURALSPOTLIGHT_WORKER_DIR' "$work/status"
if "$helper" download-model --index "$work/index" --to "$work/models" > "$work/download" 2>&1; then
  echo "portable helper incorrectly advertised a native model download" >&2
  exit 1
fi
grep -q 'NEURALSPOTLIGHT_WORKER_DIR' "$work/download"
if "$helper" worker --text-only --index "$work/index" < /dev/null > "$work/worker" 2>&1; then
  echo "portable helper incorrectly started a native Core ML worker" >&2
  exit 1
fi
grep -q 'NEURALSPOTLIGHT_WORKER_DIR' "$work/worker"
mkdir "$work/files"
if "$helper" index-service "$work/files" --index "$work/index" --linger 0 < /dev/null > "$work/service" 2>&1; then
  echo "portable indexing silently waited for an unavailable native runtime" >&2
  exit 1
fi
grep -q 'NEURALSPOTLIGHT_WORKER_DIR' "$work/service"
echo "PASS: portable helper starts, metadata works, native operations require an external worker"
