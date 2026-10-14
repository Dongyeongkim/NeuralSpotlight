#!/bin/bash
# Installs the Windows package and checks its helper's libraries and external-worker requirement.
# Run in Git for Windows' bash. Semantic integration needs a separately provisioned worker.
set -euo pipefail
cd "$(dirname "$0")/.."
setup=${1:?the installer}
MSYS_NO_PATHCONV=1 "$setup" /S
installed=$(cygpath -u "$LOCALAPPDATA")/DigUp
ls "$installed/DigUp.exe" "$installed/helper/digup.exe" > /dev/null
helper=$installed/helper/digup.exe
# Only the packaged DLLs and Windows may satisfy the helper's runtime dependencies.
PATH="/c/Windows/System32:/c/Windows:/c/Windows/System32/Wbem" "$helper" --help > /dev/null
bash scripts/check-portable-helper.sh "$helper"
