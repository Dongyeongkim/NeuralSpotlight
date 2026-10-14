#!/bin/bash
# Checks installed Linux package libraries and the explicit external-worker requirement.
# No native model is available on Linux. Semantic integration needs a separately provisioned worker.
set -euo pipefail
cd "$(dirname "$0")/.."
app=$(realpath "${1:-/usr/bin/digup-shell}")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if [[ $app == *.AppImage ]]; then
  mkdir "$work/image"
  (cd "$work/image" && "$app" --appimage-extract > /dev/null)
  helper=$(find "$work/image/squashfs-root" -type f -path '*/helper/digup' -print -quit)
  [[ -n $helper ]] || { echo "AppImage contains no digup helper" >&2; exit 1; }
else
  helper=$(dirname "$app")/../lib/DigUp/helper/digup
fi
missing=$(for file in "$app" "$(dirname "$helper")"/*; do
  [[ -f $file && ! -L $file ]] && ldd "$file" 2>&1 | grep "not found" | sed "s|^|$file: |"
done || true)
if [[ -n $missing ]]; then
  echo "missing libraries: $missing" >&2
  exit 1
fi
bash scripts/check-portable-helper.sh "$helper"
