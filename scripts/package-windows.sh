#!/bin/bash
# Builds NeuralSpotlight's Windows installer: the Tauri shell (shell/) with NeuralSpotlight's helper, PDFium, the Swift and C++
# runtimes and ffmpeg in its helper folder, as an NSIS installer that installs for the current user (no admin prompt).
# Semantic embedding requires an explicitly provisioned external worker; Core ML is macOS only.
# Run in Git for Windows' bash after scripts/fetch-vendor.sh. Needs Swift, Rust, Node, and an LGPL ffmpeg 8.1+ with its
# DLLs (FFMPEG: its bin folder):
#   FFMPEG=D:/ffmpeg/bin scripts/package-windows.sh      → shell/src-tauri/target/release/bundle/nsis/
set -euo pipefail
cd "$(dirname "$0")/.."
FFMPEG=${FFMPEG:?set FFMPEG to the bin folder of an LGPL ffmpeg build (ffmpeg.exe, ffprobe.exe and their DLLs)}
TAURI_CLI=2.12.2

swift build -c release --product neuralspotlight

# What the installer puts beside NeuralSpotlight.exe, in helper\ ("NeuralSpotlight.exe" and "neuralspotlight.exe" can't share a folder): the helper
# finds the DLLs and ffmpeg in its own folder. The C++ runtime goes with them, so nothing needs
# installing first; the shell itself links its C runtime statically (.cargo/config.toml).
swift_root=$(cd "$(dirname "$(command -v swift)")/../../../.." && pwd)   # …/Swift from …/Swift/Toolchains/*/usr/bin
swift_runtime=$(ls -d "$swift_root"/Runtimes/*/usr/bin | tail -1)
crt=$(ls -d "$(cygpath "$VCToolsRedistDir")"x64/Microsoft.VC*.CRT | tail -1)   # set by the developer environment
helper=shell/src-tauri/helper
rm -rf "$helper"
mkdir -p "$helper"
cp .build/release/neuralspotlight.exe Vendor/pdfium/bin/pdfium.dll "$helper/"
cp "$swift_runtime"/*.dll "$helper/"
cp "$crt"/*.dll "$helper/"   # over the Swift runtime's copies of the same: the newer C++ runtime
cp "$FFMPEG"/ffmpeg.exe "$FFMPEG"/ffprobe.exe "$FFMPEG"/*.dll "$helper/"

# With the release's signing key (TAURI_SIGNING_PRIVATE_KEY, and its password) the packages that update themselves are
# signed for in-app updates (a .sig beside each). Installed copies check those signatures with the public key in
# tauri.conf.json (plugins.updater.pubkey): a release built without it could never update.
signing=
if [[ -n "${TAURI_SIGNING_PRIVATE_KEY:-}" ]]; then
  if grep -q '"pubkey": ""' shell/src-tauri/tauri.conf.json; then
    echo "the signing key is set, but plugins.updater.pubkey in shell/src-tauri/tauri.conf.json is empty" >&2
    exit 1
  fi
  signing=',"createUpdaterArtifacts":true'
fi
# Installers of an earlier build (another version) would be taken for this one's.
rm -rf shell/src-tauri/target/release/bundle/nsis
# The helper folder goes in as the installer's resources here only: Tauri checks resources on every build.
(cd shell/src-tauri && npx --yes "@tauri-apps/cli@$TAURI_CLI" build --bundles nsis \
  --config "{\"bundle\":{\"resources\":{\"helper/\":\"helper/\"}$signing}}")
ls -la shell/src-tauri/target/release/bundle/nsis/
