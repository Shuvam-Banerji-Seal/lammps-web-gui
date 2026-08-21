#!/usr/bin/env bash
# Publish wiki/*.md to the GitHub wiki.
# Prerequisite (one-time): initialize the wiki via the web UI
#   repo -> Wiki tab -> Create the first page -> save
set -euo pipefail

REPO="Shuvam-Banerji-Seal/lammps_data_web_viewer"
SRC="$(cd "$(dirname "$0")/.." && pwd)/wiki"
WORK="$(mktemp -d)"

git clone "https://github.com/${REPO}.wiki.git" "$WORK/wiki"
cp "$SRC"/*.md "$WORK/wiki/"
cd "$WORK/wiki"

if [[ -n "$(git status --porcelain)" ]]; then
  git add .
  git commit -m "docs: sync wiki from repository ($(date -u +%Y-%m-%d))"
  git push
  echo "Wiki updated."
else
  echo "Wiki already up to date."
fi
