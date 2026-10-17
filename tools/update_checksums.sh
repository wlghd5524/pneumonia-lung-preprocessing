#!/usr/bin/env bash
# Rewrite CHECKSUMS.sha256 for every file in the release (tracked or untracked but
# not ignored), except the checksum file itself. Run from anywhere; verify with
#   sha256sum -c CHECKSUMS.sha256
set -euo pipefail
cd "$(dirname "$0")/.."
git ls-files -co --exclude-standard -z \
  | grep -zv '^CHECKSUMS\.sha256$' \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > CHECKSUMS.sha256
echo "wrote CHECKSUMS.sha256 ($(wc -l < CHECKSUMS.sha256) files)"
