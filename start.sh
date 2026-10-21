#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for candidate in python3.12 python3.13 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13) and sys.maxsize > 2**32' 2>/dev/null; then
    exec "$candidate" launch.py "$@"
  fi
done
echo 'Install 64-bit Python 3.12 (including venv/pip) and run this script again.' >&2
exit 1
