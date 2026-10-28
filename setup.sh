#!/usr/bin/env bash
# Creates .venv inside this folder and installs every package the app needs into it.
# Nothing is installed system-wide; pip's download cache also stays inside .venv.
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Python 3.10 or newer is required. Install it, or run: PYTHON=/path/to/python3.12 ./setup.sh"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi

export PIP_CACHE_DIR="$PWD/.venv/pip-cache"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete. Start the app with:  ./run.sh"
echo "Run the tests with:                  .venv/bin/python -m pytest"
