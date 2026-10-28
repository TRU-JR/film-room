#!/usr/bin/env bash
# Starts the app with the virtual environment's Python (runs setup first if .venv is missing).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  ./setup.sh
fi
exec .venv/bin/python -m streamlit run app.py "$@"
