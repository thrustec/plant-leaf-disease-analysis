#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export JUPYTER_CONFIG_DIR="$PWD/.jupyter"
export JUPYTER_DATA_DIR="$PWD/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$PWD/.jupyter/runtime"
export IPYTHONDIR="$PWD/.jupyter/ipython"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
exec .venv/bin/python -m jupyterlab --notebook-dir="$PWD" --ip=127.0.0.1 --port=8888 "$@"
