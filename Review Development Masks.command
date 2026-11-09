#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export XDG_CACHE_HOME="$PWD/.cache"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
exec .venv-labelme/bin/labelme data/annotations/labelme/development --config labelme_config.yaml
