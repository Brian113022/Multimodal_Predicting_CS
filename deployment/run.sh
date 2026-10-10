#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${PYTHON_EXECUTABLE:-}" ]; then
    web_python=$PYTHON_EXECUTABLE
elif [ -x "$script_dir/../.venv/bin/python" ]; then
    web_python=$script_dir/../.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    web_python=python3
else
    echo "Python missing. Activate your validated webapp environment first." >&2
    exit 1
fi
if [ "$#" -eq 0 ]; then
    exec "$web_python" "$script_dir/deploy.py" run
fi
exec "$web_python" "$script_dir/deploy.py" "$@"
