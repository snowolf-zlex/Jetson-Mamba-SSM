#!/bin/bash
# Wrapper script to run Python with Mamba-SSM fixes loaded
# Usage: ./run_with_mamba.sh <python_script> [args...]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add project directory to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

# Import fix_causal_conv1d before running any script
python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); import fix_causal_conv1d" 2>/dev/null || true

# Run the requested command
python3 "$@"
