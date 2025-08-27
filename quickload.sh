#!/bin/bash
VENV_NAME=".venv"
FUNCS_FILE=".env_funcs.sh"

# Activate virtual environment
if [ -d "$VENV_NAME" ]; then
    source "$VENV_NAME/bin/activate"
    echo "✅ Virtual environment activated."
else
    echo "⚠️ Virtual environment $VENV_NAME not found!"
fi

# Load functions
if [ -f "$FUNCS_FILE" ]; then
    source "$FUNCS_FILE"
    echo "✅ longrun and gitpush functions loaded."
else
    echo "⚠️ Functions file $FUNCS_FILE not found!"
fi
