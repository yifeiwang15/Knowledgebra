#!/bin/bash
KGHOME=$(pwd)
export PYTHONPATH="$KGHOME:$PYTHONPATH"
export LOG_DIR="$KGHOME/logs"
export DATA_PATH="$KGHOME/data"
source semi_env/bin/activate
