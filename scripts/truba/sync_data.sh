#!/bin/bash
# Sync data between local machine and TRUBA
# Usage: ./sync_data.sh [push|pull]

set -euo pipefail

LOCAL_DIR="$PROJECT_DIR"
REMOTE_HOST="arf-ui1"
REMOTE_DIR="$PROJECT_DIR"

DIRECTION="${1:-pull}"

case "$DIRECTION" in
    push)
        echo "Pushing data to TRUBA..."
        rsync -avz --progress \
            --exclude='.venv/' \
            --exclude='__pycache__/' \
            --exclude='.git/' \
            --exclude='mlruns/' \
            "$LOCAL_DIR/data/" "$REMOTE_HOST:$REMOTE_DIR/data/"
        echo "Push complete."
        ;;
    pull)
        echo "Pulling data from TRUBA..."
        rsync -avz --progress \
            "$REMOTE_HOST:$REMOTE_DIR/data/" "$LOCAL_DIR/data/"
        echo "Pull complete."
        ;;
    *)
        echo "Usage: $0 [push|pull]"
        exit 1
        ;;
esac
