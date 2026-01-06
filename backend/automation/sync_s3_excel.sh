#!/bin/bash
# Sync Excel files from S3 to local directory
# This script can be run manually or scheduled to run after Lambda functions execute

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Run sync script
cd "$PROJECT_ROOT"
python3 scripts/sync_excel_from_s3.py --all


