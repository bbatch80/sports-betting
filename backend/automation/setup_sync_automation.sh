#!/bin/bash
# Setup script for S3 Excel sync automation
# This runs daily at 7:00 AM EST (30 minutes after Lambda collection)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PLIST_FILE="$SCRIPT_DIR/com.sportsbetting.syncs3excel.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_NAME="com.sportsbetting.syncs3excel.plist"
TARGET_PLIST="$LAUNCH_AGENTS_DIR/$PLIST_NAME"

echo "Setting up S3 Excel sync automation..."
echo "======================================"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"
echo "✓ Created logs directory"

# Update plist with correct paths
if [ -f "$PLIST_FILE" ]; then
    # Make a copy and update paths
    cp "$PLIST_FILE" "$PLIST_FILE.tmp"
    sed "s|/Users/robertbatchelor/Documents/Projects/sports-betting/backend|$PROJECT_ROOT|g" "$PLIST_FILE.tmp" > "$PLIST_FILE.tmp2"
    mv "$PLIST_FILE.tmp2" "$PLIST_FILE"
    rm -f "$PLIST_FILE.tmp"
    
    cp "$PLIST_FILE" "$TARGET_PLIST"
    echo "✓ Copied plist file to LaunchAgents"
else
    echo "✗ Error: plist file not found at $PLIST_FILE"
    exit 1
fi

# Load the launch agent
if launchctl list | grep -q "com.sportsbetting.syncs3excel"; then
    echo "  Unloading existing agent..."
    launchctl unload "$TARGET_PLIST" 2>/dev/null
fi

launchctl load "$TARGET_PLIST"
echo "✓ Loaded launch agent"

echo ""
echo "Automation setup complete!"
echo "=========================="
echo "The script will run daily at 7:00 AM EST (after Lambda collection)"
echo ""
echo "To check status: launchctl list | grep syncs3excel"
echo "To view logs: tail -f $PROJECT_ROOT/logs/s3_sync.log"
echo "To stop: launchctl unload $TARGET_PLIST"
echo "To start: launchctl load $TARGET_PLIST"
echo ""
echo "To run manually: python3 $PROJECT_ROOT/scripts/sync_excel_from_s3.py --all"


