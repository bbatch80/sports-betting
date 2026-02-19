#!/bin/bash
# Setup script for daily collection automation

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PLIST_FILE="$SCRIPT_DIR/com.sportsbetting.dailycollection.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_NAME="com.sportsbetting.dailycollection.plist"
TARGET_PLIST="$LAUNCH_AGENTS_DIR/$PLIST_NAME"

echo "Setting up daily collection automation..."
echo "========================================"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"
echo "✓ Created logs directory"

# Copy plist to LaunchAgents
if [ -f "$PLIST_FILE" ]; then
    cp "$PLIST_FILE" "$TARGET_PLIST"
    echo "✓ Copied plist file to LaunchAgents"
else
    echo "✗ Error: plist file not found at $PLIST_FILE"
    exit 1
fi

# Update the plist with correct paths
sed -i '' "s|REPLACE_PROJECT_ROOT|$PROJECT_ROOT|g" "$TARGET_PLIST"
echo "✓ Updated paths in plist file"

# Load the launch agent
if launchctl list | grep -q "com.sportsbetting.dailycollection"; then
    echo "  Unloading existing agent..."
    launchctl unload "$TARGET_PLIST" 2>/dev/null
fi

launchctl load "$TARGET_PLIST"
echo "✓ Loaded launch agent"

echo ""
echo "Automation setup complete!"
echo "=========================="
echo "The script will run daily at 6:00 AM EST"
echo ""
echo "To check status: launchctl list | grep sportsbetting"
echo "To view logs: tail -f $PROJECT_ROOT/logs/daily_collection.log"
echo "To stop: launchctl unload $TARGET_PLIST"
echo "To start: launchctl load $TARGET_PLIST"

