"""
Quick Test - Test Lambda with visible progress
Patches logger to show output in real-time
"""

import sys
import os
import time
from datetime import datetime, timedelta, timezone
import logging

# Set up logging BEFORE importing Lambda function
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Now import Lambda function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'collect_yesterday_games'))

# Patch the logger in the Lambda module to use our handler
from lambda_function import lambda_handler
import lambda_function as lf_module

# Replace the logger with one that outputs to stdout
lf_module.logger = logging.getLogger('lambda_function')
lf_module.logger.setLevel(logging.INFO)
lf_module.logger.addHandler(logging.StreamHandler(sys.stdout))

def create_test_event():
    return {}

def create_test_context():
    class MockContext:
        def __init__(self):
            self.function_name = "test-collect"
            self.memory_limit_in_mb = 512
            self.invoked_function_arn = "test"
            self.aws_request_id = "test"
            self.log_group_name = "/aws/lambda/test"
            self.log_stream_name = "test"
            self.function_version = "$LATEST"
            self.remaining_time_in_millis = lambda: 900000
    return MockContext()

print("="*80)
print("QUICK TEST - Collect Yesterday's Games")
print("="*80)
print("\nThis will show real-time progress...")
print("(Press Ctrl+C to stop if it takes too long)\n")

start_time = time.time()

try:
    event = create_test_event()
    context = create_test_context()
    
    print("\nCalling lambda_handler()...\n")
    response = lambda_handler(event, context)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"COMPLETED in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}")
    
    if isinstance(response, dict):
        import json
        body = json.loads(response.get('body', '{}'))
        print(f"\nStatus: {response.get('statusCode')}")
        print(f"Results: {json.dumps(body, indent=2)}")
    
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n\n⚠ INTERRUPTED after {elapsed:.2f} seconds")
    print("The function was still running - it may have been processing many games.")
    
except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n\n✗ ERROR after {elapsed:.2f} seconds: {e}")
    import traceback
    traceback.print_exc()



