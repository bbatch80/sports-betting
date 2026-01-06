# Manual Testing Guide - AWS Lambda Functions

This guide shows you how to manually test your Lambda functions in the AWS Console.

## Step 1: Access AWS Lambda Console

1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Sign in with your AWS account
3. In the search bar at the top, type "Lambda" and click on **Lambda**
4. You should see the Lambda dashboard

## Step 2: Find Your Lambda Functions

You should see two functions:
- `collect-yesterday-games`
- `generate-predictions`

Click on either function name to open it.

## Step 3: Test a Lambda Function

### For `collect-yesterday-games`:

1. **Open the function:**
   - Click on `collect-yesterday-games` in the functions list

2. **Create a test event:**
   - Click the **"Test"** tab (next to "Code" and "Configuration")
   - Click **"Create new event"**
   - Event name: `test-collect-games`
   - Event JSON: Leave it as the default (empty object `{}`)
   - Click **"Save"**

3. **Run the test:**
   - Click **"Test"** button
   - Wait for execution (may take 30-60 seconds)
   - You'll see the execution results

4. **View results:**
   - **Execution result:** Shows if it succeeded (200) or failed
   - **Response:** Shows the JSON response with results
   - **Logs:** Click "Click here" to view detailed logs in CloudWatch

### For `generate-predictions`:

1. **Open the function:**
   - Click on `generate-predictions` in the functions list

2. **Create a test event:**
   - Click the **"Test"** tab
   - Click **"Create new event"**
   - Event name: `test-generate-predictions`
   - Event JSON: Leave it as the default (empty object `{}`)
   - Click **"Save"**

3. **Run the test:**
   - Click **"Test"** button
   - Wait for execution (may take 1-2 minutes)
   - You'll see the execution results

4. **View results:**
   - Check the response for prediction results
   - View logs to see detailed execution

## Step 4: View CloudWatch Logs

1. **From Lambda function page:**
   - Click the **"Monitor"** tab
   - Click **"View CloudWatch logs"**
   - This opens CloudWatch Logs Insights

2. **Or directly in CloudWatch:**
   - Go to [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/)
   - Click **"Logs"** → **"Log groups"**
   - Find: `/aws/lambda/collect-yesterday-games` or `/aws/lambda/generate-predictions`
   - Click on a log group
   - Click on the most recent log stream to see detailed logs

## Step 5: Verify S3 Updates

### Check Data Files:

1. Go to [S3 Console](https://console.aws.amazon.com/s3/)
2. Click on bucket: `sports-betting-analytics-data`
3. Navigate to `data/results/`
4. Check the **"Last modified"** timestamp on:
   - `nfl_season_results.xlsx`
   - `nba_season_results.xlsx`
   - `ncaam_season_results.xlsx`
5. If you just ran the test, these should have recent timestamps

### Check Predictions:

1. In the same S3 bucket, navigate to `predictions/`
2. Check for:
   - `predictions_nfl.json`
   - `predictions_nba.json`
   - `predictions_ncaam.json`
3. Click on a file and click **"Download"** to view the JSON

### Check Scores:

1. In the same S3 bucket, navigate to `scores/`
2. Check for:
   - `nfl_scores.json`
   - `nba_scores.json`
   - `ncaam_scores.json`

## Step 6: View Function Metrics

1. In the Lambda function page, click the **"Monitor"** tab
2. You'll see:
   - **Invocations:** How many times the function ran
   - **Duration:** How long each execution took
   - **Errors:** Any errors that occurred
   - **Throttles:** If the function was throttled

## Step 7: Check EventBridge Schedule

1. Go to [EventBridge Console](https://console.aws.amazon.com/events/)
2. Click **"Rules"** in the left sidebar
3. You should see:
   - `collect-yesterday-games-schedule`
   - `generate-predictions-schedule`
4. Click on a rule to see:
   - **Schedule expression:** When it runs (e.g., `cron(0 11 * * ? *)`)
   - **Targets:** Which Lambda function it triggers
   - **State:** Should be "Enabled"

## Troubleshooting

### Function Times Out:
- Check CloudWatch logs for where it got stuck
- May need to increase timeout in Configuration → General configuration

### Function Errors:
- Check CloudWatch logs for error details
- Common issues:
  - Missing permissions (check IAM role)
  - API key not found (check Secrets Manager)
  - S3 access issues (check bucket permissions)

### No Logs Appearing:
- Wait a few seconds after execution
- Check that CloudWatch Logs are enabled (should be automatic)
- Check the correct log group name

### S3 Files Not Updating:
- Check function execution succeeded (status 200)
- Check CloudWatch logs for S3 write operations
- Verify IAM role has S3 write permissions

## Quick Test Checklist

- [ ] Lambda function exists in console
- [ ] Test event created
- [ ] Test execution succeeds (status 200)
- [ ] CloudWatch logs show detailed execution
- [ ] S3 files updated (check timestamps)
- [ ] Predictions JSON files created (if testing predictions)
- [ ] EventBridge rules are enabled

## Expected Results

### collect-yesterday-games:
- **Status:** 200
- **Response:** JSON with `message: "Collection complete"` and results for each sport
- **S3 Updates:** Excel and Parquet files updated
- **Duration:** 20-60 seconds depending on number of games

### generate-predictions:
- **Status:** 200
- **Response:** JSON with `message: "Predictions generated"` and results
- **S3 Updates:** Predictions JSON files created/updated
- **Duration:** 30-120 seconds depending on data size



