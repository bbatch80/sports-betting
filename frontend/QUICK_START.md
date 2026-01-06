# Quick Start Guide

## Step 1: Install Dependencies

```bash
npm install
```

## Step 2: Configure API URL

**CRITICAL:** Update the API Gateway URL in `src/constants/api.js`

```javascript
export const API_BASE_URL = 'https://YOUR-ACTUAL-API-ID.execute-api.us-east-1.amazonaws.com/prod';
```

To find your API Gateway URL:
1. Go to AWS Console → API Gateway
2. Find your API: `sports-betting-predictions-api`
3. Click on "Stages" → `prod`
4. Copy the "Invoke URL"

Or run in backend directory:
```bash
python3 scripts/deploy_api_gateway.py
```

## Step 3: Start Expo

```bash
npm start
```

## Step 4: Open on Your Phone

1. Install **Expo Go** app:
   - iOS: [Download](https://apps.apple.com/app/expo-go/id982107779)
   - Android: [Download](https://play.google.com/store/apps/details?id=host.exp.exponent)

2. Scan QR code from terminal:
   - **iOS**: Use Camera app
   - **Android**: Open Expo Go app → Scan QR code

3. App loads on your phone!

## That's It! 🎉

The app will:
- Show 3 strategy buttons on the home screen
- Fetch predictions from all sports when you select a strategy
- Filter and display games matching your selected strategy

## Troubleshooting

**"Network request failed"**
→ Check API URL in `src/constants/api.js`

**"No predictions found"**
→ Verify API Gateway is deployed and predictions exist in S3

**App won't load**
→ Run `npx expo start -c` to clear cache

