# Sports Betting Predictions - Expo Go Mobile App

A React Native mobile app built with Expo Go that displays sports betting predictions based on different strategy types.

## Features

- **Three Strategy Types:**
  1. **Home Team Strategy** - Shows predictions where home team has better spread coverage
  2. **Away Team Strategy** - Shows predictions where away team has better spread coverage
  3. **Spread Coverage Strategy** - Shows predictions with >10% coverage difference between teams

- **Multi-Sport Support:** NFL, NBA, and NCAAM
- **Real-time Data:** Fetches predictions from AWS API Gateway
- **Pull-to-Refresh:** Refresh predictions anytime
- **Beautiful UI:** Clean, modern interface with card-based design

## Prerequisites

- Node.js 18+ installed
- Expo Go app on your phone (iOS or Android)
- Your API Gateway URL from AWS

## Setup Instructions

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure API URL

**IMPORTANT:** You need to update the API Gateway URL before the app will work.

1. Open `src/constants/api.js`
2. Replace `YOUR-API-ID` with your actual API Gateway URL:
   ```javascript
   export const API_BASE_URL = 'https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod';
   ```

3. To find your API Gateway URL:
   - Check AWS Console → API Gateway → Your API → Stages → `prod`
   - Or run: `python3 scripts/deploy_api_gateway.py` in the backend directory

### 3. Start the Development Server

```bash
npm start
# or
npx expo start
```

### 4. Run on Your Phone

1. Install **Expo Go** app on your phone:
   - iOS: [App Store](https://apps.apple.com/app/expo-go/id982107779)
   - Android: [Google Play](https://play.google.com/store/apps/details?id=host.exp.exponent)

2. Scan the QR code:
   - **iOS**: Use Camera app to scan QR code
   - **Android**: Open Expo Go app and scan QR code

3. The app will load on your phone!

## Project Structure

```
sport-betting-frontend/
├── App.js                      # Main app entry point with navigation
├── app.json                    # Expo configuration
├── package.json                # Dependencies
├── src/
│   ├── constants/
│   │   └── api.js             # API configuration and constants
│   ├── services/
│   │   └── apiService.js     # API service layer
│   ├── screens/
│   │   ├── HomeScreen.js     # Main screen with strategy selection
│   │   └── StrategyResultsScreen.js  # Results screen
│   ├── components/
│   │   └── PredictionCard.js # Game prediction card component
│   └── utils/
│       └── strategyFilters.js # Strategy filtering logic
└── assets/                    # Images, icons, etc.
```

## How It Works

1. **Home Screen**: User selects one of three strategies
2. **API Call**: App fetches predictions from all sports (NFL, NBA, NCAAM)
3. **Filtering**: Games are filtered based on selected strategy:
   - **Home Team**: `home_cover_pct_handicap > away_cover_pct_handicap`
   - **Away Team**: `away_cover_pct_handicap > home_cover_pct_handicap`
   - **Spread Coverage**: `|home_cover_pct_handicap - away_cover_pct_handicap| > 10`
4. **Display**: Filtered games are shown with team info, spreads, and coverage percentages

## Troubleshooting

### "Network request failed"
- Check that API Gateway URL is correct in `src/constants/api.js`
- Verify API Gateway is deployed and accessible
- Check your internet connection

### "No predictions found"
- Verify predictions exist in S3 bucket
- Check API Gateway logs in CloudWatch
- Ensure predictions JSON files are properly formatted

### App won't load
- Clear Expo cache: `npx expo start -c`
- Restart Expo server
- Reinstall Expo Go app

### Module not found errors
- Run `npm install` again
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

## Development

### Hot Reload
Changes automatically reload in Expo Go. Just save your file!

### Debugging
- Shake your device to open developer menu
- Enable remote debugging in Expo Go
- Check console logs in terminal

### Testing
- Test on both iOS and Android
- Test with poor network conditions
- Test error scenarios (no internet, API down)

## Next Steps

- [ ] Add date filtering for historical predictions
- [ ] Add sport filtering (show only NFL, NBA, or NCAAM)
- [ ] Add favorite teams feature
- [ ] Add push notifications for new predictions
- [ ] Add prediction accuracy tracking

## License

Private project

