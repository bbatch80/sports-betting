# Sports Betting Frontend - Expo Go Mobile App Guide

## Backend Overview

Your sports betting backend is a well-structured AWS-based system with the following components:

### Architecture
- **Data Collection**: Automated daily collection of betting lines from The Odds API
- **Storage**: S3 bucket (`sports-betting-analytics-data`) storing:
  - Game results (Excel/Parquet files)
  - Predictions (JSON files)
  - Historical scores
- **APIs**: AWS Lambda functions behind API Gateway serving:
  - **Predictions API**: Real-time betting predictions
  - **Results API**: Historical game results

### Supported Sports
- **NFL** (americanfootball_nfl)
- **NBA** (basketball_nba)
- **NCAAM** (basketball_ncaab)

---

## Available API Endpoints

### Base URL
```
https://{api-id}.execute-api.us-east-1.amazonaws.com/prod
```

**Note**: You'll need to get your actual API Gateway URL. Check your AWS Console or run the deployment script to see the URL.

### Predictions API

#### Get Predictions for a Sport
```
GET /api/predictions/{sport}
```

**Parameters:**
- `sport`: `nfl`, `nba`, or `ncaam`
- `date` (optional query param): `YYYY-MM-DD` format for specific date predictions

**Examples:**
- `GET /api/predictions/nba` - Today's NBA predictions
- `GET /api/predictions/nfl?date=2025-01-28` - NFL predictions for specific date
- `GET /api/predictions/all` - All sports predictions

**Response Format:**
```json
{
  "sport": "nba",
  "sport_name": "NBA",
  "handicap_points": 9,
  "generated_at": "2025-12-12T19:28:22Z",
  "games": [
    {
      "game_date": "2025-01-28",
      "home_team": "Lakers",
      "away_team": "Warriors",
      "closing_spread": -3.5,
      "predicted_advantage": 2.1,
      "coverage_percentage": 65.5
    }
  ],
  "opportunities": [
    {
      "game_date": "2025-01-28",
      "team": "Lakers",
      "opponent": "Warriors",
      "spread": -3.5,
      "advantage": 2.1,
      "confidence": "high"
    }
  ],
  "summary": {
    "total_games": 10,
    "opportunities": 5,
    "high_confidence": 3
  }
}
```

### Results API

#### Get Game Results for a Sport
```
GET /api/results/{sport}
```

**Parameters:**
- `sport`: `nfl`, `nba`, or `ncaam`
- `start_date` (optional query param): Filter from date (YYYY-MM-DD)
- `end_date` (optional query param): Filter to date (YYYY-MM-DD)

**Examples:**
- `GET /api/results/nba` - All NBA results
- `GET /api/results/nfl?start_date=2025-01-01&end_date=2025-01-31` - NFL results for January
- `GET /api/results/all` - All sports results

**Response Format:**
```json
{
  "sport": "nba",
  "sport_name": "NBA",
  "total_games": 150,
  "last_updated": "2025-01-28T12:00:00",
  "games": [
    {
      "game_date": "2025-01-27",
      "home_team": "Lakers",
      "away_team": "Warriors",
      "closing_spread": -3.5,
      "home_score": 110,
      "away_score": 105,
      "spread_result_difference": 1.5
    }
  ]
}
```

### CORS
All endpoints support CORS with `Access-Control-Allow-Origin: *`, so your mobile app can make requests directly.

---

## Expo Go Mobile App Setup

### Why Expo Go?
- **Fast Development**: Test on real devices instantly without building native apps
- **No Native Code**: Pure JavaScript/React Native
- **Easy Sharing**: Share QR code for instant testing
- **Cross-Platform**: Works on iOS and Android
- **Hot Reload**: See changes instantly

### Prerequisites

1. **Node.js** (v18+ recommended)
   ```bash
   node --version
   ```

2. **Expo CLI** (install globally)
   ```bash
   npm install -g expo-cli
   ```

3. **Expo Go App** (on your phone)
   - iOS: Download from App Store
   - Android: Download from Google Play Store

4. **Git** (for version control)

---

## Step-by-Step Implementation

### Step 1: Initialize Expo Project

```bash
cd /Users/robertbatchelor/Documents/Projects/sport-betting-frontend
npx create-expo-app@latest . --template blank
```

Or if you want a fresh start:
```bash
npx create-expo-app sports-betting-app
cd sports-betting-app
```

### Step 2: Install Dependencies

```bash
npm install @react-navigation/native @react-navigation/bottom-tabs
npm install react-native-screens react-native-safe-area-context
npm install react-native-gesture-handler
npm install axios  # For API calls
npm install @expo/vector-icons  # For icons
```

### Step 3: Project Structure

Create this folder structure:
```
sport-betting-frontend/
├── App.js                 # Main app entry point
├── app.json              # Expo configuration
├── package.json
├── src/
│   ├── constants/
│   │   └── api.js        # API configuration
│   ├── services/
│   │   └── apiService.js # API service layer
│   ├── screens/
│   │   ├── PredictionsScreen.js
│   │   ├── ResultsScreen.js
│   │   └── HomeScreen.js
│   ├── components/
│   │   ├── PredictionCard.js
│   │   ├── GameCard.js
│   │   └── LoadingSpinner.js
│   └── utils/
│       └── formatters.js # Date/number formatting
└── assets/               # Images, fonts, etc.
```

### Step 4: Configure API Base URL

Create `src/constants/api.js`:
```javascript
// Replace with your actual API Gateway URL
export const API_BASE_URL = 'https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod';

export const SPORTS = {
  NFL: 'nfl',
  NBA: 'nba',
  NCAAM: 'ncaam',
};

export const SPORT_NAMES = {
  [SPORTS.NFL]: 'NFL',
  [SPORTS.NBA]: 'NBA',
  [SPORTS.NCAAM]: 'NCAAM',
};
```

### Step 5: Create API Service

Create `src/services/apiService.js`:
```javascript
import axios from 'axios';
import { API_BASE_URL } from '../constants/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getPredictions = async (sport, date = null) => {
  try {
    const url = date 
      ? `/api/predictions/${sport}?date=${date}`
      : `/api/predictions/${sport}`;
    const response = await apiClient.get(url);
    return response.data;
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw error;
  }
};

export const getAllPredictions = async () => {
  try {
    const response = await apiClient.get('/api/predictions/all');
    return response.data;
  } catch (error) {
    console.error('Error fetching all predictions:', error);
    throw error;
  }
};

export const getResults = async (sport, startDate = null, endDate = null) => {
  try {
    let url = `/api/results/${sport}`;
    const params = [];
    if (startDate) params.push(`start_date=${startDate}`);
    if (endDate) params.push(`end_date=${endDate}`);
    if (params.length > 0) url += `?${params.join('&')}`;
    
    const response = await apiClient.get(url);
    return response.data;
  } catch (error) {
    console.error('Error fetching results:', error);
    throw error;
  }
};

export const getAllResults = async () => {
  try {
    const response = await apiClient.get('/api/results/all');
    return response.data;
  } catch (error) {
    console.error('Error fetching all results:', error);
    throw error;
  }
};
```

### Step 6: Create UI Components

#### PredictionCard Component
```javascript
// src/components/PredictionCard.js
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const PredictionCard = ({ opportunity }) => {
  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.team}>{opportunity.team}</Text>
        <Text style={styles.opponent}>vs {opportunity.opponent}</Text>
      </View>
      <View style={styles.details}>
        <Text style={styles.spread}>Spread: {opportunity.spread}</Text>
        <Text style={styles.advantage}>Advantage: {opportunity.advantage}</Text>
        <Text style={styles.confidence}>Confidence: {opportunity.confidence}</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 16,
    marginVertical: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  team: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  opponent: {
    fontSize: 16,
    color: '#666',
  },
  details: {
    marginTop: 8,
  },
  spread: {
    fontSize: 14,
    color: '#333',
  },
  advantage: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: '600',
  },
  confidence: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
});

export default PredictionCard;
```

### Step 7: Create Screens

#### PredictionsScreen
```javascript
// src/screens/PredictionsScreen.js
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, RefreshControl, ActivityIndicator } from 'react-native';
import { getPredictions } from '../services/apiService';
import PredictionCard from '../components/PredictionCard';
import { SPORT_NAMES } from '../constants/api';

const PredictionsScreen = ({ route }) => {
  const { sport } = route.params;
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadPredictions = async () => {
    try {
      setError(null);
      const data = await getPredictions(sport);
      setPredictions(data);
    } catch (err) {
      setError('Failed to load predictions');
      console.error(err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadPredictions();
  }, [sport]);

  const onRefresh = () => {
    setRefreshing(true);
    loadPredictions();
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#2196F3" />
        <Text style={styles.loadingText}>Loading predictions...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={predictions?.opportunities || []}
        renderItem={({ item }) => <PredictionCard opportunity={item} />}
        keyExtractor={(item, index) => `prediction-${index}`}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={
          <View style={styles.center}>
            <Text>No predictions available</Text>
          </View>
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 16,
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    color: '#666',
  },
  errorText: {
    color: '#f44336',
    fontSize: 16,
  },
});

export default PredictionsScreen;
```

### Step 8: Set Up Navigation

Update `App.js`:
```javascript
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import PredictionsScreen from './src/screens/PredictionsScreen';
import { SPORTS, SPORT_NAMES } from './src/constants/api';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={{
            headerShown: true,
            tabBarActiveTintColor: '#2196F3',
            tabBarInactiveTintColor: '#999',
          }}
        >
          <Tab.Screen
            name="NFL"
            component={PredictionsScreen}
            initialParams={{ sport: SPORTS.NFL }}
            options={{
              title: SPORT_NAMES[SPORTS.NFL],
              tabBarIcon: ({ color, size }) => (
                <Ionicons name="football" size={size} color={color} />
              ),
            }}
          />
          <Tab.Screen
            name="NBA"
            component={PredictionsScreen}
            initialParams={{ sport: SPORTS.NBA }}
            options={{
              title: SPORT_NAMES[SPORTS.NBA],
              tabBarIcon: ({ color, size }) => (
                <Ionicons name="basketball" size={size} color={color} />
              ),
            }}
          />
          <Tab.Screen
            name="NCAAM"
            component={PredictionsScreen}
            initialParams={{ sport: SPORTS.NCAAM }}
            options={{
              title: SPORT_NAMES[SPORTS.NCAAM],
              tabBarIcon: ({ color, size }) => (
                <Ionicons name="basketball-outline" size={size} color={color} />
              ),
            }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
```

### Step 9: Configure Expo

Update `app.json`:
```json
{
  "expo": {
    "name": "Sports Betting Predictions",
    "slug": "sports-betting-predictions",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/icon.png",
    "userInterfaceStyle": "light",
    "splash": {
      "image": "./assets/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#ffffff"
    },
    "assetBundlePatterns": [
      "**/*"
    ],
    "ios": {
      "supportsTablet": true,
      "bundleIdentifier": "com.sportsbetting.predictions"
    },
    "android": {
      "adaptiveIcon": {
        "foregroundImage": "./assets/adaptive-icon.png",
        "backgroundColor": "#ffffff"
      },
      "package": "com.sportsbetting.predictions"
    },
    "web": {
      "favicon": "./assets/favicon.png"
    }
  }
}
```

### Step 10: Run the App

1. **Start Expo development server:**
   ```bash
   npx expo start
   ```

2. **Scan QR code:**
   - Open Expo Go app on your phone
   - Scan the QR code from terminal (iOS) or Expo Go app (Android)

3. **Development workflow:**
   - Make changes to your code
   - App will automatically reload
   - Shake device to open developer menu

---

## Getting Your API Gateway URL

If you don't know your API Gateway URL:

1. **Check AWS Console:**
   - Go to API Gateway → APIs → `sports-betting-predictions-api`
   - Click on "Stages" → `prod`
   - Copy the "Invoke URL"

2. **Or run the deployment script:**
   ```bash
   cd /Users/robertbatchelor/Documents/Projects/sports-betting-backend
   python3 scripts/deploy_api_gateway.py
   ```
   The script will output the API URL at the end.

3. **Or use AWS CLI:**
   ```bash
   aws apigateway get-rest-apis --query "items[?name=='sports-betting-predictions-api']"
   aws apigateway get-stages --rest-api-id YOUR_API_ID
   ```

---

## Recommended Features to Add

### Phase 1: Core Features
- [x] Predictions display
- [ ] Results/history view
- [ ] Pull-to-refresh
- [ ] Loading states
- [ ] Error handling

### Phase 2: Enhanced UX
- [ ] Date picker for historical predictions
- [ ] Filter by confidence level
- [ ] Search/filter games
- [ ] Favorite teams
- [ ] Push notifications for new predictions

### Phase 3: Analytics
- [ ] Prediction accuracy tracking
- [ ] Performance metrics
- [ ] Charts/graphs
- [ ] Export data

---

## Testing Strategy

1. **Test API Connection:**
   - Verify API Gateway URL is correct
   - Test endpoints with curl or Postman first
   - Check CORS headers

2. **Test on Real Devices:**
   - Use Expo Go for quick testing
   - Test on both iOS and Android
   - Test with poor network conditions

3. **Error Scenarios:**
   - No internet connection
   - API timeout
   - Invalid responses
   - Empty data

---

## Troubleshooting

### Common Issues

1. **"Network request failed"**
   - Check API Gateway URL is correct
   - Verify API Gateway is deployed
   - Check CORS configuration

2. **"Module not found"**
   - Run `npm install` again
   - Clear Expo cache: `npx expo start -c`

3. **App won't load**
   - Restart Expo server
   - Reinstall Expo Go app
   - Check for syntax errors in console

4. **API returns 403/404**
   - Verify Lambda function permissions
   - Check API Gateway resource paths
   - Ensure deployment stage exists

---

## Next Steps

1. **Set up the project structure**
2. **Get your API Gateway URL**
3. **Implement basic predictions screen**
4. **Add results/history screen**
5. **Polish UI/UX**
6. **Add error handling and loading states**
7. **Test on multiple devices**

---

## Resources

- [Expo Documentation](https://docs.expo.dev/)
- [React Navigation](https://reactnavigation.org/)
- [React Native Components](https://reactnative.dev/docs/components-and-apis)
- [Axios Documentation](https://axios-http.com/docs/intro)

---

**Last Updated**: 2025-01-28

