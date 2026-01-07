/**
 * Main App Component
 * Sets up navigation and app structure
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import HomeScreen from './src/screens/HomeScreen';
import StrategyResultsScreen from './src/screens/StrategyResultsScreen';
import EliteTeamsScreen from './src/screens/EliteTeamsScreen';
import HotVsColdScreen from './src/screens/HotVsColdScreen';
import OpponentPerfectFormScreen from './src/screens/OpponentPerfectFormScreen';
import StrategyPerformanceScreen from './src/screens/StrategyPerformanceScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="auto" />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={{
            headerShown: false,
            cardStyle: { backgroundColor: '#f5f5f5' },
          }}
        >
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen
            name="StrategyResults"
            component={StrategyResultsScreen}
            options={{
              headerShown: false,
            }}
          />
          <Stack.Screen
            name="EliteTeams"
            component={EliteTeamsScreen}
            options={{
              headerShown: true,
              headerTitle: 'Elite Teams',
              headerBackTitle: 'Back',
            }}
          />
          <Stack.Screen
            name="HotVsCold"
            component={HotVsColdScreen}
            options={{
              headerShown: true,
              headerTitle: 'Hot vs Cold',
              headerBackTitle: 'Back',
            }}
          />
          <Stack.Screen
            name="OpponentPerfectForm"
            component={OpponentPerfectFormScreen}
            options={{
              headerShown: true,
              headerTitle: 'Regression Bets',
              headerBackTitle: 'Back',
            }}
          />
          <Stack.Screen
            name="StrategyPerformance"
            component={StrategyPerformanceScreen}
            options={{
              headerShown: true,
              headerTitle: 'Strategy Performance',
              headerBackTitle: 'Back',
            }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

