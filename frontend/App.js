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

