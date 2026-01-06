/**
 * StrategyResultsScreen Component
 * Displays filtered predictions based on selected strategy with bottom tabs for each sport
 */

import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { STRATEGY_INFO, SPORTS, SPORT_NAMES } from '../constants/api';
import SportTabScreen from './SportTabScreen';

const Tab = createBottomTabNavigator();

const StrategyResultsScreen = ({ route, navigation }) => {
  const { strategy } = route.params;
  const [refreshing, setRefreshing] = useState(false);
  const strategyInfo = STRATEGY_INFO[strategy];

  const handleRefresh = () => {
    setRefreshing(true);
    // Refresh will be handled by individual SportTabScreens
    setTimeout(() => setRefreshing(false), 1000);
  };

  // Create screen components for each sport
  const NFLScreen = () => (
    <SportTabScreen
      sport={SPORTS.NFL}
      strategy={strategy}
      onRefresh={handleRefresh}
      refreshing={refreshing}
    />
  );

  const NBAScreen = () => (
    <SportTabScreen
      sport={SPORTS.NBA}
      strategy={strategy}
      onRefresh={handleRefresh}
      refreshing={refreshing}
    />
  );

  const NCAAMScreen = () => (
    <SportTabScreen
      sport={SPORTS.NCAAM}
      strategy={strategy}
      onRefresh={handleRefresh}
      refreshing={refreshing}
    />
  );

  return (
    <View style={styles.container}>
      {/* Header with strategy info */}
      <SafeAreaView edges={['top']} style={styles.headerContainer}>
        <View style={styles.header}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => navigation.goBack()}
          >
            <Text style={styles.backButtonText}>← Back</Text>
          </TouchableOpacity>
          <View style={styles.headerContent}>
            <Text style={styles.headerIcon}>{strategyInfo.icon}</Text>
            <Text style={styles.headerTitle}>{strategyInfo.name}</Text>
            <Text style={styles.headerDescription}>{strategyInfo.description}</Text>
          </View>
        </View>
      </SafeAreaView>

      {/* Bottom tabs for each sport */}
      <Tab.Navigator
        screenOptions={{
          headerShown: false,
          tabBarActiveTintColor: '#2196F3',
          tabBarInactiveTintColor: '#999',
          tabBarStyle: {
            backgroundColor: '#ffffff',
            borderTopWidth: 1,
            borderTopColor: '#e0e0e0',
            height: 60,
            paddingBottom: 8,
            paddingTop: 8,
          },
          tabBarLabelStyle: {
            fontSize: 12,
            fontWeight: '600',
          },
        }}
      >
        <Tab.Screen
          name="NFL"
          component={NFLScreen}
          options={{
            tabBarLabel: SPORT_NAMES[SPORTS.NFL],
            tabBarIcon: ({ color, size }) => (
              <Text style={{ color, fontSize: size || 20 }}>🏈</Text>
            ),
          }}
        />
        <Tab.Screen
          name="NBA"
          component={NBAScreen}
          options={{
            tabBarLabel: SPORT_NAMES[SPORTS.NBA],
            tabBarIcon: ({ color, size }) => (
              <Text style={{ color, fontSize: size || 20 }}>🏀</Text>
            ),
          }}
        />
        <Tab.Screen
          name="NCAAM"
          component={NCAAMScreen}
          options={{
            tabBarLabel: SPORT_NAMES[SPORTS.NCAAM],
            tabBarIcon: ({ color, size }) => (
              <Text style={{ color, fontSize: size || 20 }}>🏀</Text>
            ),
          }}
        />
      </Tab.Navigator>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  headerContainer: {
    backgroundColor: '#ffffff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    padding: 20,
    paddingTop: 10,
  },
  backButton: {
    marginBottom: 16,
  },
  backButtonText: {
    fontSize: 16,
    color: '#2196F3',
    fontWeight: '600',
  },
  headerContent: {
    alignItems: 'center',
  },
  headerIcon: {
    fontSize: 48,
    marginBottom: 8,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  headerDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
});

export default StrategyResultsScreen;

