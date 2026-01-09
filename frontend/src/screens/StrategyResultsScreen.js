/**
 * StrategyResultsScreen Component
 * Displays filtered predictions based on selected strategy with bottom tabs for each sport
 * Includes date navigation to view past predictions with results
 */

import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { STRATEGY_INFO, SPORTS, SPORT_NAMES } from '../constants/api';
import SportTabScreen from './SportTabScreen';

const Tab = createBottomTabNavigator();

// Helper function to format date for display
const formatDisplayDate = (dateStr) => {
  const date = new Date(dateStr + 'T12:00:00');
  const today = new Date();
  today.setHours(12, 0, 0, 0);

  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const dateOnly = new Date(date);
  dateOnly.setHours(12, 0, 0, 0);

  if (dateOnly.getTime() === today.getTime()) {
    return 'Today';
  } else if (dateOnly.getTime() === yesterday.getTime()) {
    return 'Yesterday';
  }

  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
};

// Helper function to get date string in YYYY-MM-DD format
const getDateString = (date) => {
  return date.toISOString().split('T')[0];
};

// Helper function to check if a date is today
const isToday = (dateStr) => {
  const today = new Date();
  return dateStr === getDateString(today);
};

const StrategyResultsScreen = ({ route, navigation }) => {
  const { strategy } = route.params;
  const [refreshing, setRefreshing] = useState(false);
  const [selectedDate, setSelectedDate] = useState(getDateString(new Date()));
  const strategyInfo = STRATEGY_INFO[strategy];

  // Navigate to previous day
  const goToPreviousDay = () => {
    const current = new Date(selectedDate + 'T12:00:00');
    current.setDate(current.getDate() - 1);
    setSelectedDate(getDateString(current));
  };

  // Navigate to next day
  const goToNextDay = () => {
    const current = new Date(selectedDate + 'T12:00:00');
    const today = new Date();
    today.setHours(12, 0, 0, 0);

    const next = new Date(current);
    next.setDate(next.getDate() + 1);

    // Don't allow going past today
    if (next <= today) {
      setSelectedDate(getDateString(next));
    }
  };

  // Go to today
  const goToToday = () => {
    setSelectedDate(getDateString(new Date()));
  };

  // Check if can go to next day
  const canGoNext = !isToday(selectedDate);

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
      date={selectedDate}
      onRefresh={handleRefresh}
      refreshing={refreshing}
    />
  );

  const NBAScreen = () => (
    <SportTabScreen
      sport={SPORTS.NBA}
      strategy={strategy}
      date={selectedDate}
      onRefresh={handleRefresh}
      refreshing={refreshing}
    />
  );

  const NCAAMScreen = () => (
    <SportTabScreen
      sport={SPORTS.NCAAM}
      strategy={strategy}
      date={selectedDate}
      onRefresh={handleRefresh}
      refreshing={refreshing}
    />
  );

  // Render date navigation
  const renderDateNavigation = () => (
    <View style={styles.dateNavContainer}>
      <TouchableOpacity
        style={styles.dateNavButton}
        onPress={goToPreviousDay}
      >
        <Text style={styles.dateNavButtonText}>{'<'}</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.dateDisplay}
        onPress={goToToday}
      >
        <Text style={styles.dateText}>{formatDisplayDate(selectedDate)}</Text>
        {!isToday(selectedDate) && (
          <Text style={styles.tapToReturnText}>Tap to return to today</Text>
        )}
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.dateNavButton, !canGoNext && styles.dateNavButtonDisabled]}
        onPress={goToNextDay}
        disabled={!canGoNext}
      >
        <Text style={[styles.dateNavButtonText, !canGoNext && styles.dateNavButtonTextDisabled]}>{'>'}</Text>
      </TouchableOpacity>
    </View>
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
        {/* Date navigation */}
        {renderDateNavigation()}
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
  dateNavContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: '#f5f5f5',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  dateNavButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#2196F3',
    justifyContent: 'center',
    alignItems: 'center',
  },
  dateNavButtonDisabled: {
    backgroundColor: '#e0e0e0',
  },
  dateNavButtonText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  dateNavButtonTextDisabled: {
    color: '#999',
  },
  dateDisplay: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 16,
  },
  dateText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  tapToReturnText: {
    fontSize: 12,
    color: '#2196F3',
    marginTop: 2,
  },
});

export default StrategyResultsScreen;

