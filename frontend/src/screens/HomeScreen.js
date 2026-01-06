/**
 * HomeScreen Component
 * Main screen displaying 3 strategy selection buttons
 */

import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { STRATEGIES, STRATEGY_INFO } from '../constants/api';

const HomeScreen = ({ navigation }) => {
  const handleStrategySelect = (strategy) => {
    navigation.navigate('StrategyResults', { strategy });
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.header}>
        <Text style={styles.title}>Sports Betting Predictions</Text>
        <Text style={styles.subtitle}>Select a strategy to view predictions</Text>
      </View>

      <View style={styles.strategiesContainer}>
        {/* Home Team Strategy */}
        <TouchableOpacity
          style={styles.strategyCard}
          onPress={() => handleStrategySelect(STRATEGIES.HOME_TEAM)}
          activeOpacity={0.7}
        >
          <View style={styles.strategyIconContainer}>
            <Text style={styles.strategyIcon}>{STRATEGY_INFO[STRATEGIES.HOME_TEAM].icon}</Text>
          </View>
          <View style={styles.strategyContent}>
            <Text style={styles.strategyName}>
              {STRATEGY_INFO[STRATEGIES.HOME_TEAM].name}
            </Text>
            <Text style={styles.strategyDescription}>
              {STRATEGY_INFO[STRATEGIES.HOME_TEAM].description}
            </Text>
          </View>
          <View style={styles.arrowContainer}>
            <Text style={styles.arrow}>→</Text>
          </View>
        </TouchableOpacity>

        {/* Away Team Strategy */}
        <TouchableOpacity
          style={styles.strategyCard}
          onPress={() => handleStrategySelect(STRATEGIES.AWAY_TEAM)}
          activeOpacity={0.7}
        >
          <View style={styles.strategyIconContainer}>
            <Text style={styles.strategyIcon}>{STRATEGY_INFO[STRATEGIES.AWAY_TEAM].icon}</Text>
          </View>
          <View style={styles.strategyContent}>
            <Text style={styles.strategyName}>
              {STRATEGY_INFO[STRATEGIES.AWAY_TEAM].name}
            </Text>
            <Text style={styles.strategyDescription}>
              {STRATEGY_INFO[STRATEGIES.AWAY_TEAM].description}
            </Text>
          </View>
          <View style={styles.arrowContainer}>
            <Text style={styles.arrow}>→</Text>
          </View>
        </TouchableOpacity>

        {/* Spread Coverage Strategy */}
        <TouchableOpacity
          style={styles.strategyCard}
          onPress={() => handleStrategySelect(STRATEGIES.SPREAD_COVERAGE)}
          activeOpacity={0.7}
        >
          <View style={styles.strategyIconContainer}>
            <Text style={styles.strategyIcon}>
              {STRATEGY_INFO[STRATEGIES.SPREAD_COVERAGE].icon}
            </Text>
          </View>
          <View style={styles.strategyContent}>
            <Text style={styles.strategyName}>
              {STRATEGY_INFO[STRATEGIES.SPREAD_COVERAGE].name}
            </Text>
            <Text style={styles.strategyDescription}>
              {STRATEGY_INFO[STRATEGIES.SPREAD_COVERAGE].description}
            </Text>
          </View>
          <View style={styles.arrowContainer}>
            <Text style={styles.arrow}>→</Text>
          </View>
        </TouchableOpacity>

        {/* Elite Team Strategy */}
        <TouchableOpacity
          style={[styles.strategyCard, styles.eliteTeamCard]}
          onPress={() => navigation.navigate('EliteTeams')}
          activeOpacity={0.7}
        >
          <View style={[styles.strategyIconContainer, styles.eliteIconContainer]}>
            <Text style={styles.strategyIcon}>
              {STRATEGY_INFO[STRATEGIES.ELITE_TEAM].icon}
            </Text>
          </View>
          <View style={styles.strategyContent}>
            <Text style={styles.strategyName}>
              {STRATEGY_INFO[STRATEGIES.ELITE_TEAM].name}
            </Text>
            <Text style={styles.strategyDescription}>
              {STRATEGY_INFO[STRATEGIES.ELITE_TEAM].description}
            </Text>
          </View>
          <View style={styles.arrowContainer}>
            <Text style={styles.arrow}>→</Text>
          </View>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Predictions are based on historical spread coverage data
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  contentContainer: {
    padding: 20,
  },
  header: {
    marginBottom: 32,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  strategiesContainer: {
    marginBottom: 24,
  },
  strategyCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  strategyIconContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#E3F2FD',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  strategyIcon: {
    fontSize: 32,
  },
  strategyContent: {
    flex: 1,
  },
  strategyName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  strategyDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  arrowContainer: {
    marginLeft: 12,
  },
  arrow: {
    fontSize: 24,
    color: '#2196F3',
  },
  footer: {
    marginTop: 16,
    padding: 16,
    backgroundColor: '#fff',
    borderRadius: 8,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
  },
  eliteTeamCard: {
    borderColor: '#FFD700',
    borderWidth: 2,
  },
  eliteIconContainer: {
    backgroundColor: '#FFF8E1',
  },
});

export default HomeScreen;

