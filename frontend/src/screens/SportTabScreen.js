/**
 * SportTabScreen Component
 * Displays filtered predictions for a specific sport based on strategy
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { getPredictions } from '../services/apiService';
import { filterGamesByStrategy } from '../utils/strategyFilters';
import { SPORTS, SPORT_NAMES, STRATEGIES } from '../constants/api';
import PredictionCard from '../components/PredictionCard';
import StrategyOpportunityCard from '../components/StrategyOpportunityCard';

// Strategies that use pre-computed opportunities from backend
const STRATEGY_BASED_STRATEGIES = [
  STRATEGIES.ELITE_TEAM,
  STRATEGIES.ELITE_TEAM_WINPCT,
  STRATEGIES.ELITE_TEAM_COVERAGE,
  STRATEGIES.HOT_VS_COLD,
  STRATEGIES.HOT_VS_COLD_3,
  STRATEGIES.HOT_VS_COLD_5,
  STRATEGIES.HOT_VS_COLD_7,
  STRATEGIES.OPPONENT_PERFECT_FORM,
];

// Map strategy names to their key in the strategies object from API
const STRATEGY_API_KEYS = {
  [STRATEGIES.ELITE_TEAM]: 'elite_team',
  [STRATEGIES.ELITE_TEAM_WINPCT]: 'elite_team_winpct',
  [STRATEGIES.ELITE_TEAM_COVERAGE]: 'elite_team_coverage',
  [STRATEGIES.HOT_VS_COLD]: 'hot_vs_cold_5',
  [STRATEGIES.HOT_VS_COLD_3]: 'hot_vs_cold_3',
  [STRATEGIES.HOT_VS_COLD_5]: 'hot_vs_cold_5',
  [STRATEGIES.HOT_VS_COLD_7]: 'hot_vs_cold_7',
  [STRATEGIES.OPPONENT_PERFECT_FORM]: 'opponent_perfect_form',
};

const SportTabScreen = ({ sport, strategy, date, onRefresh: parentRefresh, refreshing: parentRefreshing }) => {
  const [games, setGames] = useState([]);
  const [filteredGames, setFilteredGames] = useState([]);
  const [strategiesData, setStrategiesData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hasResults, setHasResults] = useState(false);

  const sportName = SPORT_NAMES[sport] || sport.toUpperCase();
  const isStrategyBased = STRATEGY_BASED_STRATEGIES.includes(strategy);

  useEffect(() => {
    loadGames();
  }, [sport, strategy, date]);

  useEffect(() => {
    // Filter games whenever games/strategiesData or strategy changes
    if (isStrategyBased && strategiesData) {
      // For strategy-based strategies, get opportunities from strategies object
      const strategyKey = STRATEGY_API_KEYS[strategy] || strategy;
      const strategyData = strategiesData[strategyKey];
      const opportunities = strategyData?.opportunities || [];

      // Add sport info to each opportunity
      const opportunitiesWithSport = opportunities.map(opp => ({
        ...opp,
        sport: sport,
        sport_name: sportName,
      }));
      setFilteredGames(opportunitiesWithSport);
    } else if (games.length > 0) {
      const filtered = filterGamesByStrategy(games, strategy);
      setFilteredGames(filtered);
    } else {
      setFilteredGames([]);
    }
  }, [games, strategiesData, strategy, isStrategyBased]);

  const loadGames = async () => {
    try {
      setError(null);
      setLoading(true);
      // Pass date to getPredictions - null for today, date string for past dates
      const predictionData = await getPredictions(sport, date);

      // Track if results are available for past dates
      setHasResults(predictionData.has_results || false);

      // Store strategies data for strategy-based filtering
      if (predictionData.strategies) {
        setStrategiesData(predictionData.strategies);
      } else {
        setStrategiesData(null);
      }

      // Handle both single sport response and all sports response
      if (predictionData.games && Array.isArray(predictionData.games)) {
        // Add sport info to each game
        const gamesWithSport = predictionData.games.map(game => ({
          ...game,
          sport: sport,
          sport_name: sportName,
        }));
        setGames(gamesWithSport);
      } else if (predictionData.error) {
        setError(predictionData.message || 'No predictions available');
        setGames([]);
        setStrategiesData(null);
      } else {
        setGames([]);
      }
    } catch (err) {
      console.error(`Error loading ${sport} games:`, err);
      // Show more helpful error messages
      if (err.message && err.message.includes('API Gateway URL')) {
        setError('API not configured. Please update src/constants/api.js with your API Gateway URL.');
      } else if (err.message) {
        setError(err.message);
      } else {
        setError(`Failed to load ${sportName} predictions. Check your API configuration.`);
      }
      setGames([]);
      setStrategiesData(null);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    if (parentRefresh) {
      parentRefresh();
    }
    await loadGames();
  };

  const renderEmpty = () => {
    if (loading) return null;
    
    return (
      <View style={styles.emptyContainer}>
        <Text style={styles.emptyIcon}>📊</Text>
        <Text style={styles.emptyTitle}>No {sportName} predictions found</Text>
        <Text style={styles.emptyText}>
          No {sportName} games match this strategy criteria at the moment.
        </Text>
        <Text style={styles.emptySubtext}>
          Try refreshing or check back later for new predictions.
        </Text>
      </View>
    );
  };

  if (loading && games.length === 0) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#2196F3" />
        <Text style={styles.loadingText}>Loading {sportName} predictions...</Text>
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
      </View>
    );
  }

  if (error && games.length === 0 && !loading) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.errorIcon}>⚠️</Text>
        <Text style={styles.errorTitle}>Error Loading {sportName} Predictions</Text>
        <Text style={styles.errorText}>{error}</Text>
      </View>
    );
  }

  // Render the appropriate card based on strategy type
  const renderCard = ({ item }) => {
    if (isStrategyBased) {
      return <StrategyOpportunityCard opportunity={item} strategyType={strategy} />;
    }
    return <PredictionCard game={item} />;
  };

  return (
    <View style={styles.container}>
      <FlatList
        data={filteredGames}
        renderItem={renderCard}
        keyExtractor={(item, index) => `game-${sport}-${strategy}-${index}-${item.home_team}-${item.away_team}`}
        ListEmptyComponent={renderEmpty}
        refreshControl={
          <RefreshControl
            refreshing={parentRefreshing || false}
            onRefresh={onRefresh}
          />
        }
        contentContainerStyle={
          filteredGames.length === 0 ? styles.emptyListContainer : styles.listContainer
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#666',
  },
  listContainer: {
    paddingBottom: 20,
  },
  emptyListContainer: {
    flexGrow: 1,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
  errorContainer: {
    marginTop: 20,
    padding: 16,
    backgroundColor: '#FFEBEE',
    borderRadius: 8,
    maxWidth: 300,
  },
  errorIcon: {
    fontSize: 48,
    marginBottom: 16,
    textAlign: 'center',
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#d32f2f',
    marginBottom: 8,
    textAlign: 'center',
  },
  errorText: {
    fontSize: 14,
    color: '#d32f2f',
    textAlign: 'center',
    marginBottom: 8,
  },
});

export default SportTabScreen;

