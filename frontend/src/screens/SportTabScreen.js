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
import { SPORTS, SPORT_NAMES } from '../constants/api';
import PredictionCard from '../components/PredictionCard';

const SportTabScreen = ({ sport, strategy, onRefresh: parentRefresh, refreshing: parentRefreshing }) => {
  const [games, setGames] = useState([]);
  const [filteredGames, setFilteredGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const sportName = SPORT_NAMES[sport] || sport.toUpperCase();

  useEffect(() => {
    loadGames();
  }, [sport, strategy]);

  useEffect(() => {
    // Filter games whenever games or strategy changes
    if (games.length > 0) {
      const filtered = filterGamesByStrategy(games, strategy);
      setFilteredGames(filtered);
    } else {
      setFilteredGames([]);
    }
  }, [games, strategy]);

  const loadGames = async () => {
    try {
      setError(null);
      setLoading(true);
      const predictionData = await getPredictions(sport);
      
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

  return (
    <View style={styles.container}>
      <FlatList
        data={filteredGames}
        renderItem={({ item }) => <PredictionCard game={item} />}
        keyExtractor={(item, index) => `game-${sport}-${index}-${item.home_team}-${item.away_team}`}
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

