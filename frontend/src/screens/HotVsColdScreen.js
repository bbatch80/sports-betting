/**
 * HotVsColdScreen Component
 * Displays Hot vs Cold opportunities across all sports
 * Hot teams (60%+ coverage in last 5) vs Cold teams (40%- coverage in last 5)
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  RefreshControl,
  ActivityIndicator,
  TouchableOpacity,
} from 'react-native';
import { getHotVsColdOpportunities } from '../services/apiService';
import { SPORT_NAMES } from '../constants/api';

const HotVsColdScreen = () => {
  const [opportunities, setOpportunities] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSport, setSelectedSport] = useState(null); // null = all sports

  const fetchOpportunities = useCallback(async () => {
    try {
      setError(null);
      const data = await getHotVsColdOpportunities(selectedSport);
      setOpportunities(data.opportunities || []);
      setSummary(data.summary || null);
    } catch (err) {
      console.error('Error fetching hot vs cold opportunities:', err);
      setError(err.message || 'Failed to load opportunities');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedSport]);

  useEffect(() => {
    fetchOpportunities();
  }, [fetchOpportunities]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchOpportunities();
  }, [fetchOpportunities]);

  const formatSpread = (spread) => {
    if (spread === null || spread === undefined) return 'N/A';
    return spread > 0 ? `+${spread}` : spread.toString();
  };

  const formatPercentage = (pct) => {
    if (pct === null || pct === undefined) return 'N/A';
    return `${(pct * 100).toFixed(0)}%`;
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return '';
    try {
      const date = new Date(timeStr);
      return date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      });
    } catch {
      return timeStr;
    }
  };

  const renderOpportunityCard = ({ item }) => {
    const isHotHome = item.hot_team_position === 'home';

    return (
      <View style={styles.card}>
        {/* Sport Badge */}
        <View style={styles.sportBadge}>
          <Text style={styles.sportBadgeText}>
            {item.sport_name || item.sport?.toUpperCase()}
          </Text>
        </View>

        {/* Game Time */}
        <Text style={styles.gameTime}>{formatTime(item.game_time_est)}</Text>

        {/* Teams */}
        <View style={styles.teamsContainer}>
          <View style={[styles.teamRow, isHotHome && styles.hotTeamRow]}>
            <Text style={[styles.teamName, isHotHome && styles.hotTeamName]}>
              {isHotHome ? '🔥 ' : ''}{item.home_team}
            </Text>
            {isHotHome && (
              <View style={styles.formBadge}>
                <Text style={styles.hotBadge}>HOT</Text>
                <Text style={styles.formText}>{item.hot_team_last_5_results}</Text>
              </View>
            )}
            {!isHotHome && (
              <View style={styles.formBadge}>
                <Text style={styles.coldBadge}>COLD</Text>
                <Text style={styles.formText}>{item.cold_team_last_5_results}</Text>
              </View>
            )}
          </View>
          <Text style={styles.vsText}>vs</Text>
          <View style={[styles.teamRow, !isHotHome && styles.hotTeamRow]}>
            <Text style={[styles.teamName, !isHotHome && styles.hotTeamName]}>
              {!isHotHome ? '🔥 ' : ''}{item.away_team}
            </Text>
            {!isHotHome && (
              <View style={styles.formBadge}>
                <Text style={styles.hotBadge}>HOT</Text>
                <Text style={styles.formText}>{item.hot_team_last_5_results}</Text>
              </View>
            )}
            {isHotHome && (
              <View style={styles.formBadge}>
                <Text style={styles.coldBadge}>COLD</Text>
                <Text style={styles.formText}>{item.cold_team_last_5_results}</Text>
              </View>
            )}
          </View>
        </View>

        {/* Current Spread */}
        <View style={styles.spreadContainer}>
          <Text style={styles.spreadLabel}>Current Spread</Text>
          <Text style={styles.spreadValue}>{formatSpread(item.current_spread)}</Text>
        </View>

        {/* Form Stats */}
        <View style={styles.statsContainer}>
          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Hot Team Form</Text>
              <Text style={[styles.statValue, styles.hotValue]}>
                {formatPercentage(item.hot_team_last_5_coverage_pct)}
              </Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Cold Team Form</Text>
              <Text style={[styles.statValue, styles.coldValue]}>
                {formatPercentage(item.cold_team_last_5_coverage_pct)}
              </Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Form Diff</Text>
              <Text style={styles.statValue}>
                {formatPercentage(item.form_differential)}
              </Text>
            </View>
          </View>
        </View>

        {/* Bet Recommendation */}
        <View style={styles.recommendationContainer}>
          <Text style={styles.recommendationLabel}>BET ON</Text>
          <Text style={styles.recommendationTeam}>{item.bet_on}</Text>
          <Text style={styles.handicapText}>({item.handicap_points}pt handicap)</Text>
        </View>
      </View>
    );
  };

  const renderSportFilter = () => {
    const sports = [null, 'nfl', 'nba', 'ncaam'];

    return (
      <View style={styles.filterContainer}>
        {sports.map((sport) => (
          <TouchableOpacity
            key={sport || 'all'}
            style={[
              styles.filterButton,
              selectedSport === sport && styles.filterButtonActive,
            ]}
            onPress={() => {
              setSelectedSport(sport);
              setLoading(true);
            }}
          >
            <Text
              style={[
                styles.filterButtonText,
                selectedSport === sport && styles.filterButtonTextActive,
              ]}
            >
              {sport ? SPORT_NAMES[sport] || sport.toUpperCase() : 'All'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  if (loading && !refreshing) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#FF5722" />
        <Text style={styles.loadingText}>Loading hot vs cold matchups...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={fetchOpportunities}>
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Hot vs Cold Matchups</Text>
        <Text style={styles.headerSubtitle}>
          Hot teams (60%+ form) vs Cold teams (40%- form)
        </Text>
        {summary && (
          <Text style={styles.summaryText}>
            {summary.total_opportunities} opportunities across {summary.sports_with_opportunities} sport(s)
          </Text>
        )}
      </View>

      {/* Sport Filter */}
      {renderSportFilter()}

      {/* Opportunities List */}
      {opportunities.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>🔍</Text>
          <Text style={styles.emptyTitle}>No Hot vs Cold Matchups Today</Text>
          <Text style={styles.emptySubtitle}>
            Check back later or try a different sport
          </Text>
        </View>
      ) : (
        <FlatList
          data={opportunities}
          renderItem={renderOpportunityCard}
          keyExtractor={(item, index) => `${item.hot_team}-${item.cold_team}-${index}`}
          contentContainerStyle={styles.listContainer}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  errorText: {
    fontSize: 16,
    color: '#f44336',
    textAlign: 'center',
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: '#FF5722',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    backgroundColor: '#fff',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
  },
  summaryText: {
    fontSize: 14,
    color: '#FF5722',
    fontWeight: '600',
  },
  filterContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 8,
    backgroundColor: '#f0f0f0',
  },
  filterButtonActive: {
    backgroundColor: '#FF5722',
  },
  filterButtonText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  filterButtonTextActive: {
    color: '#fff',
  },
  listContainer: {
    padding: 12,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderLeftWidth: 4,
    borderLeftColor: '#FF5722',
  },
  sportBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: '#FFF3E0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  sportBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#E64A19',
  },
  gameTime: {
    fontSize: 12,
    color: '#999',
    marginBottom: 8,
  },
  teamsContainer: {
    marginBottom: 12,
  },
  teamRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 4,
  },
  hotTeamRow: {
    backgroundColor: '#FFF3E0',
    marginHorizontal: -8,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  teamName: {
    fontSize: 16,
    color: '#333',
    flex: 1,
  },
  hotTeamName: {
    fontWeight: 'bold',
    color: '#E64A19',
  },
  formBadge: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  hotBadge: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#FF5722',
    backgroundColor: '#FFCCBC',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginRight: 4,
  },
  coldBadge: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#1976D2',
    backgroundColor: '#BBDEFB',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginRight: 4,
  },
  formText: {
    fontSize: 12,
    color: '#666',
    fontWeight: '600',
  },
  vsText: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
    marginVertical: 2,
  },
  spreadContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    padding: 10,
    borderRadius: 8,
    marginBottom: 12,
  },
  spreadLabel: {
    fontSize: 14,
    color: '#666',
  },
  spreadValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  statsContainer: {
    marginBottom: 12,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statItem: {
    alignItems: 'center',
    flex: 1,
  },
  statLabel: {
    fontSize: 11,
    color: '#999',
    marginBottom: 2,
  },
  statValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  hotValue: {
    color: '#FF5722',
  },
  coldValue: {
    color: '#1976D2',
  },
  recommendationContainer: {
    backgroundColor: '#E8F5E9',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  recommendationLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  recommendationTeam: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginBottom: 2,
  },
  handicapText: {
    fontSize: 12,
    color: '#666',
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  emptySubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
});

export default HotVsColdScreen;
