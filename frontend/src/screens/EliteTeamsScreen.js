/**
 * EliteTeamsScreen Component
 * Displays elite team opportunities across all sports
 * Elite teams are top 25% by win% with good recent form (last 5 games)
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
import { getEliteTeams } from '../services/apiService';
import { SPORT_NAMES } from '../constants/api';

const EliteTeamsScreen = () => {
  const [eliteTeams, setEliteTeams] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSport, setSelectedSport] = useState(null); // null = all sports

  const fetchEliteTeams = useCallback(async () => {
    try {
      setError(null);
      const data = await getEliteTeams(selectedSport);
      setEliteTeams(data.elite_teams || []);
      setSummary(data.summary || null);
    } catch (err) {
      console.error('Error fetching elite teams:', err);
      setError(err.message || 'Failed to load elite teams');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedSport]);

  useEffect(() => {
    fetchEliteTeams();
  }, [fetchEliteTeams]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchEliteTeams();
  }, [fetchEliteTeams]);

  const formatSpread = (spread) => {
    if (spread === null || spread === undefined) return 'N/A';
    return spread > 0 ? `+${spread}` : spread.toString();
  };

  const formatWinPct = (pct) => {
    if (pct === null || pct === undefined) return 'N/A';
    return `${pct.toFixed(1)}%`;
  };

  const formatPointDiff = (diff) => {
    if (diff === null || diff === undefined) return 'N/A';
    return diff > 0 ? `+${diff.toFixed(1)}` : diff.toFixed(1);
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
    const isHome = item.elite_team_position === 'home';
    const tierColors = {
      Elite: '#4CAF50',
      Mid: '#FF9800',
      Bottom: '#f44336',
      Unknown: '#9E9E9E',
    };

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
          <View style={[styles.teamRow, isHome && styles.eliteTeamRow]}>
            <Text style={[styles.teamName, isHome && styles.eliteTeamName]}>
              {isHome ? '⭐ ' : ''}{item.home_team}
            </Text>
            {isHome && <Text style={styles.eliteBadge}>ELITE</Text>}
          </View>
          <Text style={styles.vsText}>vs</Text>
          <View style={[styles.teamRow, !isHome && styles.eliteTeamRow]}>
            <Text style={[styles.teamName, !isHome && styles.eliteTeamName]}>
              {!isHome ? '⭐ ' : ''}{item.away_team}
            </Text>
            {!isHome && <Text style={styles.eliteBadge}>ELITE</Text>}
          </View>
        </View>

        {/* Current Spread */}
        <View style={styles.spreadContainer}>
          <Text style={styles.spreadLabel}>Current Spread</Text>
          <Text style={styles.spreadValue}>{formatSpread(item.current_spread)}</Text>
        </View>

        {/* Elite Team Stats */}
        <View style={styles.statsContainer}>
          <Text style={styles.statsHeader}>Elite Team: {item.elite_team}</Text>

          <View style={styles.statsRow}>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Win %</Text>
              <Text style={styles.statValue}>{formatWinPct(item.elite_team_win_pct)}</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Pt Diff</Text>
              <Text style={styles.statValue}>{formatPointDiff(item.elite_team_point_diff)}</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Last 5 Form</Text>
              <Text style={[
                styles.statValue,
                item.elite_team_last_5_spread > 0 ? styles.positive : styles.negative
              ]}>
                {formatPointDiff(item.elite_team_last_5_spread)}
              </Text>
            </View>
          </View>
        </View>

        {/* Opponent Info */}
        <View style={styles.opponentContainer}>
          <Text style={styles.opponentLabel}>
            vs {item.opponent} ({item.opponent_tier} tier - {formatWinPct(item.opponent_win_pct)})
          </Text>
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
        <ActivityIndicator size="large" color="#2196F3" />
        <Text style={styles.loadingText}>Loading elite teams...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={fetchEliteTeams}>
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Elite Team Opportunities</Text>
        <Text style={styles.headerSubtitle}>
          Top-tier teams in good form with games today
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
      {eliteTeams.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>🔍</Text>
          <Text style={styles.emptyTitle}>No Elite Team Opportunities Today</Text>
          <Text style={styles.emptySubtitle}>
            Check back later or try a different sport
          </Text>
        </View>
      ) : (
        <FlatList
          data={eliteTeams}
          renderItem={renderOpportunityCard}
          keyExtractor={(item, index) => `${item.elite_team}-${item.opponent}-${index}`}
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
    backgroundColor: '#2196F3',
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
    color: '#2196F3',
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
    backgroundColor: '#2196F3',
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
    borderLeftColor: '#FFD700',
  },
  sportBadge: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: '#E3F2FD',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  sportBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1976D2',
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
    paddingVertical: 4,
  },
  eliteTeamRow: {
    backgroundColor: '#FFF8E1',
    marginHorizontal: -8,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
  teamName: {
    fontSize: 16,
    color: '#333',
    flex: 1,
  },
  eliteTeamName: {
    fontWeight: 'bold',
    color: '#333',
  },
  eliteBadge: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#FFB300',
    backgroundColor: '#FFF3E0',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
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
  statsHeader: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
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
  positive: {
    color: '#4CAF50',
  },
  negative: {
    color: '#f44336',
  },
  opponentContainer: {
    borderTopWidth: 1,
    borderTopColor: '#eee',
    paddingTop: 8,
  },
  opponentLabel: {
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic',
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

export default EliteTeamsScreen;
