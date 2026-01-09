/**
 * StrategyPerformanceScreen Component
 * Displays cumulative win rate charts and statistics for all betting strategies
 * Shows historical performance from season start to current date
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  ActivityIndicator,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { getStrategyPerformance } from '../services/apiService';
import { SPORT_NAMES, STRATEGY_INFO } from '../constants/api';

const screenWidth = Dimensions.get('window').width;

// Strategy colors for chart lines
const STRATEGY_COLORS = {
  // 11-point handicap strategies
  home_focus: '#2196F3',           // Blue
  away_focus: '#4CAF50',           // Green
  elite_team_winpct: '#9C27B0',    // Purple
  elite_team_coverage: '#673AB7',  // Deep Purple
  hot_vs_cold_3: '#E91E63',        // Pink
  hot_vs_cold_5: '#F44336',        // Red
  hot_vs_cold_7: '#FF5722',        // Deep Orange
  opponent_perfect_form: '#00BCD4', // Cyan
  // 0-point handicap strategies
  coverage_based: '#FF9800',       // Orange
  common_opponent: '#795548',      // Brown
};

// Filter strategies by handicap value
const filterStrategiesByHandicap = (strategies, filter) => {
  if (filter === 'all') return strategies;

  const targetHandicap = parseInt(filter, 10);
  const filtered = {};

  Object.entries(strategies).forEach(([key, value]) => {
    const info = STRATEGY_INFO[key];
    if (info && info.handicap === targetHandicap) {
      filtered[key] = value;
    }
  });

  return filtered;
};

const StrategyPerformanceScreen = () => {
  const [performanceData, setPerformanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSport, setSelectedSport] = useState('nba');
  const [visibleStrategies, setVisibleStrategies] = useState({});
  const [handicapFilter, setHandicapFilter] = useState('all'); // 'all' | '11' | '0'

  const fetchPerformance = useCallback(async () => {
    try {
      setError(null);
      const data = await getStrategyPerformance(selectedSport);
      setPerformanceData(data);

      // Initialize all strategies as visible
      const visible = {};
      Object.keys(data.strategies || {}).forEach(key => {
        visible[key] = true;
      });
      setVisibleStrategies(visible);
    } catch (err) {
      console.error('Error fetching strategy performance:', err);
      setError(err.message || 'Failed to load strategy performance');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [selectedSport]);

  useEffect(() => {
    setLoading(true);
    fetchPerformance();
  }, [fetchPerformance]);

  // Reset visible strategies when handicap filter changes
  useEffect(() => {
    if (performanceData?.strategies) {
      const filtered = filterStrategiesByHandicap(performanceData.strategies, handicapFilter);
      const visible = {};
      Object.keys(filtered).forEach(key => {
        visible[key] = true;
      });
      setVisibleStrategies(visible);
    }
  }, [handicapFilter, performanceData]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchPerformance();
  }, [fetchPerformance]);

  const toggleStrategy = (strategyKey) => {
    setVisibleStrategies(prev => ({
      ...prev,
      [strategyKey]: !prev[strategyKey],
    }));
  };

  const formatWinRate = (rate) => {
    if (rate === null || rate === undefined) return 'N/A';
    return `${(rate * 100).toFixed(1)}%`;
  };

  const formatROI = (roi) => {
    if (roi === null || roi === undefined) return 'N/A';
    const prefix = roi >= 0 ? '+' : '';
    return `${prefix}${roi.toFixed(1)}%`;
  };

  const getChartData = () => {
    if (!performanceData?.strategies) return null;

    const strategies = filterStrategiesByHandicap(performanceData.strategies, handicapFilter);
    const visibleKeys = Object.keys(strategies).filter(key => visibleStrategies[key]);

    if (visibleKeys.length === 0) return null;

    // Get all dates from the first visible strategy
    const firstStrategy = strategies[visibleKeys[0]];
    const chartData = firstStrategy?.chart_data || [];

    if (chartData.length === 0) return null;

    // Sample data points if too many (max 15 labels for readability)
    const maxLabels = 15;
    const step = Math.max(1, Math.floor(chartData.length / maxLabels));
    const sampledIndices = [];
    for (let i = 0; i < chartData.length; i += step) {
      sampledIndices.push(i);
    }
    // Always include the last point
    if (sampledIndices[sampledIndices.length - 1] !== chartData.length - 1) {
      sampledIndices.push(chartData.length - 1);
    }

    // Build labels (dates)
    const labels = sampledIndices.map(i => {
      const date = chartData[i]?.date;
      if (!date) return '';
      // Format as MM/DD
      const parts = date.split('-');
      return `${parts[1]}/${parts[2]}`;
    });

    // Build datasets
    const datasets = visibleKeys.map(key => {
      const strat = strategies[key];
      const data = sampledIndices.map(i => {
        const point = strat.chart_data?.[i];
        return point ? point.rate * 100 : 50;
      });

      return {
        data,
        color: (opacity = 1) => STRATEGY_COLORS[key] || '#999',
        strokeWidth: 2,
      };
    });

    return {
      labels,
      datasets,
      legend: visibleKeys.map(key => strategies[key]?.name || key),
    };
  };

  const renderSportTabs = () => {
    const sports = ['nfl', 'nba', 'ncaam'];

    return (
      <View style={styles.tabContainer}>
        {sports.map((sport) => (
          <TouchableOpacity
            key={sport}
            style={[
              styles.tab,
              selectedSport === sport && styles.tabActive,
            ]}
            onPress={() => {
              setSelectedSport(sport);
              setLoading(true);
            }}
          >
            <Text
              style={[
                styles.tabText,
                selectedSport === sport && styles.tabTextActive,
              ]}
            >
              {SPORT_NAMES[sport] || sport.toUpperCase()}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const renderHandicapFilter = () => {
    const filters = [
      { key: 'all', label: 'All' },
      { key: '11', label: '11pt Handicap' },
      { key: '0', label: '0pt Handicap' },
    ];

    return (
      <View style={styles.handicapFilterContainer}>
        <Text style={styles.filterLabel}>Handicap Filter:</Text>
        <View style={styles.segmentedControl}>
          {filters.map((filter) => (
            <TouchableOpacity
              key={filter.key}
              style={[
                styles.segmentButton,
                handicapFilter === filter.key && styles.segmentButtonActive,
              ]}
              onPress={() => setHandicapFilter(filter.key)}
            >
              <Text style={[
                styles.segmentButtonText,
                handicapFilter === filter.key && styles.segmentButtonTextActive,
              ]}>
                {filter.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  };

  const renderStrategyLegend = () => {
    if (!performanceData?.strategies) return null;

    const filteredStrategies = filterStrategiesByHandicap(performanceData.strategies, handicapFilter);

    return (
      <View style={styles.legendContainer}>
        <Text style={styles.legendTitle}>Toggle Strategies</Text>
        <View style={styles.legendItems}>
          {Object.entries(filteredStrategies).map(([key, strat]) => (
            <TouchableOpacity
              key={key}
              style={[
                styles.legendItem,
                !visibleStrategies[key] && styles.legendItemDisabled,
              ]}
              onPress={() => toggleStrategy(key)}
            >
              <View
                style={[
                  styles.legendColor,
                  { backgroundColor: STRATEGY_COLORS[key] || '#999' },
                  !visibleStrategies[key] && styles.legendColorDisabled,
                ]}
              />
              <Text
                style={[
                  styles.legendText,
                  !visibleStrategies[key] && styles.legendTextDisabled,
                ]}
                numberOfLines={1}
              >
                {strat.name}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  };

  const renderChart = () => {
    const chartData = getChartData();

    if (!chartData) {
      return (
        <View style={styles.noChartContainer}>
          <Text style={styles.noChartText}>
            No chart data available. Select at least one strategy.
          </Text>
        </View>
      );
    }

    return (
      <View style={styles.chartContainer}>
        <Text style={styles.chartTitle}>Cumulative Win Rate Over Time</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={true}>
          <LineChart
            data={chartData}
            width={Math.max(screenWidth - 32, chartData.labels.length * 50)}
            height={220}
            yAxisSuffix="%"
            yAxisInterval={1}
            chartConfig={{
              backgroundColor: '#fff',
              backgroundGradientFrom: '#fff',
              backgroundGradientTo: '#fff',
              decimalPlaces: 0,
              color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(102, 102, 102, ${opacity})`,
              style: {
                borderRadius: 16,
              },
              propsForDots: {
                r: '3',
              },
              propsForBackgroundLines: {
                strokeDasharray: '',
                stroke: '#e0e0e0',
              },
            }}
            bezier
            style={styles.chart}
            fromZero={false}
            segments={4}
          />
        </ScrollView>
        <Text style={styles.chartSubtitle}>
          50% line represents break-even
        </Text>
      </View>
    );
  };

  const renderStrategyCard = (key, strat) => {
    const isPositiveROI = strat.roi >= 0;
    const isWinning = strat.win_rate > 0.5;

    return (
      <View
        key={key}
        style={[
          styles.strategyCard,
          { borderLeftColor: STRATEGY_COLORS[key] || '#999' },
        ]}
      >
        <View style={styles.cardHeader}>
          <Text style={styles.strategyName}>{strat.name}</Text>
          <View
            style={[
              styles.streakBadge,
              strat.streak_type === 'win' ? styles.winStreak : styles.lossStreak,
            ]}
          >
            <Text style={styles.streakText}>
              {strat.current_streak} {strat.streak_type === 'win' ? 'W' : 'L'}
            </Text>
          </View>
        </View>

        <View style={styles.cardStats}>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Record</Text>
            <Text style={styles.statValue}>
              {strat.wins}-{strat.losses}
            </Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Win Rate</Text>
            <Text
              style={[
                styles.statValue,
                isWinning ? styles.positive : styles.negative,
              ]}
            >
              {formatWinRate(strat.win_rate)}
            </Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>ROI</Text>
            <Text
              style={[
                styles.statValue,
                isPositiveROI ? styles.positive : styles.negative,
              ]}
            >
              {formatROI(strat.roi)}
            </Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Units</Text>
            <Text
              style={[
                styles.statValue,
                strat.units_won >= 0 ? styles.positive : styles.negative,
              ]}
            >
              {strat.units_won >= 0 ? '+' : ''}{strat.units_won?.toFixed(1) || '0'}
            </Text>
          </View>
        </View>
      </View>
    );
  };

  if (loading && !refreshing) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#2196F3" />
        <Text style={styles.loadingText}>Loading strategy performance...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity style={styles.retryButton} onPress={fetchPerformance}>
          <Text style={styles.retryButtonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const allStrategies = performanceData?.strategies || {};
  const strategies = filterStrategiesByHandicap(allStrategies, handicapFilter);
  const hasData = Object.keys(strategies).length > 0;

  return (
    <View style={styles.container}>
      {/* Sport Tabs */}
      {renderSportTabs()}

      {/* Handicap Filter */}
      {renderHandicapFilter()}

      <ScrollView
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        contentContainerStyle={styles.scrollContent}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Strategy Performance</Text>
          <Text style={styles.headerSubtitle}>
            {performanceData?.sport_name || selectedSport.toUpperCase()} - Season {performanceData?.season_start?.split('-')[0] || ''}
          </Text>
          {performanceData?.last_updated && (
            <Text style={styles.lastUpdated}>
              Last updated: {new Date(performanceData.last_updated).toLocaleDateString()}
            </Text>
          )}
        </View>

        {!hasData ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyIcon}>📊</Text>
            <Text style={styles.emptyTitle}>No Performance Data Yet</Text>
            <Text style={styles.emptySubtitle}>
              Strategy tracking will begin once predictions are matched to game results.
            </Text>
          </View>
        ) : (
          <>
            {/* Strategy Legend */}
            {renderStrategyLegend()}

            {/* Line Chart */}
            {renderChart()}

            {/* Strategy Cards */}
            <View style={styles.cardsContainer}>
              <Text style={styles.sectionTitle}>Strategy Details</Text>
              {Object.entries(strategies)
                .sort((a, b) => (b[1].win_rate || 0) - (a[1].win_rate || 0))
                .map(([key, strat]) => renderStrategyCard(key, strat))}
            </View>
          </>
        )}
      </ScrollView>
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
  tabContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  tab: {
    flex: 1,
    paddingVertical: 14,
    alignItems: 'center',
    borderBottomWidth: 3,
    borderBottomColor: 'transparent',
  },
  tabActive: {
    borderBottomColor: '#2196F3',
  },
  tabText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  tabTextActive: {
    color: '#2196F3',
  },
  handicapFilterContainer: {
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  filterLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
    fontWeight: '500',
  },
  segmentedControl: {
    flexDirection: 'row',
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 2,
  },
  segmentButton: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    alignItems: 'center',
  },
  segmentButtonActive: {
    backgroundColor: '#2196F3',
  },
  segmentButtonText: {
    fontSize: 13,
    color: '#666',
    fontWeight: '500',
  },
  segmentButtonTextActive: {
    color: '#fff',
  },
  scrollContent: {
    paddingBottom: 24,
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
  },
  lastUpdated: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  legendContainer: {
    backgroundColor: '#fff',
    padding: 16,
    marginTop: 8,
  },
  legendTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  legendItems: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    paddingHorizontal: 10,
    marginRight: 8,
    marginBottom: 8,
    backgroundColor: '#f5f5f5',
    borderRadius: 16,
  },
  legendItemDisabled: {
    backgroundColor: '#e0e0e0',
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 6,
  },
  legendColorDisabled: {
    opacity: 0.3,
  },
  legendText: {
    fontSize: 12,
    color: '#333',
  },
  legendTextDisabled: {
    color: '#999',
  },
  chartContainer: {
    backgroundColor: '#fff',
    padding: 16,
    marginTop: 8,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  chart: {
    borderRadius: 8,
  },
  chartSubtitle: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
    marginTop: 8,
  },
  noChartContainer: {
    backgroundColor: '#fff',
    padding: 40,
    marginTop: 8,
    alignItems: 'center',
  },
  noChartText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  cardsContainer: {
    padding: 16,
    paddingTop: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
    marginTop: 8,
  },
  strategyCard: {
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
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  strategyName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  streakBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  winStreak: {
    backgroundColor: '#E8F5E9',
  },
  lossStreak: {
    backgroundColor: '#FFEBEE',
  },
  streakText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
  },
  cardStats: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statBox: {
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
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
    marginTop: 40,
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

export default StrategyPerformanceScreen;
