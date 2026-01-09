/**
 * StrategyOpportunityCard Component
 * Displays a single strategy opportunity with bet recommendation
 * Shows scores and bet results for completed games
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { SPORT_NAMES } from '../constants/api';

const StrategyOpportunityCard = ({ opportunity, strategyType }) => {
  if (!opportunity) return null;

  const isCompleted = opportunity.game_completed === true;
  const hasScore = opportunity.home_score !== undefined && opportunity.away_score !== undefined;
  const sportName = SPORT_NAMES[opportunity.sport] || opportunity.sport_name || '';

  // Get result badge style based on bet result
  const getResultStyle = () => {
    if (!opportunity.bet_result) return {};
    switch (opportunity.bet_result.toLowerCase()) {
      case 'win':
        return { backgroundColor: '#4CAF50' };
      case 'loss':
        return { backgroundColor: '#F44336' };
      case 'push':
        return { backgroundColor: '#FF9800' };
      default:
        return { backgroundColor: '#9E9E9E' };
    }
  };

  const getResultText = () => {
    if (!opportunity.bet_result) return '';
    return opportunity.bet_result.toUpperCase();
  };

  // Format percentage for display
  const formatPct = (value) => {
    if (typeof value !== 'number') return 'N/A';
    // Check if value is already in percentage form (>1 means it's already %)
    if (value > 1) {
      return `${value.toFixed(1)}%`;
    }
    return `${(value * 100).toFixed(1)}%`;
  };

  // Get strategy-specific details
  const getStrategyDetails = () => {
    if (strategyType?.includes('elite_team')) {
      return {
        betOnLabel: 'Bet On Elite Team:',
        metrics: [
          { label: 'Win %', value: formatPct(opportunity.elite_team_win_pct) },
          { label: 'Last 5 Coverage', value: formatPct(opportunity.elite_team_last_5_spread) },
          { label: 'Opponent Win %', value: opportunity.opponent_win_pct ? `${opportunity.opponent_win_pct.toFixed(1)}%` : 'N/A' },
        ],
      };
    } else if (strategyType?.includes('hot_vs_cold')) {
      const lookback = opportunity.lookback || 5;
      return {
        betOnLabel: 'Bet On Hot Team:',
        metrics: [
          { label: `Hot (Last ${lookback})`, value: `${opportunity.hot_team_last_n_coverage_pct || 'N/A'}%` },
          { label: `Cold (Last ${lookback})`, value: `${opportunity.cold_team_last_n_coverage_pct || 'N/A'}%` },
          { label: 'Form Diff', value: opportunity.form_differential ? `${opportunity.form_differential.toFixed(1)}%` : 'N/A' },
        ],
        extraInfo: {
          hotStreak: opportunity.hot_team_current_streak,
          coldStreak: opportunity.cold_team_current_streak,
        },
      };
    } else if (strategyType?.includes('opponent_perfect_form')) {
      return {
        betOnLabel: 'Bet Against Perfect Form:',
        metrics: [
          { label: 'Perfect Team', value: opportunity.perfect_form_team || 'N/A' },
          { label: 'Their Form', value: opportunity.perfect_form_team_results || '5/5' },
          { label: 'Our Pick Form', value: opportunity.opponent_results || 'N/A' },
        ],
      };
    }
    return {
      betOnLabel: 'Recommended:',
      metrics: [],
    };
  };

  const details = getStrategyDetails();
  const betOn = opportunity.bet_on || opportunity.recommended_team || 'N/A';

  return (
    <View style={[styles.card, isCompleted && styles.cardCompleted]}>
      {/* Header with Sport Badge and Result Badge */}
      <View style={styles.headerRow}>
        <View style={styles.sportBadge}>
          <Text style={styles.sportText}>{sportName}</Text>
        </View>
        {isCompleted && opportunity.bet_result && (
          <View style={[styles.resultBadge, getResultStyle()]}>
            <Text style={styles.resultText}>{getResultText()}</Text>
          </View>
        )}
      </View>

      {/* Game Info */}
      <View style={styles.gameInfo}>
        <View style={styles.teamRow}>
          <View style={styles.teamContainer}>
            <Text style={styles.teamName}>{opportunity.home_team}</Text>
            {hasScore && (
              <Text style={styles.scoreText}>{opportunity.home_score}</Text>
            )}
            <Text style={styles.teamLabel}>Home</Text>
          </View>

          <View style={styles.vsContainer}>
            {hasScore ? (
              <Text style={styles.finalText}>FINAL</Text>
            ) : (
              <Text style={styles.vsText}>VS</Text>
            )}
            {(opportunity.spread !== undefined || opportunity.current_spread !== undefined) && (
              <Text style={styles.spreadText}>
                Spread: {(() => {
                  const spread = opportunity.spread ?? opportunity.current_spread;
                  return spread > 0 ? `+${spread}` : spread;
                })()}
              </Text>
            )}
          </View>

          <View style={styles.teamContainer}>
            <Text style={styles.teamName}>{opportunity.away_team}</Text>
            {hasScore && (
              <Text style={styles.scoreText}>{opportunity.away_score}</Text>
            )}
            <Text style={styles.teamLabel}>Away</Text>
          </View>
        </View>

        {/* Bet Recommendation */}
        <View style={styles.recommendedContainer}>
          <View style={[
            styles.recommendedBadge,
            isCompleted && opportunity.bet_result === 'win' && styles.recommendedBadgeWin,
            isCompleted && opportunity.bet_result === 'loss' && styles.recommendedBadgeLoss,
          ]}>
            <Text style={styles.recommendedLabel}>{details.betOnLabel}</Text>
            <Text style={[
              styles.recommendedTeam,
              isCompleted && opportunity.bet_result === 'win' && styles.recommendedTeamWin,
              isCompleted && opportunity.bet_result === 'loss' && styles.recommendedTeamLoss,
            ]}>
              {betOn}
            </Text>
          </View>
        </View>

        {/* Strategy Metrics */}
        {details.metrics.length > 0 && (
          <View style={styles.metricsContainer}>
            {details.metrics.map((metric, index) => (
              <View key={index} style={styles.metricItem}>
                <Text style={styles.metricLabel}>{metric.label}</Text>
                <Text style={styles.metricValue}>{metric.value}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Streak info for hot_vs_cold */}
        {details.extraInfo?.hotStreak && details.extraInfo?.coldStreak && (
          <View style={styles.streakContainer}>
            <Text style={styles.streakText}>
              Hot Streak: {details.extraInfo.hotStreak} | Cold Streak: {details.extraInfo.coldStreak}
            </Text>
          </View>
        )}

        {/* Rationale if available */}
        {opportunity.rationale && (
          <View style={styles.rationaleContainer}>
            <Text style={styles.rationaleText}>{opportunity.rationale}</Text>
          </View>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  cardCompleted: {
    backgroundColor: '#fafafa',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sportBadge: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  sportText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  resultBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
  },
  resultText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  gameInfo: {
    width: '100%',
  },
  teamRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  teamContainer: {
    flex: 1,
    alignItems: 'center',
  },
  teamName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 4,
  },
  teamLabel: {
    fontSize: 12,
    color: '#666',
  },
  scoreText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 4,
  },
  vsContainer: {
    alignItems: 'center',
    marginHorizontal: 12,
  },
  vsText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#999',
    marginBottom: 4,
  },
  finalText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#666',
    marginBottom: 4,
  },
  spreadText: {
    fontSize: 12,
    color: '#666',
  },
  recommendedContainer: {
    marginTop: 12,
    marginBottom: 8,
  },
  recommendedBadge: {
    backgroundColor: '#E3F2FD',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  recommendedBadgeWin: {
    backgroundColor: '#E8F5E9',
    borderColor: '#4CAF50',
    borderWidth: 1,
  },
  recommendedBadgeLoss: {
    backgroundColor: '#FFEBEE',
    borderColor: '#F44336',
    borderWidth: 1,
  },
  recommendedLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  recommendedTeam: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  recommendedTeamWin: {
    color: '#4CAF50',
  },
  recommendedTeamLoss: {
    color: '#F44336',
  },
  metricsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  metricItem: {
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  streakContainer: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    alignItems: 'center',
  },
  streakText: {
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic',
  },
  rationaleContainer: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    alignItems: 'center',
  },
  rationaleText: {
    fontSize: 11,
    color: '#888',
    textAlign: 'center',
    fontStyle: 'italic',
  },
});

export default StrategyOpportunityCard;
