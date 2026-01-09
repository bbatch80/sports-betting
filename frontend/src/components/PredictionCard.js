/**
 * PredictionCard Component
 * Displays a single game prediction with team info and coverage percentages
 * Shows scores and bet results for completed games
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { formatCoveragePercentage, getRecommendedTeam } from '../utils/strategyFilters';
import { SPORT_NAMES } from '../constants/api';

const PredictionCard = ({ game }) => {
  if (!game) return null;

  const recommended = getRecommendedTeam(game);
  const sportName = SPORT_NAMES[game.sport] || game.sport_name || '';
  const isCompleted = game.game_completed === true;
  const hasScore = game.home_score !== undefined && game.away_score !== undefined;

  // Get result badge style based on bet result
  const getResultStyle = () => {
    if (!game.bet_result) return {};
    switch (game.bet_result.toLowerCase()) {
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
    if (!game.bet_result) return '';
    return game.bet_result.toUpperCase();
  };

  return (
    <View style={[styles.card, isCompleted && styles.cardCompleted]}>
      {/* Header with Sport Badge and Result Badge */}
      <View style={styles.headerRow}>
        <View style={styles.sportBadge}>
          <Text style={styles.sportText}>{sportName}</Text>
        </View>
        {isCompleted && game.bet_result && (
          <View style={[styles.resultBadge, getResultStyle()]}>
            <Text style={styles.resultText}>{getResultText()}</Text>
          </View>
        )}
      </View>

      {/* Game Info */}
      <View style={styles.gameInfo}>
        <View style={styles.teamRow}>
          <View style={styles.teamContainer}>
            <Text style={styles.teamName}>{game.home_team}</Text>
            {hasScore && (
              <Text style={styles.scoreText}>{game.home_score}</Text>
            )}
            <Text style={styles.teamLabel}>Home</Text>
            <Text style={styles.coverageText}>
              {formatCoveragePercentage(game.home_cover_pct_handicap)}
            </Text>
          </View>

          <View style={styles.vsContainer}>
            {hasScore ? (
              <Text style={styles.finalText}>FINAL</Text>
            ) : (
              <Text style={styles.vsText}>VS</Text>
            )}
            <Text style={styles.spreadText}>
              Spread: {game.current_spread > 0 ? '+' : ''}{game.current_spread}
            </Text>
          </View>

          <View style={styles.teamContainer}>
            <Text style={styles.teamName}>{game.away_team}</Text>
            {hasScore && (
              <Text style={styles.scoreText}>{game.away_score}</Text>
            )}
            <Text style={styles.teamLabel}>Away</Text>
            <Text style={styles.coverageText}>
              {formatCoveragePercentage(game.away_cover_pct_handicap)}
            </Text>
          </View>
        </View>

        {/* Recommended Team with Result */}
        <View style={styles.recommendedContainer}>
          <View style={[
            styles.recommendedBadge,
            isCompleted && game.bet_result === 'win' && styles.recommendedBadgeWin,
            isCompleted && game.bet_result === 'loss' && styles.recommendedBadgeLoss,
          ]}>
            <Text style={styles.recommendedLabel}>
              {isCompleted ? 'Recommendation:' : 'Recommended:'}
            </Text>
            <Text style={[
              styles.recommendedTeam,
              isCompleted && game.bet_result === 'win' && styles.recommendedTeamWin,
              isCompleted && game.bet_result === 'loss' && styles.recommendedTeamLoss,
            ]}>
              {recommended.team}
            </Text>
            <Text style={styles.recommendedCoverage}>
              {formatCoveragePercentage(recommended.coverage)} coverage
            </Text>
          </View>
        </View>

        {/* Coverage Difference */}
        <View style={styles.differenceContainer}>
          <Text style={styles.differenceText}>
            Coverage Difference: {formatCoveragePercentage(
              Math.abs((game.home_cover_pct_handicap || 0) - (game.away_cover_pct_handicap || 0))
            )}
          </Text>
        </View>
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
  scoreText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 4,
  },
  finalText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#666',
    marginBottom: 4,
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
    marginBottom: 4,
  },
  coverageText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#4CAF50',
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
  recommendedLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  recommendedTeam: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2196F3',
    marginBottom: 4,
  },
  recommendedCoverage: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: '600',
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
  recommendedTeamWin: {
    color: '#4CAF50',
  },
  recommendedTeamLoss: {
    color: '#F44336',
  },
  differenceContainer: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  differenceText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
});

export default PredictionCard;

