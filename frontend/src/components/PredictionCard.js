/**
 * PredictionCard Component
 * Displays a single game prediction with team info and coverage percentages
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { formatCoveragePercentage, getRecommendedTeam } from '../utils/strategyFilters';
import { SPORT_NAMES } from '../constants/api';

const PredictionCard = ({ game }) => {
  if (!game) return null;

  const recommended = getRecommendedTeam(game);
  const sportName = SPORT_NAMES[game.sport] || game.sport_name || '';

  return (
    <View style={styles.card}>
      {/* Sport Badge */}
      <View style={styles.sportBadge}>
        <Text style={styles.sportText}>{sportName}</Text>
      </View>

      {/* Game Info */}
      <View style={styles.gameInfo}>
        <View style={styles.teamRow}>
          <View style={styles.teamContainer}>
            <Text style={styles.teamName}>{game.home_team}</Text>
            <Text style={styles.teamLabel}>Home</Text>
            <Text style={styles.coverageText}>
              {formatCoveragePercentage(game.home_cover_pct_handicap)}
            </Text>
          </View>

          <View style={styles.vsContainer}>
            <Text style={styles.vsText}>VS</Text>
            <Text style={styles.spreadText}>
              Spread: {game.current_spread > 0 ? '+' : ''}{game.current_spread}
            </Text>
          </View>

          <View style={styles.teamContainer}>
            <Text style={styles.teamName}>{game.away_team}</Text>
            <Text style={styles.teamLabel}>Away</Text>
            <Text style={styles.coverageText}>
              {formatCoveragePercentage(game.away_cover_pct_handicap)}
            </Text>
          </View>
        </View>

        {/* Recommended Team */}
        <View style={styles.recommendedContainer}>
          <View style={styles.recommendedBadge}>
            <Text style={styles.recommendedLabel}>Recommended:</Text>
            <Text style={styles.recommendedTeam}>{recommended.team}</Text>
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
  sportBadge: {
    backgroundColor: '#2196F3',
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginBottom: 12,
  },
  sportText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
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

