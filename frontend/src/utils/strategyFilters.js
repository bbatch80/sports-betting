/**
 * Strategy Filtering Utilities
 * Filters games based on different betting strategies
 */

import { STRATEGIES } from '../constants/api';

/**
 * Filter games for Home Team Strategy
 * Shows games where home team's 10 point handicap spread coverage percentage 
 * exceeds the away team's 10 point handicap spread coverage percentage
 * @param {Array} games - Array of game objects
 * @returns {Array} Filtered games
 */
export const filterHomeTeamStrategy = (games) => {
  if (!Array.isArray(games)) return [];
  
  return games.filter(game => {
    const homeCoverage = game.home_cover_pct_handicap; // 10 point handicap coverage
    const awayCoverage = game.away_cover_pct_handicap; // 10 point handicap coverage
    
    // Both values must exist and be numbers
    if (typeof homeCoverage !== 'number' || typeof awayCoverage !== 'number') {
      return false;
    }
    
    // Home team's 10 point handicap coverage must exceed away team's
    return homeCoverage > awayCoverage;
  });
};

/**
 * Filter games for Away Team Strategy
 * Shows games where away team's 10 point handicap spread coverage percentage 
 * exceeds the home team's 10 point handicap spread coverage percentage
 * @param {Array} games - Array of game objects
 * @returns {Array} Filtered games
 */
export const filterAwayTeamStrategy = (games) => {
  if (!Array.isArray(games)) return [];
  
  return games.filter(game => {
    const homeCoverage = game.home_cover_pct_handicap; // 10 point handicap coverage
    const awayCoverage = game.away_cover_pct_handicap; // 10 point handicap coverage
    
    // Both values must exist and be numbers
    if (typeof homeCoverage !== 'number' || typeof awayCoverage !== 'number') {
      return false;
    }
    
    // Away team's 10 point handicap coverage must exceed home team's
    return awayCoverage > homeCoverage;
  });
};

/**
 * Filter games for Spread Coverage Strategy
 * Shows games where one team has 10% higher or more season-long spread coverage 
 * on the standard closing line than the other team
 * @param {Array} games - Array of game objects
 * @returns {Array} Filtered games
 */
export const filterSpreadCoverageStrategy = (games) => {
  if (!Array.isArray(games)) return [];
  
  return games.filter(game => {
    // Use regular coverage percentages (standard closing line), not handicap
    const homeCoverage = game.home_cover_pct; // Season-long spread coverage on standard closing line
    const awayCoverage = game.away_cover_pct; // Season-long spread coverage on standard closing line
    
    // Both values must exist and be numbers
    if (typeof homeCoverage !== 'number' || typeof awayCoverage !== 'number') {
      return false;
    }
    
    // Calculate absolute difference
    const coverageDifference = Math.abs(homeCoverage - awayCoverage);
    
    // Difference must be 10% or higher (>= 10%)
    return coverageDifference >= 10;
  });
};

/**
 * Filter games based on strategy type
 * @param {Array} games - Array of game objects
 * @param {string} strategy - Strategy type (from STRATEGIES constant)
 * @returns {Array} Filtered games
 */
export const filterGamesByStrategy = (games, strategy) => {
  switch (strategy) {
    case STRATEGIES.HOME_TEAM:
      return filterHomeTeamStrategy(games);
    case STRATEGIES.AWAY_TEAM:
      return filterAwayTeamStrategy(games);
    case STRATEGIES.SPREAD_COVERAGE:
      return filterSpreadCoverageStrategy(games);
    default:
      console.warn(`Unknown strategy: ${strategy}`);
      return [];
  }
};

/**
 * Format coverage percentage for display
 * @param {number} percentage - Coverage percentage
 * @returns {string} Formatted percentage string
 */
export const formatCoveragePercentage = (percentage) => {
  if (typeof percentage !== 'number') return 'N/A';
  return `${percentage.toFixed(1)}%`;
};

/**
 * Get the recommended team for a game based on coverage percentages
 * @param {Object} game - Game object
 * @returns {Object} { team: string, coverage: number, opponent: string }
 */
export const getRecommendedTeam = (game) => {
  const homeCoverage = game.home_cover_pct_handicap || 0;
  const awayCoverage = game.away_cover_pct_handicap || 0;
  
  if (homeCoverage > awayCoverage) {
    return {
      team: game.home_team,
      coverage: homeCoverage,
      opponent: game.away_team,
      isHome: true,
    };
  } else {
    return {
      team: game.away_team,
      coverage: awayCoverage,
      opponent: game.home_team,
      isHome: false,
    };
  }
};

