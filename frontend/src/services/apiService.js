/**
 * API Service
 * Handles all API calls to the backend
 */

import axios from 'axios';
import { API_BASE_URL, SPORTS } from '../constants/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000, // 15 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Fetch predictions for a specific sport
 * @param {string} sport - Sport key (nfl, nba, ncaam)
 * @param {string|null} date - Optional date in YYYY-MM-DD format
 * @returns {Promise<Object>} Prediction data
 */
export const getPredictions = async (sport, date = null) => {
  try {
    // Check if API URL is configured
    if (API_BASE_URL.includes('YOUR-API-ID')) {
      throw new Error('API Gateway URL not configured. Please update src/constants/api.js with your API Gateway URL.');
    }
    
    const url = date 
      ? `/api/predictions/${sport}?date=${date}`
      : `/api/predictions/${sport}`;
    const response = await apiClient.get(url);
    return response.data;
  } catch (error) {
    console.error(`Error fetching predictions for ${sport}:`, error);
    
    // Provide helpful error messages
    if (error.message && error.message.includes('API Gateway URL')) {
      throw error;
    } else if (error.response) {
      // API returned an error
      throw new Error(`API Error: ${error.response.status} - ${error.response.statusText}`);
    } else if (error.request) {
      // Request was made but no response
      throw new Error('Network error: Could not reach API. Check your API Gateway URL and internet connection.');
    } else {
      throw new Error(error.message || 'Failed to fetch predictions');
    }
  }
};

/**
 * Fetch predictions for all sports
 * @param {string|null} date - Optional date in YYYY-MM-DD format
 * @returns {Promise<Object>} Object with predictions for each sport
 */
export const getAllPredictions = async (date = null) => {
  try {
    // Fetch all sports in parallel
    const promises = Object.values(SPORTS).map(sport => 
      getPredictions(sport, date).catch(err => {
        console.error(`Failed to fetch ${sport} predictions:`, err);
        return { sport, error: err.message, games: [] };
      })
    );
    
    const results = await Promise.all(promises);
    
    // Organize by sport
    const predictionsBySport = {};
    results.forEach(result => {
      if (result.sport) {
        predictionsBySport[result.sport] = result;
      }
    });
    
    return predictionsBySport;
  } catch (error) {
    console.error('Error fetching all predictions:', error);
    throw error;
  }
};

/**
 * Get all games from all sports predictions
 * Combines games from NFL, NBA, and NCAAM into a single array
 * @param {string|null} date - Optional date in YYYY-MM-DD format
 * @returns {Promise<Array>} Array of all games with sport info
 */
export const getAllGames = async (date = null) => {
  try {
    const predictionsBySport = await getAllPredictions(date);
    const allGames = [];
    
    // Flatten games from all sports and add sport identifier
    Object.keys(predictionsBySport).forEach(sport => {
      const prediction = predictionsBySport[sport];
      if (prediction.games && Array.isArray(prediction.games)) {
        prediction.games.forEach(game => {
          allGames.push({
            ...game,
            sport: sport,
            sport_name: prediction.sport_name || sport.toUpperCase(),
          });
        });
      }
    });
    
    return allGames;
  } catch (error) {
    console.error('Error fetching all games:', error);
    throw error;
  }
};

/**
 * Fetch elite team opportunities
 * Elite teams are top 25% by win% with good recent form (last 5 games)
 * @param {string|null} sport - Optional sport key (nfl, nba, ncaam). If null, returns all sports
 * @returns {Promise<Object>} Elite team opportunities data
 */
export const getEliteTeams = async (sport = null) => {
  try {
    // Check if API URL is configured
    if (API_BASE_URL.includes('YOUR-API-ID')) {
      throw new Error('API Gateway URL not configured. Please update src/constants/api.js with your API Gateway URL.');
    }

    const url = sport
      ? `/api/elite-teams/${sport}`
      : `/api/elite-teams`;
    const response = await apiClient.get(url);
    return response.data;
  } catch (error) {
    console.error('Error fetching elite teams:', error);

    // Provide helpful error messages
    if (error.message && error.message.includes('API Gateway URL')) {
      throw error;
    } else if (error.response) {
      throw new Error(`API Error: ${error.response.status} - ${error.response.statusText}`);
    } else if (error.request) {
      throw new Error('Network error: Could not reach API. Check your API Gateway URL and internet connection.');
    } else {
      throw new Error(error.message || 'Failed to fetch elite teams');
    }
  }
};

/**
 * Fetch Hot vs Cold opportunities from all sports
 * Hot team (60%+ last 5) vs Cold team (40%- last 5)
 * @param {string|null} sport - Optional sport filter
 * @returns {Promise<Object>} Hot vs Cold opportunities data
 */
export const getHotVsColdOpportunities = async (sport = null) => {
  try {
    const predictionsBySport = await getAllPredictions();
    const allOpportunities = [];
    let totalOpportunities = 0;
    let sportsWithOpportunities = 0;

    // Extract hot_vs_cold opportunities from each sport
    Object.keys(predictionsBySport).forEach(sportKey => {
      if (sport && sportKey !== sport) return; // Filter by sport if specified

      const prediction = predictionsBySport[sportKey];
      const hotVsCold = prediction?.strategies?.hot_vs_cold;

      if (hotVsCold?.opportunities?.length > 0) {
        sportsWithOpportunities++;
        hotVsCold.opportunities.forEach(opp => {
          allOpportunities.push({
            ...opp,
            sport: sportKey,
            sport_name: prediction.sport_name || sportKey.toUpperCase(),
          });
          totalOpportunities++;
        });
      }
    });

    return {
      opportunities: allOpportunities,
      summary: {
        total_opportunities: totalOpportunities,
        sports_with_opportunities: sportsWithOpportunities,
      },
    };
  } catch (error) {
    console.error('Error fetching hot vs cold opportunities:', error);
    throw error;
  }
};

/**
 * Fetch Opponent Perfect Form opportunities from all sports
 * Teams with 5/5 perfect form - bet on opponent (regression to mean)
 * @param {string|null} sport - Optional sport filter
 * @returns {Promise<Object>} Opponent Perfect Form opportunities data
 */
export const getOpponentPerfectFormOpportunities = async (sport = null) => {
  try {
    const predictionsBySport = await getAllPredictions();
    const allOpportunities = [];
    let totalOpportunities = 0;
    let sportsWithOpportunities = 0;

    // Extract opponent_perfect_form opportunities from each sport
    Object.keys(predictionsBySport).forEach(sportKey => {
      if (sport && sportKey !== sport) return; // Filter by sport if specified

      const prediction = predictionsBySport[sportKey];
      const perfectForm = prediction?.strategies?.opponent_perfect_form;

      if (perfectForm?.opportunities?.length > 0) {
        sportsWithOpportunities++;
        perfectForm.opportunities.forEach(opp => {
          allOpportunities.push({
            ...opp,
            sport: sportKey,
            sport_name: prediction.sport_name || sportKey.toUpperCase(),
          });
          totalOpportunities++;
        });
      }
    });

    return {
      opportunities: allOpportunities,
      summary: {
        total_opportunities: totalOpportunities,
        sports_with_opportunities: sportsWithOpportunities,
      },
    };
  } catch (error) {
    console.error('Error fetching opponent perfect form opportunities:', error);
    throw error;
  }
};

/**
 * Fetch strategy performance data for a sport
 * Returns cumulative win rates and chart data for all strategies
 * @param {string} sport - Sport key (nfl, nba, ncaam)
 * @returns {Promise<Object>} Strategy performance data with chart data
 */
export const getStrategyPerformance = async (sport) => {
  try {
    // Check if API URL is configured
    if (API_BASE_URL.includes('YOUR-API-ID')) {
      throw new Error('API Gateway URL not configured. Please update src/constants/api.js with your API Gateway URL.');
    }

    const url = `/api/strategy-performance/${sport}`;
    const response = await apiClient.get(url);
    return response.data;
  } catch (error) {
    console.error(`Error fetching strategy performance for ${sport}:`, error);

    // Provide helpful error messages
    if (error.message && error.message.includes('API Gateway URL')) {
      throw error;
    } else if (error.response) {
      throw new Error(`API Error: ${error.response.status} - ${error.response.statusText}`);
    } else if (error.request) {
      throw new Error('Network error: Could not reach API. Check your API Gateway URL and internet connection.');
    } else {
      throw new Error(error.message || 'Failed to fetch strategy performance');
    }
  }
};

