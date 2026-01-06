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

