"""
Odds API Client
Simple client for interacting with The Odds API
"""

import os
import requests
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class OddsAPIError(Exception):
    """Custom exception for Odds API errors"""
    pass


class OddsAPIClient:
    """Client for The Odds API"""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set ODDS_API_KEY environment variable.")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """Make API request"""
        if params is None:
            params = {}
        
        params['apiKey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {response.status_code}"
            try:
                error_detail = response.json()
                if isinstance(error_detail, dict) and 'message' in error_detail:
                    error_msg += f" - {error_detail['message']}"
            except:
                error_msg += f" - {response.text[:100]}"
            raise OddsAPIError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise OddsAPIError(f"Request error: {str(e)}") from e
    
    def get_scores(self, sport: str, days_from: int = 3, date_format: str = "iso") -> List[Dict[str, Any]]:
        """Get scores for completed games"""
        params = {
            "dateFormat": date_format,
            "daysFrom": str(days_from)
        }
        endpoint = f"sports/{sport}/scores"
        return self._make_request(endpoint, params)
    
    def get_historical_events(self, sport: str, date: str, date_format: str = "iso") -> Dict[str, Any]:
        """Get historical events for a specific date"""
        params = {
            "dateFormat": date_format,
            "date": date
        }
        endpoint = f"historical/sports/{sport}/events"
        return self._make_request(endpoint, params)
    
    def get_historical_event_odds(self, sport: str, event_id: str, date: str,
                                   markets: List[str] = None, regions: List[str] = None,
                                   odds_format: str = "american", date_format: str = "iso") -> Dict[str, Any]:
        """Get historical odds for a specific event at a specific timestamp"""
        params = {
            "date": date,
            "oddsFormat": odds_format,
            "dateFormat": date_format
        }
        
        if regions:
            params["regions"] = ",".join(regions)
        else:
            params["regions"] = "us"
        
        if markets:
            params["markets"] = ",".join(markets)
        else:
            params["markets"] = "h2h"
        
        endpoint = f"historical/sports/{sport}/events/{event_id}/odds"
        return self._make_request(endpoint, params)
    
    def get_usage(self) -> Dict[str, Any]:
        """Get API usage information"""
        # Usage info is in response headers, but we'll return a simple dict
        # You can check headers from any request: x-requests-used, x-requests-remaining
        return {
            "requests_used": 0,  # Would need to track from headers
            "requests_remaining": 0  # Would need to track from headers
        }
