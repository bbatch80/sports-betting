"""
Local score storage
Stores scores in JSON format
"""

import os
import json
from typing import Dict, Optional, Any
from datetime import datetime


def get_scores_file_path(sport_key: str) -> str:
    """Get path to scores storage file"""
    # Get project root (parent of src directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"{sport_key}_scores.json")


def load_stored_scores(sport_key: str) -> Dict[str, Dict[str, Any]]:
    """Load stored scores from JSON file"""
    file_path = get_scores_file_path(sport_key)
    
    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_score(sport_key: str, event_id: str, home_score: int, away_score: int,
               home_team: str, away_team: str, game_date: str) -> None:
    """Save a score to local storage"""
    scores = load_stored_scores(sport_key)
    
    scores[event_id] = {
        "home_score": home_score,
        "away_score": away_score,
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date,
        "stored_at": datetime.now().isoformat()
    }
    
    file_path = get_scores_file_path(sport_key)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(scores, f, indent=2)
    except IOError:
        pass
