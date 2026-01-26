-- Migration: Add todays_recommendations table
-- Purpose: Pre-computed daily recommendations for fast dashboard loads
-- Run: psql $DATABASE_URL -f migrations/001_add_todays_recommendations.sql

CREATE TABLE IF NOT EXISTS todays_recommendations (
    id SERIAL PRIMARY KEY,

    -- Game identification
    sport VARCHAR(50) NOT NULL,
    game_date DATE NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    game_time VARCHAR(50),

    -- Spread info
    spread FLOAT,
    spread_source VARCHAR(100),

    -- Team tiers and ratings
    home_tier VARCHAR(20),
    away_tier VARCHAR(20),
    home_ats_rating FLOAT,
    away_ats_rating FLOAT,

    -- Streak info
    home_streak_length INTEGER,
    home_streak_type VARCHAR(10),
    away_streak_length INTEGER,
    away_streak_type VARCHAR(10),

    -- Recommendations stored as JSON array
    recommendations_json TEXT,

    -- Timestamp
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure one record per game per day
    UNIQUE (sport, game_date, home_team, away_team)
);

-- Index for fast lookups by sport and date
CREATE INDEX IF NOT EXISTS idx_todays_rec_sport_date
    ON todays_recommendations(sport, game_date);

-- Confirm creation
SELECT 'todays_recommendations table created successfully' AS status;
