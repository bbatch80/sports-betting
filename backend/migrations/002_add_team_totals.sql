-- Migration: Add team totals columns to games table
-- Date: 2026-01-28
-- Purpose: Store individual team O/U lines (e.g., "Lakers O/U 115.5") separate from game totals

-- Add columns for team-specific totals
-- These are the betting lines for each team's individual score (not combined game total)

-- Home team total line (e.g., 115.5)
ALTER TABLE games ADD COLUMN IF NOT EXISTS home_team_total FLOAT;

-- Away team total line (e.g., 108.5)
ALTER TABLE games ADD COLUMN IF NOT EXISTS away_team_total FLOAT;

-- Home team total result: home_score - home_team_total
-- Positive = OVER, Negative = UNDER
ALTER TABLE games ADD COLUMN IF NOT EXISTS home_team_total_result FLOAT;

-- Away team total result: away_score - away_team_total
-- Positive = OVER, Negative = UNDER
ALTER TABLE games ADD COLUMN IF NOT EXISTS away_team_total_result FLOAT;

-- Example:
-- If Lakers are home with team_total=115.5 and score 120:
--   home_team_total = 115.5
--   home_team_total_result = 120 - 115.5 = 4.5 (OVER)
