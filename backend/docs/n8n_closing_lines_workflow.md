# n8n Workflow: Closing Line Capture

Captures closing lines (spread, total, team totals) 30 minutes before each game tips off.
These populate the `closing_*` columns on `todays_games` for line movement analysis.

## Prerequisites

- `collect_todays_games` Lambda runs at 6:30 AM EST and populates `todays_games`
- Migration `add_line_movement_columns` has been run (adds `closing_*` columns)
- n8n has PostgreSQL credential and Odds API key configured

## Credentials Needed

### PostgreSQL — "Sports Betting RDS"
- Host: `sports-betting-analytics.cklmc4c0wc9i.us-east-1.rds.amazonaws.com`
- Database: `analytics`
- User: `sbadmin`
- Password: *(stored in AWS Secrets Manager)*
- Port: `5432`
- SSL: Enable

### Odds API
- Passed as query parameter `apiKey` in HTTP Request node
- Key stored in AWS Secrets Manager as `odds-api-key`

## Workflow Diagram

```
[Schedule Trigger: 7:00 AM EST]
  → [Postgres: Get today's games]
  → [Code: Calculate wait times + filter]
  → [Wait: Until commence_time - 30min]
  → [Postgres: Check still needs closing lines]
  → [IF: Needs update?]
  → [HTTP Request: Fetch odds from API]
  → [Code: Parse closing lines]
  → [Postgres: Write closing lines]
```

## Node-by-Node Configuration

### Node 1: Schedule Trigger

| Setting | Value |
|---------|-------|
| Trigger | Cron |
| Expression | At 7:00 AM, every day |
| Timezone | America/New_York |

**Why 7:00 AM:** The `collect_todays_games` Lambda runs at 6:30 AM and populates the
`todays_games` table. By 7:00 AM the data is ready.

---

### Node 2: Postgres — Get Today's Games

| Setting | Value |
|---------|-------|
| Credential | Sports Betting RDS |
| Operation | Execute Query |

**Query:**
```sql
SELECT id, sport, game_date,
       commence_time,
       home_team, away_team
FROM todays_games
WHERE game_date = CURRENT_DATE
  AND closing_captured_at IS NULL
ORDER BY commence_time;
```

---

### Node 3: Code — Calculate Wait Times

| Setting | Value |
|---------|-------|
| Mode | Run Once for All Items |

```javascript
const sportApiKeys = {
  'NFL': 'americanfootball_nfl',
  'NBA': 'basketball_nba',
  'NCAAM': 'basketball_ncaab'
};

const now = new Date();
const results = [];

for (const item of $input.all()) {
  const commence = new Date(item.json.commence_time);
  // Wait until 30 minutes before game time
  const waitUntil = new Date(commence.getTime() - 30 * 60 * 1000);

  // Skip games where the 30-min-before window has already passed
  if (waitUntil <= now) {
    continue;
  }

  results.push({
    json: {
      ...item.json,
      wait_until: waitUntil.toISOString(),
      api_sport_key: sportApiKeys[item.json.sport] || null
    }
  });
}

return results;
```

---

### Node 4: Wait

| Setting | Value |
|---------|-------|
| Resume | At Specific Date and Time |
| Date and Time | Expression: `{{ $json.wait_until }}` |

Each game item pauses independently until its `wait_until` time. n8n persists
these across restarts.

---

### Node 5: Postgres — Check Still Needs Update

| Setting | Value |
|---------|-------|
| Credential | Sports Betting RDS |
| Operation | Execute Query |

```sql
SELECT id
FROM todays_games
WHERE id = {{ $json.id }}
  AND closing_captured_at IS NULL;
```

Prevents double-processing if a sibling game's execution already wrote data.

---

### Node 6: IF — Has Results?

| Setting | Value |
|---------|-------|
| Condition | `{{ $json.id }}` is not empty |

True branch → HTTP Request. False branch → unconnected (stops).

---

### Node 7: HTTP Request — Fetch Odds

| Setting | Value |
|---------|-------|
| Method | GET |
| URL | `https://api.the-odds-api.com/v4/sports/{{ $('Code — Calculate Wait Times').item.json.api_sport_key }}/odds` |

**Query Parameters:**

| Name | Value |
|------|-------|
| apiKey | *(your Odds API key)* |
| regions | us |
| markets | spreads,totals,team_totals |
| oddsFormat | american |
| dateFormat | iso |

---

### Node 8: Code — Parse Closing Lines

| Setting | Value |
|---------|-------|
| Mode | Run Once for Each Item |

```javascript
const homeTeam = $('Code — Calculate Wait Times').item.json.home_team;
const awayTeam = $('Code — Calculate Wait Times').item.json.away_team;
const gameId = $('Code — Calculate Wait Times').item.json.id;

// API response is an array of games
const apiGames = $input.first().json.body || $input.first().json;
const gamesArray = Array.isArray(apiGames) ? apiGames : [];

// Find our game by home team
let matchedGame = null;
for (const game of gamesArray) {
  if (game.home_team === homeTeam) {
    matchedGame = game;
    break;
  }
}

if (!matchedGame) {
  return [{ json: { gameId, error: `Game not found: ${homeTeam}` } }];
}

// Extract closing lines — prefer DraftKings, fallback to any
let closingSpread = null;
let closingTotal = null;
let closingHomeTT = null;
let closingAwayTT = null;

let fallbackSpread = null;
let fallbackTotal = null;
let fallbackHomeTT = null;
let fallbackAwayTT = null;

const bookmakers = matchedGame.bookmakers || [];

for (const bookmaker of bookmakers) {
  const isDK = bookmaker.key?.toLowerCase().includes('draftkings');

  for (const market of bookmaker.markets || []) {
    // Spreads
    if (market.key === 'spreads') {
      for (const outcome of market.outcomes || []) {
        if (outcome.name === homeTeam) {
          if (isDK) closingSpread = outcome.point;
          else if (fallbackSpread === null) fallbackSpread = outcome.point;
          break;
        }
      }
    }

    // Totals
    if (market.key === 'totals') {
      for (const outcome of market.outcomes || []) {
        if (outcome.point != null) {
          if (isDK) closingTotal = outcome.point;
          else if (fallbackTotal === null) fallbackTotal = outcome.point;
          break;
        }
      }
    }

    // Team Totals
    if (market.key === 'team_totals') {
      for (const outcome of market.outcomes || []) {
        if (outcome.name?.toLowerCase() !== 'over' || outcome.point == null) continue;
        const desc = (outcome.description || '').toLowerCase();
        const isHome = homeTeam.toLowerCase().includes(desc) ||
                       desc.includes(homeTeam.toLowerCase());
        const isAway = awayTeam.toLowerCase().includes(desc) ||
                       desc.includes(awayTeam.toLowerCase());

        if (isHome) {
          if (isDK) closingHomeTT = outcome.point;
          else if (fallbackHomeTT === null) fallbackHomeTT = outcome.point;
        } else if (isAway) {
          if (isDK) closingAwayTT = outcome.point;
          else if (fallbackAwayTT === null) fallbackAwayTT = outcome.point;
        }
      }
    }
  }
}

// Use fallbacks if DraftKings not found
return [{
  json: {
    gameId,
    closingSpread: closingSpread ?? fallbackSpread,
    closingTotal: closingTotal ?? fallbackTotal,
    closingHomeTT: closingHomeTT ?? fallbackHomeTT,
    closingAwayTT: closingAwayTT ?? fallbackAwayTT
  }
}];
```

---

### Node 9: Postgres — Write Closing Lines

| Setting | Value |
|---------|-------|
| Credential | Sports Betting RDS |
| Operation | Execute Query |

```sql
UPDATE todays_games
SET closing_spread = {{ $json.closingSpread }},
    closing_total = {{ $json.closingTotal }},
    closing_home_tt = {{ $json.closingHomeTT }},
    closing_away_tt = {{ $json.closingAwayTT }},
    closing_captured_at = NOW()
WHERE id = {{ $json.gameId }}
  AND closing_captured_at IS NULL;
```

## Estimated Costs

| Resource | Daily Usage | Monthly |
|----------|------------|---------|
| n8n executions | 1 trigger + ~15-20 game items | ~500-600/month |
| Odds API calls | 1 per game (~15-20/day) | ~450-600 requests |

## Verification

After a game day, confirm closing lines are populated:
```sql
SELECT sport, home_team, spread, closing_spread,
       total, closing_total,
       home_team_total, closing_home_tt,
       closing_captured_at
FROM todays_games
WHERE game_date = CURRENT_DATE
  AND closing_captured_at IS NOT NULL;
```

## Future Optimization

To reduce Odds API calls, each game's Code node could also update sibling games
in the same sport from the same API response. This would reduce ~15-20 calls/day
to ~3-5 calls/day (one per sport per time cluster).
