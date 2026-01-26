"""Analysis module for sports betting metrics."""

from .metrics import (
    ats_cover_rate,
    ats_record,
    spread_margin_avg,
    time_series_ats,
    handicap_cover_rate,
    ou_cover_rate,
    ou_record,
    team_ou_cover_rate,
    team_ou_record,
)

from .aggregations import (
    macro_ou_summary,
    macro_ou_time_series,
    macro_ou_by_total_bucket,
    micro_ou_all_teams,
)

from .insights import (
    detect_ou_streak,
    get_current_ou_streaks,
    get_cached_ou_streaks,
)

__all__ = [
    # ATS metrics
    'ats_cover_rate',
    'ats_record',
    'spread_margin_avg',
    'time_series_ats',
    'handicap_cover_rate',
    # O/U metrics
    'ou_cover_rate',
    'ou_record',
    'team_ou_cover_rate',
    'team_ou_record',
    # O/U aggregations
    'macro_ou_summary',
    'macro_ou_time_series',
    'macro_ou_by_total_bucket',
    'micro_ou_all_teams',
    # O/U insights
    'detect_ou_streak',
    'get_current_ou_streaks',
    'get_cached_ou_streaks',
]
