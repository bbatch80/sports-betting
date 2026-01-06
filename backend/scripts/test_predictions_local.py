"""
Test the generate_predictions Lambda function locally
This script imports the Lambda function code and runs it locally for testing

Usage:
    python test_predictions_local.py --sport nba
    python test_predictions_local.py --sport all
    python test_predictions_local.py --sport nba --no-save  # Test without saving to S3
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime, timezone

# Add Lambda function directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda_functions', 'generate_predictions'))

# Import the Lambda function
from lambda_function import (
    get_api_key,
    generate_predictions_for_sport,
    write_json_to_s3,
    SPORTS_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_predictions_for_sport(sport_key: str, save_to_s3: bool = True):
    """
    Test prediction generation for a single sport
    
    Args:
        sport_key: Sport to test (nfl, nba, ncaam)
        save_to_s3: Whether to save results to S3 (default: True)
    """
    logger.info("="*80)
    logger.info(f"Testing prediction generation for {sport_key.upper()}")
    logger.info("="*80)
    
    try:
        # Get API key from Secrets Manager (same as Lambda)
        api_key = get_api_key()
        logger.info("✓ Retrieved API key from Secrets Manager")
        
        # Generate predictions
        logger.info(f"\nGenerating predictions for {sport_key.upper()}...")
        predictions = generate_predictions_for_sport(api_key, sport_key)
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("PREDICTION RESULTS")
        logger.info("="*80)
        logger.info(f"Sport: {predictions.get('sport_name', sport_key.upper())}")
        logger.info(f"Handicap Points: {predictions.get('handicap_points', 'N/A')}")
        logger.info(f"Prediction Date: {predictions.get('prediction_date', 'N/A')}")
        logger.info(f"Generated At: {predictions.get('generated_at', 'N/A')}")
        
        summary = predictions.get('summary', {})
        logger.info(f"\nSummary:")
        logger.info(f"  Total Games: {summary.get('total_games', 0)}")
        logger.info(f"  Games with Sufficient Data: {summary.get('games_with_sufficient_data', 0)}")
        logger.info(f"  Opportunities Found: {summary.get('opportunities', 0)}")
        
        if summary.get('opportunities', 0) > 0:
            logger.info(f"  Average Difference: {summary.get('average_difference', 0):.2f}%")
            logger.info(f"  Largest Difference: {summary.get('largest_difference', 0):.2f}%")
            logger.info(f"  Smallest Difference: {summary.get('smallest_difference', 0):.2f}%")
        
        # Display opportunities
        opportunities = predictions.get('opportunities', [])
        focus_team = predictions.get('focus_team', 'away')
        if len(opportunities) > 0:
            focus_label = 'Home Team' if focus_team == 'home' else 'Away Team'
            logger.info(f"\n{'='*80}")
            logger.info(f"OPPORTUNITIES ({focus_label} Focus)")
            logger.info("="*80)
            logger.info(f"{'Away Team':<25} {'Home Team':<25} {'Away %':<10} {'Home %':<10} {'Diff %':<10} {'Spread':<10}")
            logger.info("-"*80)
            
            for opp in opportunities[:10]:  # Show top 10
                away_team = opp.get('away_team', 'N/A')[:24]
                home_team = opp.get('home_team', 'N/A')[:24]
                away_pct = opp.get('away_cover_pct_handicap', 0)
                home_pct = opp.get('home_cover_pct_handicap', 0)
                diff = opp.get('handicap_pct_difference', 0)
                spread = opp.get('current_spread', 'N/A')
                
                logger.info(f"{away_team:<25} {home_team:<25} {away_pct:<10.1f} {home_pct:<10.1f} {diff:<10.1f} {spread:<10}")
            
            if len(opportunities) > 10:
                logger.info(f"\n... and {len(opportunities) - 10} more opportunities")
        else:
            logger.info("\nNo opportunities found for today's games")
        
        # Save to S3 if requested
        if save_to_s3:
            prediction_date = predictions.get('prediction_date')
            write_json_to_s3(predictions, sport_key, prediction_date)
            logger.info(f"\n✓ Saved predictions to S3")
        else:
            logger.info(f"\n⚠ Skipped saving to S3 (--no-save flag)")
        
        # Save local copy for inspection
        local_file = f"test_predictions_{sport_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(local_file, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        logger.info(f"✓ Saved local copy to: {local_file}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"✗ Error testing {sport_key.upper()}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Test the generate_predictions Lambda function locally',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test NBA predictions
  python test_predictions_local.py --sport nba
  
  # Test all sports
  python test_predictions_local.py --sport all
  
  # Test without saving to S3
  python test_predictions_local.py --sport nba --no-save
        """
    )
    parser.add_argument(
        '--sport',
        required=True,
        choices=['nfl', 'nba', 'ncaam', 'all'],
        help='Sport to test (or "all" for all sports)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving results to S3 (for testing only)'
    )
    
    args = parser.parse_args()
    
    # Determine which sports to test
    if args.sport == 'all':
        sports_to_test = list(SPORTS_CONFIG.keys())
    else:
        sports_to_test = [args.sport]
    
    # Test each sport
    results = {}
    for sport_key in sports_to_test:
        predictions = test_predictions_for_sport(sport_key, save_to_s3=not args.no_save)
        results[sport_key] = predictions
        
        if sport_key != sports_to_test[-1]:
            logger.info("\n" + "="*80 + "\n")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    for sport_key, predictions in results.items():
        if predictions:
            summary = predictions.get('summary', {})
            logger.info(f"{sport_key.upper()}: {summary.get('opportunities', 0)} opportunities found")
        else:
            logger.info(f"{sport_key.upper()}: FAILED")
    
    logger.info("\n✓ Local testing complete!")


if __name__ == "__main__":
    main()

