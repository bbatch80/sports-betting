"""
Sync Excel files from S3 to local directory

This script downloads Excel files from S3 bucket to local data/results directory.
Useful for keeping local copies synchronized with cloud storage after Lambda runs.

Usage:
    python scripts/sync_excel_from_s3.py
    python scripts/sync_excel_from_s3.py --sport nfl
    python scripts/sync_excel_from_s3.py --all
"""

import os
import sys
import boto3
import argparse
from botocore.exceptions import ClientError

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configuration
BUCKET_NAME = 'sports-betting-analytics-data'
SPORTS = ['nfl', 'nba', 'ncaam']
LOCAL_RESULTS_DIR = os.path.join(project_root, 'data', 'results')


def ensure_local_directory():
    """Ensure local results directory exists"""
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    print(f"Local directory: {LOCAL_RESULTS_DIR}")


def download_excel_from_s3(s3_client, sport_key: str) -> bool:
    """
    Download Excel file from S3 to local directory
    
    Returns True if successful, False otherwise
    """
    s3_key = f"data/results/{sport_key}_season_results.xlsx"
    local_file = os.path.join(LOCAL_RESULTS_DIR, f"{sport_key}_season_results.xlsx")
    
    try:
        print(f"\nDownloading {sport_key.upper()} Excel file...")
        print(f"  S3: s3://{BUCKET_NAME}/{s3_key}")
        print(f"  Local: {local_file}")
        
        # Download file
        s3_client.download_file(BUCKET_NAME, s3_key, local_file)
        
        # Check file size
        file_size = os.path.getsize(local_file)
        print(f"  ✓ Downloaded successfully ({file_size:,} bytes)")
        
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            print(f"  ⚠ File not found in S3: {s3_key}")
        else:
            print(f"  ✗ Error downloading: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_parquet_from_s3(s3_client, sport_key: str) -> bool:
    """
    Download Parquet file from S3 to local directory (optional)
    
    Returns True if successful, False otherwise
    """
    s3_key = f"data/results/{sport_key}_season_results.parquet"
    local_file = os.path.join(LOCAL_RESULTS_DIR, f"{sport_key}_season_results.parquet")
    
    try:
        s3_client.download_file(BUCKET_NAME, s3_key, local_file)
        file_size = os.path.getsize(local_file)
        print(f"  ✓ Downloaded Parquet ({file_size:,} bytes)")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            return False  # Parquet is optional
        else:
            print(f"  ⚠ Error downloading Parquet: {e}")
            return False
    except Exception as e:
        return False  # Parquet is optional, don't fail on error


def sync_all_sports():
    """Sync all sports Excel files from S3"""
    print("="*80)
    print("Syncing Excel files from S3 to local directory")
    print("="*80)
    
    ensure_local_directory()
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        print(f"\nConnected to S3 bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"✗ Error connecting to S3: {e}")
        return
    
    results = {}
    
    for sport in SPORTS:
        excel_success = download_excel_from_s3(s3_client, sport)
        parquet_success = download_parquet_from_s3(s3_client, sport)
        results[sport] = {
            'excel': excel_success,
            'parquet': parquet_success
        }
    
    # Print summary
    print("\n" + "="*80)
    print("SYNC SUMMARY")
    print("="*80)
    
    for sport in SPORTS:
        status = "✓" if results[sport]['excel'] else "✗"
        print(f"{status} {sport.upper()}: Excel {'downloaded' if results[sport]['excel'] else 'failed'}")
    
    successful = sum(1 for r in results.values() if r['excel'])
    print(f"\n✓ Successfully synced {successful}/{len(SPORTS)} sports")


def sync_single_sport(sport_key: str):
    """Sync Excel file for a single sport"""
    if sport_key not in SPORTS:
        print(f"✗ Invalid sport: {sport_key}")
        print(f"  Valid sports: {', '.join(SPORTS)}")
        return
    
    print(f"Syncing {sport_key.upper()} Excel file from S3...")
    
    ensure_local_directory()
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        excel_success = download_excel_from_s3(s3_client, sport_key)
        download_parquet_from_s3(s3_client, sport_key)  # Try parquet too
        
        if excel_success:
            print(f"\n✓ Successfully synced {sport_key.upper()}")
        else:
            print(f"\n✗ Failed to sync {sport_key.upper()}")
            
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Sync Excel files from S3 to local directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync all sports
  python scripts/sync_excel_from_s3.py --all
  
  # Sync a specific sport
  python scripts/sync_excel_from_s3.py --sport nfl
  
  # Sync all (default)
  python scripts/sync_excel_from_s3.py
        """
    )
    parser.add_argument(
        '--sport',
        choices=SPORTS,
        help='Sport to sync (nfl, nba, ncaam)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Sync all sports (default if no --sport specified)'
    )
    
    args = parser.parse_args()
    
    if args.sport:
        sync_single_sport(args.sport)
    else:
        sync_all_sports()


if __name__ == "__main__":
    main()


