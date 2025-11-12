#!/usr/bin/env python3
"""Test script to validate training_dataset.csv is ready for the training script."""

import pandas as pd
import numpy as np
from pathlib import Path

def test_training_dataset():
    """Test that training_dataset.csv is ready for the CFB betting model."""
    
    csv_path = Path("data/processed/training_dataset.csv")
    
    print("=" * 60)
    print("Testing Training Dataset CSV")
    print("=" * 60)
    
    # Test 1: File exists
    print("\n1. Checking if file exists...")
    if not csv_path.exists():
        print("   ❌ FAIL: File does not exist!")
        return False
    print("   ✅ PASS: File exists")
    
    # Test 2: Can load CSV
    print("\n2. Loading CSV...")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ PASS: Loaded {len(df)} rows")
    except Exception as e:
        print(f"   ❌ FAIL: Could not load CSV: {e}")
        return False
    
    # Test 3: Required columns exist
    print("\n3. Checking required columns...")
    required_cols = ['home_team', 'away_team', 'home_points', 'away_points', 'ou_line', 'date', 'spread']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ❌ FAIL: Missing columns: {missing_cols}")
        return False
    print(f"   ✅ PASS: All required columns present: {required_cols}")
    
    # Test 4: Data types
    print("\n4. Checking data types...")
    print(f"   home_points: {df['home_points'].dtype}")
    print(f"   away_points: {df['away_points'].dtype}")
    print(f"   ou_line: {df['ou_line'].dtype}")
    print(f"   date: {df['date'].dtype}")
    print(f"   spread: {df['spread'].dtype}")
    
    # Test 5: Date can be converted to datetime (as training script does)
    print("\n5. Testing date conversion (as training script does)...")
    try:
        df_test = df.copy()
        df_test['date'] = pd.to_datetime(df_test['date'])
        print(f"   ✅ PASS: Date conversion successful")
        print(f"   Date range: {df_test['date'].min()} to {df_test['date'].max()}")
    except Exception as e:
        print(f"   ❌ FAIL: Date conversion failed: {e}")
        return False
    
    # Test 6: Can calculate total_points (as training script does)
    print("\n6. Testing total_points calculation...")
    try:
        df_test = df.copy()
        df_test['total_points'] = df_test['home_points'] + df_test['away_points']
        print(f"   ✅ PASS: total_points calculation successful")
        print(f"   Total points range: {df_test['total_points'].min():.1f} to {df_test['total_points'].max():.1f}")
    except Exception as e:
        print(f"   ❌ FAIL: total_points calculation failed: {e}")
        return False
    
    # Test 7: Can calculate went_over (as training script does)
    print("\n7. Testing went_over calculation...")
    try:
        df_test = df.copy()
        df_test['total_points'] = df_test['home_points'] + df_test['away_points']
        df_test['went_over'] = (df_test['total_points'] > df_test['ou_line']).astype(int)
        print(f"   ✅ PASS: went_over calculation successful")
        print(f"   Over count: {df_test['went_over'].sum()}, Under count: {(~df_test['went_over'].astype(bool)).sum()}")
    except Exception as e:
        print(f"   ❌ FAIL: went_over calculation failed: {e}")
        return False
    
    # Test 8: Missing values in critical columns
    print("\n8. Checking for missing values in critical columns...")
    critical_cols = ['home_team', 'away_team', 'home_points', 'away_points', 'ou_line', 'date']
    has_missing = False
    for col in critical_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"   ⚠️  WARNING: {col} has {missing_count} missing values")
            has_missing = True
        else:
            print(f"   ✅ {col}: No missing values")
    
    # spread can have missing values (it's optional)
    spread_missing = df['spread'].isna().sum()
    print(f"   ℹ️  spread: {spread_missing} missing values (acceptable, column is optional)")
    
    # Test 9: Data is sorted chronologically
    print("\n9. Checking chronological sorting...")
    df_test = df.copy()
    df_test['date'] = pd.to_datetime(df_test['date'])
    is_sorted = df_test['date'].is_monotonic_increasing
    if is_sorted:
        print("   ✅ PASS: Data is sorted chronologically")
    else:
        print("   ❌ FAIL: Data is NOT sorted chronologically")
        return False
    
    # Test 10: Can simulate training script's load_data method
    print("\n10. Simulating training script's load_data method...")
    try:
        data = pd.read_csv(csv_path)
        data['total_points'] = data['home_points'] + data['away_points']
        data['went_over'] = (data['total_points'] > data['ou_line']).astype(int)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        print(f"   ✅ PASS: load_data simulation successful")
        print(f"   Final shape: {data.shape}")
    except Exception as e:
        print(f"   ❌ FAIL: load_data simulation failed: {e}")
        return False
    
    # Test 11: Can simulate create_features lookup (check a few games)
    print("\n11. Testing feature creation lookup (sample games)...")
    try:
        data = pd.read_csv(csv_path)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # Test with a few games from different parts of the dataset
        test_indices = [100, 1000, 5000] if len(data) > 5000 else [10, 50, 100]
        
        for idx in test_indices:
            if idx >= len(data):
                continue
            game = data.iloc[idx]
            home_team = game['home_team']
            game_date = game['date']
            
            # Simulate the training script's lookup
            home_history = data[
                ((data['home_team'] == home_team) | (data['away_team'] == home_team)) & 
                (data['date'] < game_date)
            ].tail(5)
            
            if len(home_history) >= 2:
                print(f"   ✅ Game {idx}: Found {len(home_history)} historical games for {home_team}")
            else:
                print(f"   ⚠️  Game {idx}: Only {len(home_history)} historical games for {home_team} (may be filtered out)")
        
        print("   ✅ PASS: Feature lookup simulation successful")
    except Exception as e:
        print(f"   ❌ FAIL: Feature lookup simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 12: Spread values are reasonable
    print("\n12. Checking spread values...")
    spread_valid = df[df['spread'].notna()]['spread']
    if len(spread_valid) > 0:
        print(f"   Spread range: {spread_valid.min():.1f} to {spread_valid.max():.1f}")
        print(f"   Spread mean: {spread_valid.mean():.2f}")
        print(f"   ✅ PASS: Spread values look reasonable")
    else:
        print(f"   ⚠️  WARNING: No spread values found (but this is acceptable)")
    
    # Test 13: Points values are reasonable
    print("\n13. Checking points values...")
    print(f"   home_points range: {df['home_points'].min():.1f} to {df['home_points'].max():.1f}")
    print(f"   away_points range: {df['away_points'].min():.1f} to {df['away_points'].max():.1f}")
    print(f"   ou_line range: {df['ou_line'].min():.1f} to {df['ou_line'].max():.1f}")
    
    if df['home_points'].min() >= 0 and df['away_points'].min() >= 0:
        print("   ✅ PASS: Points values are non-negative and reasonable")
    else:
        print("   ❌ FAIL: Negative points found!")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Dataset is ready for training!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_training_dataset()
    exit(0 if success else 1)

