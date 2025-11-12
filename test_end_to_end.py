#!/usr/bin/env python3
"""End-to-end test: Simulate the actual training script workflow."""

import pandas as pd
import numpy as np

def test_end_to_end():
    """Test that the CSV works exactly as the training script expects."""
    
    print("=" * 60)
    print("End-to-End Test: Training Script Workflow")
    print("=" * 60)
    
    # Step 1: Load data (exactly as training script does)
    print("\n1. Loading data (simulating load_data method)...")
    try:
        data = pd.read_csv('data/processed/training_dataset.csv')
        data['total_points'] = data['home_points'] + data['away_points']
        data['went_over'] = (data['total_points'] > data['ou_line']).astype(int)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        print(f"   ✅ Loaded {len(data)} games")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Step 2: Test create_features logic (sample a few games)
    print("\n2. Testing create_features logic (sampling games)...")
    feature_list = []
    test_games = data.iloc[500:550]  # Sample games from middle of dataset
    
    for idx, game in test_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        game_date = game['date']
        
        # Exactly as training script does
        home_history = data[
            ((data['home_team'] == home_team) | (data['away_team'] == home_team)) & 
            (data['date'] < game_date)
        ].tail(5)
        
        away_history = data[
            ((data['home_team'] == away_team) | (data['away_team'] == away_team)) & 
            (data['date'] < game_date)
        ].tail(5)
        
        if len(home_history) < 2 or len(away_history) < 2:
            continue  # Skip games with insufficient history (as training script does)
        
        # Calculate features exactly as training script does
        hp_for, hp_against, h_totals = [], [], []
        for _, hg in home_history.iterrows():
            pf, pa = (hg['home_points'], hg['away_points']) if hg['home_team'] == home_team else (hg['away_points'], hg['home_points'])
            hp_for.append(pf)
            hp_against.append(pa)
            h_totals.append(hg['total_points'])
        
        ap_for, ap_against, a_totals = [], [], []
        for _, ag in away_history.iterrows():
            pf, pa = (ag['home_points'], ag['away_points']) if ag['home_team'] == away_team else (ag['away_points'], ag['home_points'])
            ap_for.append(pf)
            ap_against.append(pa)
            a_totals.append(ag['total_points'])
        
        f_dict = {
            'ou_line': game['ou_line'],
            'spread': float(game.get('spread', 0.0)),
            'home_avg_pts': np.mean(hp_for),
            'home_avg_pts_allowed': np.mean(hp_against),
            'home_avg_total': np.mean(h_totals),
            'home_std_total': np.std(h_totals),
            'away_avg_pts': np.mean(ap_for),
            'away_avg_pts_allowed': np.mean(ap_against),
            'away_avg_total': np.mean(a_totals),
            'away_std_total': np.std(a_totals),
            'combined_avg_total': (np.mean(h_totals) + np.mean(a_totals)) / 2,
            'went_over': game['went_over']
        }
        
        feature_list.append(f_dict)
    
    if len(feature_list) == 0:
        print("   ❌ FAILED: No features created")
        return False
    
    features_df = pd.DataFrame(feature_list)
    features_df['pace_factor'] = features_df['combined_avg_total'] - features_df['ou_line']
    
    print(f"   ✅ Created {len(feature_list)} feature rows from {len(test_games)} test games")
    print(f"   Feature columns: {list(features_df.columns)}")
    
    # Step 3: Verify feature values are reasonable
    print("\n3. Verifying feature values...")
    print(f"   ou_line range: {features_df['ou_line'].min():.1f} to {features_df['ou_line'].max():.1f}")
    print(f"   home_avg_pts range: {features_df['home_avg_pts'].min():.1f} to {features_df['home_avg_pts'].max():.1f}")
    print(f"   away_avg_pts range: {features_df['away_avg_pts'].min():.1f} to {features_df['away_avg_pts'].max():.1f}")
    print(f"   pace_factor range: {features_df['pace_factor'].min():.2f} to {features_df['pace_factor'].max():.2f}")
    
    # Check for NaN or inf values
    if features_df.isna().any().any():
        print("   ⚠️  WARNING: Some features contain NaN values")
        print(f"   NaN columns: {features_df.columns[features_df.isna().any()].tolist()}")
    else:
        print("   ✅ No NaN values in features")
    
    if np.isinf(features_df.select_dtypes(include=[np.number])).any().any():
        print("   ⚠️  WARNING: Some features contain infinite values")
    else:
        print("   ✅ No infinite values in features")
    
    # Step 4: Test that we can prepare X and y (as training script does)
    print("\n4. Testing train/test split preparation...")
    try:
        X = features_df.drop(columns=['went_over'])
        y = features_df['went_over']
        
        print(f"   ✅ X shape: {X.shape}")
        print(f"   ✅ y shape: {y.shape}")
        print(f"   ✅ y distribution: {y.value_counts().to_dict()}")
        
        # Check that all required columns are present
        expected_features = [
            'ou_line', 'spread', 'home_avg_pts', 'home_avg_pts_allowed',
            'home_avg_total', 'home_std_total', 'away_avg_pts', 'away_avg_pts_allowed',
            'away_avg_total', 'away_std_total', 'combined_avg_total', 'pace_factor'
        ]
        missing = [col for col in expected_features if col not in X.columns]
        if missing:
            print(f"   ❌ FAILED: Missing feature columns: {missing}")
            return False
        print(f"   ✅ All expected feature columns present")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ END-TO-END TEST PASSED!")
    print("   The CSV is ready to use with your training script.")
    print("   You can use: DB = 'data/processed/training_dataset.csv'")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_end_to_end()
    exit(0 if success else 1)

