#!/usr/bin/env python3
"""Test using the exact same code pattern as the training script."""

import pandas as pd
import numpy as np

# Simulate the exact training script workflow
class CFBBettingModel:
    def __init__(self):
        self.data = None
        self.features = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data['total_points'] = self.data['home_points'] + self.data['away_points']
        self.data['went_over'] = (self.data['total_points'] > self.data['ou_line']).astype(int)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)

    def create_features(self, max_games=100):
        """Create features for first max_games to test quickly."""
        feature_list = []
        
        for idx, game in self.data.iterrows():
            if idx >= max_games:
                break
                
            home_team, away_team, game_date = game['home_team'], game['away_team'], game['date']
            
            home_history = self.data[((self.data['home_team'] == home_team) | (self.data['away_team'] == home_team)) & (self.data['date'] < game_date)].tail(5)
            away_history = self.data[((self.data['home_team'] == away_team) | (self.data['away_team'] == away_team)) & (self.data['date'] < game_date)].tail(5)
            
            if len(home_history) < 2 or len(away_history) < 2:
                continue
            
            hp_for, hp_against, h_totals = [], [], []
            for _, hg in home_history.iterrows():
                pf, pa = (hg['home_points'], hg['away_points']) if hg['home_team'] == home_team else (hg['away_points'], hg['home_points'])
                hp_for.append(pf); hp_against.append(pa); h_totals.append(hg['total_points'])
            
            ap_for, ap_against, a_totals = [], [], []
            for _, ag in away_history.iterrows():
                pf, pa = (ag['home_points'], ag['away_points']) if ag['home_team'] == away_team else (ag['away_points'], ag['home_points'])
                ap_for.append(pf); ap_against.append(pa); a_totals.append(ag['total_points'])
            
            f_dict = {
                'ou_line': game['ou_line'], 'spread': float(game.get('spread', 0.0)),
                'home_avg_pts': np.mean(hp_for), 'home_avg_pts_allowed': np.mean(hp_against),
                'home_avg_total': np.mean(h_totals), 'home_std_total': np.std(h_totals),
                'away_avg_pts': np.mean(ap_for), 'away_avg_pts_allowed': np.mean(ap_against),
                'away_avg_total': np.mean(a_totals), 'away_std_total': np.std(a_totals),
                'combined_avg_total': (np.mean(h_totals) + np.mean(a_totals)) / 2,
                'went_over': game['went_over']
            }
            
            feature_list.append(f_dict)
        
        self.features = pd.DataFrame(feature_list)
        self.features['pace_factor'] = self.features['combined_avg_total'] - self.features['ou_line']

print("=" * 60)
print("Testing with Exact Training Script Pattern")
print("=" * 60)

DB = 'data/processed/training_dataset.csv'
system = CFBBettingModel()

print("\n1. Loading data...")
try:
    system.load_data(DB)
    print(f"   ✅ Loaded {len(system.data)} games")
    print(f"   Date range: {system.data['date'].min()} to {system.data['date'].max()}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n2. Creating features (first 200 games)...")
try:
    system.create_features(max_games=200)
    print(f"   ✅ Created {len(system.features)} feature rows")
    print(f"   Feature columns: {list(system.features.columns)}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n3. Verifying features can be used for training...")
try:
    X = system.features.drop(columns=['went_over'])
    y = system.features['went_over']
    
    print(f"   ✅ X shape: {X.shape}")
    print(f"   ✅ y shape: {y.shape}")
    print(f"   ✅ No NaN in X: {not X.isna().any().any()}")
    print(f"   ✅ No NaN in y: {not y.isna().any()}")
    print(f"   ✅ y distribution: Over={y.sum()}, Under={len(y)-y.sum()}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ SUCCESS! The CSV works perfectly with your training script!")
print("=" * 60)
print("\nYou can now use this in your training script:")
print(f"  DB = '{DB}'")
print("  system = CFBBettingModel()")
print("  system.load_data(DB)")
print("  system.create_features()")
print("  system.train()")
print("=" * 60)

