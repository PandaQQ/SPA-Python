"""
Batch compute total variance reduction for each subject in cognitive_load_set.
Uses default SPA parameters: threshold=30μV, win_size=2s, smooth_para=2.
"""
import os, sys
import numpy as np
import mne

mne.set_log_level("ERROR")

sys.path.insert(0, "/Users/pandaqq/Documents/EdDProjects/spa-python")
from spa.core import spa_eeg

DATA_DIR = "/Users/pandaqq/Documents/EdDProjects/spa-python/cognitive_load_set"

# Collect all .set files, sorted
set_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".set")])

print(f"{'Subject':<15} {'Channels':>8} {'Duration(s)':>12} {'VarReduction(%)':>16}")
print("-" * 55)

results = []
for sf in set_files:
    subject = sf.replace(".set", "")
    path = os.path.join(DATA_DIR, sf)
    
    try:
        raw = mne.io.read_raw_eeglab(path, preload=True)
        data = raw.get_data()           # (n_ch, n_times), unit: V
        srate = raw.info["sfreq"]
        
        # Run SPA with default params
        data_spa = spa_eeg(data, srate, threshold=30e-6, win_size=2.0, smooth_para=2.0)
        
        # Compute total variance reduction
        var_before = np.var(data * 1e6, axis=1)      # per-channel variance in μV²
        var_after  = np.var(data_spa * 1e6, axis=1)
        total_reduction = (1 - var_after.sum() / var_before.sum()) * 100
        
        duration = data.shape[1] / srate
        print(f"{subject:<15} {data.shape[0]:>8} {duration:>12.1f} {total_reduction:>16.2f}")
        results.append((subject, data.shape[0], duration, total_reduction))
    except Exception as e:
        print(f"{subject:<15} ERROR: {e}")
        results.append((subject, 0, 0, float('nan')))

# Summary
print("-" * 55)
valid = [r[3] for r in results if not np.isnan(r[3])]
print(f"{'Average':<15} {'':>8} {'':>12} {np.mean(valid):>16.2f}")
print(f"{'Std Dev':<15} {'':>8} {'':>12} {np.std(valid):>16.2f}")
print(f"{'Min':<15} {'':>8} {'':>12} {np.min(valid):>16.2f}")
print(f"{'Max':<15} {'':>8} {'':>12} {np.max(valid):>16.2f}")
print(f"\nTotal subjects processed: {len(valid)}")
