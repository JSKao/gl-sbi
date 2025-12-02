"""
/diagnose_calibration.py

"""

import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"

def analyze_calibration():
    """Analyz the result of SBC"""
    
    # ranks
    ranks = np.load(os.path.join(RESULTS_DIR, 'sbc_ranks.npy'))
    
    print("="*70)
    print("CALIBRATION DIAGNOSIS")
    print("="*70)
    
    
    print(f"\n[1] Basic Statistics:")
    print(f"  Number of samples: {len(ranks)}")
    print(f"  Mean rank: {ranks.mean():.3f} (should be ~0.5)")
    print(f"  Std rank: {ranks.std():.3f} (should be ~0.29)")
    print(f"  Min rank: {ranks.min():.3f}")
    print(f"  Max rank: {ranks.max():.3f}")
    
    # check the type of deviation
    sorted_ranks = np.sort(ranks)
    empirical_cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    deviation = empirical_cdf - sorted_ranks
    
    print(f"\n[2] Deviation Pattern:")
    print(f"  Mean deviation: {deviation.mean():.3f}")
    if deviation.mean() > 0.05:
        print("    OVERCONFIDENT: Model underestimates uncertainty")
        print("      → Posteriors are too narrow")
    elif deviation.mean() < -0.05:
        print("    UNDERCONFIDENT: Model overestimates uncertainty")
        print("      → Posteriors are too wide")
    else:
        print("   No systematic bias")
    
    # check the performance in different region
    print(f"\n[3] Regional Analysis:")
    for region, low, high in [("Low", 0.0, 0.33), ("Mid", 0.33, 0.67), ("High", 0.67, 1.0)]:
        mask = (sorted_ranks >= low) & (sorted_ranks < high)
        if mask.sum() > 0:
            local_mad = np.abs(deviation[mask]).mean()
            print(f"  {region} confidence ({low:.2f}-{high:.2f}): MAD = {local_mad:.3f}")
    
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Calibration curve with deviation
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect')
    ax.plot(sorted_ranks, empirical_cdf, 'b-', linewidth=2.5, label='Empirical')
    ax.fill_between(sorted_ranks, sorted_ranks, empirical_cdf, 
                     alpha=0.3, color='red', label='Deviation')
    ax.set_xlabel('Expected CDF', fontsize=12)
    ax.set_ylabel('Empirical CDF', fontsize=12)
    ax.set_title(f'Calibration Curve (MAD={np.abs(deviation).mean():.3f})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Deviation profile
    ax = axes[0, 1]
    ax.plot(sorted_ranks, deviation, 'o-', color='crimson', markersize=4)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between([0, 1], [-0.05, -0.05], [0.05, 0.05], alpha=0.2, color='green')
    ax.set_xlabel('Expected CDF', fontsize=12)
    ax.set_ylabel('Deviation (Empirical - Expected)', fontsize=12)
    ax.set_title('Deviation Profile', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of ranks
    ax = axes[1, 0]
    ax.hist(ranks, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Expected mean')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Rank Distribution (Mean={ranks.mean():.3f})', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative deviation
    ax = axes[1, 1]
    cumulative_deviation = np.cumsum(np.abs(deviation))
    ax.plot(sorted_ranks, cumulative_deviation, 'g-', linewidth=2)
    ax.set_xlabel('Expected CDF', fontsize=12)
    ax.set_ylabel('Cumulative Absolute Deviation', fontsize=12)
    ax.set_title('Where Does Error Accumulate?', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'calibration_diagnosis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved diagnostic plot to {save_path}")
    plt.close()
    
    
    print(f"\n[4] Recommendations:")
    if np.abs(deviation.mean()) > 0.05:
        if deviation.mean() > 0:
            print("  → Increase model capacity (more layers/neurons)")
            print("  → Train longer (more epochs)")
            print("  → Reduce regularization")
        else:
            print("  → Add regularization (dropout, weight decay)")
            print("  → Use larger training dataset")
            print("  → Apply label smoothing")
    
    max_deviation_idx = np.argmax(np.abs(deviation))
    print(f"  Worst region: CDF ≈ {sorted_ranks[max_deviation_idx]:.2f} "
          f"(deviation = {deviation[max_deviation_idx]:+.3f})")

if __name__ == "__main__":
    analyze_calibration()