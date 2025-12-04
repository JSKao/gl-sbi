"""
diagnose_calibration.py

Deep diagnostic analysis of SBC results.
Identifies over/underconfidence patterns and regional issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from src.sim_config import (
    GRID_SIZE, ETA_MIN, ETA_MAX, B_MIN, B_MAX,
    DEFAULT_ETA, DEFAULT_B
)
from src.train_config import EPOCHS, LEARNING_RATE

RESULTS_DIR = "results"

def analyze_calibration():
    """Analyze SBC results in detail"""
    
    # Load ranks
    ranks_file = os.path.join(RESULTS_DIR, 'sbc_ranks.npy')
    if not os.path.exists(ranks_file):
        print(f"❌ Error: {ranks_file} not found!")
        print("   Please run 'python model_evaluation_suite.py' first.")
        return
    
    ranks = np.load(ranks_file)
    
    print("=" * 70)
    print("  CALIBRATION DIAGNOSIS")
    print("=" * 70)
    
    print(f"\n[Configuration]")
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  η range: [{ETA_MIN}, {ETA_MAX}]")
    print(f"  Training: {EPOCHS} epochs, LR={LEARNING_RATE}")
    
    print(f"\n[1] Basic Statistics:")
    print(f"  Samples: {len(ranks)}")
    print(f"  Mean rank: {ranks.mean():.3f} (expected ~0.5)")
    print(f"  Std rank: {ranks.std():.3f} (expected ~0.29)")
    print(f"  Range: [{ranks.min():.3f}, {ranks.max():.3f}]")
    
    # Deviation analysis
    sorted_ranks = np.sort(ranks)
    empirical_cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    deviation = empirical_cdf - sorted_ranks
    
    print(f"\n[2] Deviation Pattern:")
    print(f"  Mean deviation: {deviation.mean():.3f}")
    
    if deviation.mean() > 0.05:
        print("  ⚠️  OVERCONFIDENT: Model underestimates uncertainty")
        print("      → Posteriors are too narrow")
        print("      → Ground truth often falls outside credible intervals")
    elif deviation.mean() < -0.05:
        print("  ⚠️  UNDERCONFIDENT: Model overestimates uncertainty")
        print("      → Posteriors are too wide")
        print("      → Credible intervals are overly conservative")
    else:
        print("  ✓ No systematic bias detected")
    
    # Regional analysis
    print(f"\n[3] Regional Analysis:")
    for region, low, high in [("Low", 0.0, 0.33), 
                              ("Mid", 0.33, 0.67), 
                              ("High", 0.67, 1.0)]:
        mask = (sorted_ranks >= low) & (sorted_ranks < high)
        if mask.sum() > 0:
            local_mad = np.abs(deviation[mask]).mean()
            print(f"  {region:5s} confidence [{low:.2f}-{high:.2f}]: MAD = {local_mad:.3f}")
    
    # Generate diagnostic plots
    generate_diagnostic_plots(ranks, sorted_ranks, empirical_cdf, deviation)
    
    # Recommendations
    print(f"\n[4] Recommendations:")
    if np.abs(deviation.mean()) > 0.05:
        if deviation.mean() > 0:
            print("  Overconfidence fixes:")
            print("    → Increase model capacity (wider layers)")
            print("    → Train longer (more epochs)")
            print("    → Reduce regularization / weight decay")
            print("    → Add more diverse training data")
        else:
            print("  Underconfidence fixes:")
            print("    → Add regularization (dropout, weight decay)")
            print("    → Use larger training dataset")
            print("    → Apply label smoothing")
            print("    → Reduce model capacity slightly")
    else:
        print("  ✓ Calibration is good! No major changes needed.")
    
    max_deviation_idx = np.argmax(np.abs(deviation))
    print(f"\n  Worst region: CDF ≈ {sorted_ranks[max_deviation_idx]:.2f}")
    print(f"  Deviation: {deviation[max_deviation_idx]:+.3f}")
    
    print("\n" + "=" * 70)

def generate_diagnostic_plots(ranks, sorted_ranks, empirical_cdf, deviation):
    """Generate 4-panel diagnostic visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    mad = np.abs(deviation).mean()
    
    # Plot 1: Calibration curve with shaded deviation
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect')
    ax.plot(sorted_ranks, empirical_cdf, 'b-', linewidth=2.5, label='Empirical')
    
    # Shade deviation
    over_mask = deviation > 0
    under_mask = deviation < 0
    ax.fill_between(sorted_ranks[over_mask], sorted_ranks[over_mask], 
                     empirical_cdf[over_mask], alpha=0.3, color='red', 
                     label='Overconfident')
    ax.fill_between(sorted_ranks[under_mask], sorted_ranks[under_mask], 
                     empirical_cdf[under_mask], alpha=0.3, color='blue', 
                     label='Underconfident')
    
    ax.set_xlabel('Expected CDF', fontsize=12)
    ax.set_ylabel('Empirical CDF', fontsize=12)
    ax.set_title(f'Calibration Curve (MAD={mad:.3f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Plot 2: Deviation profile
    ax = axes[0, 1]
    ax.plot(sorted_ranks, deviation, 'o-', color='crimson', 
            markersize=4, linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.fill_between([0, 1], [-0.05, -0.05], [0.05, 0.05], 
                     alpha=0.2, color='green', label='Good range')
    ax.set_xlabel('Expected CDF', fontsize=12)
    ax.set_ylabel('Deviation (Emp - Exp)', fontsize=12)
    ax.set_title('Deviation Profile', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Rank histogram
    ax = axes[1, 0]
    ax.hist(ranks, bins=20, alpha=0.7, color='steelblue', 
            edgecolor='black', density=True)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, 
               label='Uniform density')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Rank Distribution (μ={ranks.mean():.3f})', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative absolute deviation
    ax = axes[1, 1]
    cumulative_deviation = np.cumsum(np.abs(deviation))
    ax.plot(sorted_ranks, cumulative_deviation, 'g-', linewidth=2.5)
    ax.set_xlabel('Expected CDF', fontsize=12)
    ax.set_ylabel('Cumulative |Deviation|', fontsize=12)
    ax.set_title('Error Accumulation Pattern', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add config box
    config_text = (
        f"Configuration:\n"
        f"Grid: {GRID_SIZE}×{GRID_SIZE}\n"
        f"η ∈ [{ETA_MIN}, {ETA_MAX}]\n"
        f"B ∈ [{B_MIN}, {B_MAX}]\n"
        f"Samples: {len(ranks)}\n"
        f"Epochs: {EPOCHS}"
    )
    
    fig.text(0.02, 0.98, config_text, transform=fig.transFigure,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(RESULTS_DIR, 'calibration_diagnosis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved diagnostic plot: {save_path}")
    plt.close()

if __name__ == "__main__":
    analyze_calibration()