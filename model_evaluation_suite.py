"""
Model Evaluation Suite for GL-SBI Project
Integrates: SBC analysis, posterior recovery tests, performance metrics
All parameters from sim_config.py and train_config.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from flax.training import train_state, checkpoints
import optax

from src.gl_jax import GLSolverJAX, SimConfig
from src.model import NREClassifier
from src.sim_config import (
    GRID_SIZE, L_SIZE, DT, EVOLVE_STEPS,
    ETA_MIN, ETA_MAX, B_MIN, B_MAX, NU_MIN, NU_MAX,
    DEFAULT_ETA, DEFAULT_B, DEFAULT_NU,
    TEST_ETAS, TEST_B_FIXED,
    ALPHA1, BETA1, D1, ALPHA2, BETA2, D2
)
from src.train_config import CKPT_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS

RESULTS_DIR = "results"
ASSETS_DIR = "assets"

for d in [RESULTS_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

# ========================================
# SHARED UTILITIES
# ========================================

def create_train_state(rng, input_shape):
    """Rebuild model structure for checkpoint loading"""
    model = NREClassifier()
    dummy_x = jnp.ones((1, *input_shape))
    dummy_theta = jnp.ones((1, 3))
    variables = model.init(rng, dummy_x, dummy_theta)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(LEARNING_RATE),
    )

def load_trained_model():
    """Load trained NRE model from checkpoint"""
    print("Loading trained model...")
    abs_ckpt_dir = os.path.abspath(CKPT_DIR)
    
    input_shape = (GRID_SIZE, GRID_SIZE, 3)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, input_shape)
    
    try:
        all_steps = checkpoints.list_steps(abs_ckpt_dir, prefix="calibrated_")
        if not all_steps:
            raise FileNotFoundError
        latest_step = max(all_steps)
        print(f"  Found checkpoint at step {latest_step}")
    except:
        print(f"  ❌ No checkpoint found in {abs_ckpt_dir}")
        print("  Please run 'python train_calibrated.py' first!")
        return None
    
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=abs_ckpt_dir,
        target=None,
        prefix="calibrated_",
        step=latest_step
    )
    
    if restored is None:
        return None
    
    # Extract params
    if hasattr(restored, 'params'):
        state = state.replace(params=restored.params)
    elif 'params' in restored:
        state = state.replace(params=restored['params'])
    else:
        state = state.replace(params=restored)
    
    print(f"  ✓ Model loaded successfully\n")
    return state

def generate_test_observation(eta_true, B_true, nu_true=0.0, seed=None):
    """Generate synthetic observation from ground truth parameters"""
    if seed is None:
        seed = int(eta_true * 10000)
    
    key_sim = jax.random.PRNGKey(seed)
    config = SimConfig(
        N=GRID_SIZE, L=L_SIZE, dt=DT,
        eta=eta_true, B=B_true, nu=nu_true,
        alpha1=ALPHA1, beta1=BETA1, D1=D1,
        alpha2=ALPHA2, beta2=BETA2, D2=D2
    )
    solver = GLSolverJAX(config)
    
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    p1_f, p2_f = solver.evolve(psi1, psi2, EVOLVE_STEPS)
    
    # Compute observables
    rho1 = jnp.abs(p1_f)**2
    rho2 = jnp.abs(p2_f)**2
    Jx, Jy = GLSolverJAX.compute_current(p1_f, config)
    curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config)
    
    obs_img = jnp.stack([rho1, rho2, curl_J], axis=-1)
    return jnp.expand_dims(obs_img, axis=0)

def add_config_box(fig, analysis_type=""):
    """Add configuration info to figure"""
    config_text = (
        f"{analysis_type}\n"
        f"Grid: {GRID_SIZE}×{GRID_SIZE}\n"
        f"η ∈ [{ETA_MIN:.2f}, {ETA_MAX:.2f}]\n"
        f"B ∈ [{B_MIN:.2f}, {B_MAX:.2f}]\n"
        f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}"
    )
    
    fig.text(0.02, 0.98, config_text, transform=fig.transFigure,
             fontsize=8, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

# ========================================
# ANALYSIS 1: Posterior Recovery Tests
# ========================================

def compute_posterior_1d(state, obs_img, B_fixed, nu_fixed=0.0, resolution=100):
    """Compute 1D posterior over η"""
    test_etas = jnp.linspace(ETA_MIN, ETA_MAX, resolution)
    logits_list = []
    
    for test_eta in test_etas:
        theta_test = jnp.array([[test_eta, B_fixed, nu_fixed]])
        logit = state.apply_fn({'params': state.params}, obs_img, theta_test)
        logits_list.append(logit[0, 0])
    
    logits = jnp.array(logits_list)
    probs = jax.nn.sigmoid(logits)
    probs_normalized = probs / jnp.sum(probs)
    
    return test_etas, probs_normalized

def compute_credible_interval(etas, probs, confidence=0.68):
    """Compute credible interval from discrete posterior"""
    sorted_indices = jnp.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_etas = etas[sorted_indices]
    
    cumsum = jnp.cumsum(sorted_probs)
    mask = cumsum <= confidence
    credible_etas = sorted_etas[mask]
    
    if len(credible_etas) == 0:
        peak_eta = etas[jnp.argmax(probs)]
        return float(peak_eta), float(peak_eta)
    
    return float(jnp.min(credible_etas)), float(jnp.max(credible_etas))

def compute_posterior_statistics(etas, probs, eta_true):
    """Compute comprehensive posterior statistics"""
    mean = float(jnp.sum(etas * probs))
    variance = float(jnp.sum((etas - mean)**2 * probs))
    std = float(jnp.sqrt(variance))
    
    ci_68_low, ci_68_high = compute_credible_interval(etas, probs, 0.68)
    ci_95_low, ci_95_high = compute_credible_interval(etas, probs, 0.95)
    
    mae = float(jnp.abs(mean - eta_true))
    in_68 = (eta_true >= ci_68_low) and (eta_true <= ci_68_high)
    in_95 = (eta_true >= ci_95_low) and (eta_true <= ci_95_high)
    
    return {
        'eta_true': eta_true,
        'posterior_mean': mean,
        'posterior_std': std,
        'mae': mae,
        'ci_68_low': ci_68_low,
        'ci_68_high': ci_68_high,
        'ci_68_width': ci_68_high - ci_68_low,
        'ci_95_low': ci_95_low,
        'ci_95_high': ci_95_high,
        'ci_95_width': ci_95_high - ci_95_low,
        'in_68_ci': in_68,
        'in_95_ci': in_95
    }

def run_posterior_recovery_tests(state):
    """Generate multi-panel posterior recovery figure"""
    print("=" * 70)
    print("  ANALYSIS 1: Posterior Recovery Tests")
    print("=" * 70)
    print(f"  Test cases: {TEST_ETAS}")
    print(f"  Fixed B: {TEST_B_FIXED}\n")
    
    n_tests = len(TEST_ETAS)
    fig, axes = plt.subplots(1, n_tests, figsize=(4*n_tests, 3.5))
    
    if n_tests == 1:
        axes = [axes]
    
    all_stats = []
    
    for idx, (ax, eta_true) in enumerate(zip(axes, TEST_ETAS)):
        print(f"  Test {idx+1}/{n_tests}: η={eta_true:.2f}")
        
        obs_img = generate_test_observation(eta_true, TEST_B_FIXED)
        etas, probs = compute_posterior_1d(state, obs_img, TEST_B_FIXED)
        stats = compute_posterior_statistics(etas, probs, eta_true)
        all_stats.append(stats)
        
        # Plot
        ax.plot(etas, probs, 'b-', linewidth=2, label='Posterior')
        ax.axvline(eta_true, color='red', linestyle='--', linewidth=2,
                   label=f'True η={eta_true:.2f}')
        ax.axvspan(stats['ci_68_low'], stats['ci_68_high'],
                   alpha=0.2, color='blue', label='68% CI')
        
        ax.set_xlabel(r'$\eta$', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'Ground Truth: η={eta_true:.2f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    add_config_box(fig, "Posterior Recovery")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(ASSETS_DIR, 'posterior_recovery.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {save_path}\n")
    plt.close()
    
    # Generate metrics table
    generate_metrics_table(all_stats)
    
    return all_stats

def generate_metrics_table(all_stats):
    """Generate performance metrics table"""
    df = pd.DataFrame(all_stats)
    
    summary = {
        'MAE': f"{df['mae'].mean():.3f} ± {df['mae'].std():.3f}",
        'RMSE': f"{np.sqrt((df['mae']**2).mean()):.3f}",
        'Mean Posterior Std': f"{df['posterior_std'].mean():.3f}",
        'Coverage 68%': f"{df['in_68_ci'].mean()*100:.1f}%",
        'Coverage 95%': f"{df['in_95_ci'].mean()*100:.1f}%",
    }
    
    print("  Performance Metrics:")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("\n  Summary:")
    for key, val in summary.items():
        print(f"    {key:20s}: {val}")
    
    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, 'performance_metrics.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n  ✓ Saved: {csv_path}")
    
    # Generate LaTeX
    latex_path = os.path.join(RESULTS_DIR, 'metrics_table.tex')
    with open(latex_path, 'w') as f:
        f.write(generate_latex_table(df, summary))
    print(f"  ✓ Saved: {latex_path}\n")

def generate_latex_table(df, summary):
    """Generate LaTeX table"""
    latex = r"""\begin{table}[h]
\centering
\caption{NRE Performance Metrics}
\label{tab:metrics}
\begin{tabular}{cccccc}
\hline
$\eta_{true}$ & Post. Mean & Post. Std & MAE & 68\% CI & 95\% CI \\
\hline
"""
    for _, row in df.iterrows():
        latex += (f"{row['eta_true']:.2f} & {row['posterior_mean']:.3f} & "
                  f"{row['posterior_std']:.3f} & {row['mae']:.3f} & "
                  f"{row['ci_68_width']:.3f} & {row['ci_95_width']:.3f} \\\\\n")
    
    latex += r"\hline" + "\n"
    for key, val in summary.items():
        latex += f"\\multicolumn{{6}}{{l}}{{{key}: {val}}} \\\\\n"
    latex += r"\hline" + "\n\\end{tabular}\n\\end{table}"
    
    return latex

# ========================================
# ANALYSIS 2: SBC Calibration
# ========================================

def compute_posterior_cdf_at_true(state, eta_true, B_fixed, resolution=200):
    """Compute P(η < η_true | x) for SBC"""
    obs_img = generate_test_observation(
        eta_true, B_fixed, 
        seed=int(eta_true * 10000 + np.random.randint(1000))
    )
    
    test_etas = jnp.linspace(ETA_MIN, ETA_MAX, resolution)
    logits_list = []
    
    for test_eta in test_etas:
        theta_test = jnp.array([[test_eta, B_fixed, 0.0]])
        logit = state.apply_fn({'params': state.params}, obs_img, theta_test)
        logits_list.append(logit[0, 0])
    
    logits = jnp.array(logits_list)
    probs = jax.nn.sigmoid(logits)
    probs_normalized = probs / jnp.sum(probs)
    
    mask = test_etas <= eta_true
    cdf_at_true = float(jnp.sum(probs_normalized[mask]))
    
    return cdf_at_true

def run_sbc_analysis(state, n_samples=100):
    """Run Simulation-Based Calibration"""
    print("=" * 70)
    print("  ANALYSIS 2: Simulation-Based Calibration")
    print("=" * 70)
    print(f"  Samples: {n_samples}")
    print(f"  Fixed B: {TEST_B_FIXED}\n")
    
    np.random.seed(42)
    test_etas = np.random.uniform(ETA_MIN, ETA_MAX, n_samples)
    
    ranks = []
    for eta_true in tqdm(test_etas, desc="  SBC Progress"):
        rank = compute_posterior_cdf_at_true(state, eta_true, TEST_B_FIXED)
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # Plot calibration
    plot_sbc_results(ranks)
    
    return ranks

def plot_sbc_results(ranks):
    """Generate SBC calibration plots"""
    sorted_ranks = np.sort(ranks)
    empirical_cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    expected_cdf = sorted_ranks
    mad = np.mean(np.abs(empirical_cdf - expected_cdf))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Rank histogram
    ax = axes[0]
    ax.hist(ranks, bins=20, density=True, alpha=0.7, 
            color='teal', edgecolor='black')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=2, 
               label='Ideal Uniform')
    ax.set_xlabel('Rank Statistic', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('SBC Rank Histogram', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PP plot
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, 
            label='Perfect Calibration')
    ax.plot(sorted_ranks, empirical_cdf, 'b-', linewidth=2.5, 
            label='Empirical CDF')
    
    n = len(ranks)
    epsilon = 1.36 / np.sqrt(n)
    ax.fill_between([0, 1], [0-epsilon, 1-epsilon], [0+epsilon, 1+epsilon],
                     alpha=0.2, color='gray', label='95% Confidence')
    
    ax.set_xlabel('Expected Confidence', fontsize=12)
    ax.set_ylabel('Empirical Coverage', fontsize=12)
    ax.set_title(f'Calibration Curve (MAD={mad:.4f})', 
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    add_config_box(fig, f"SBC: N={len(ranks)}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(RESULTS_DIR, 'sbc_calibration.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {save_path}")
    
    # Save ranks for diagnosis
    np.save(os.path.join(RESULTS_DIR, 'sbc_ranks.npy'), ranks)
    print(f"  ✓ Saved: {RESULTS_DIR}/sbc_ranks.npy")
    
    print(f"\n  MAD Score: {mad:.4f}")
    if mad < 0.05:
        print("  ★★★ Excellent calibration!")
    elif mad < 0.10:
        print("  ★★☆ Good calibration")
    else:
        print("  ★☆☆ Poor calibration - consider retraining")
    print()

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    print("\n" + "=" * 70)
    print("  MODEL EVALUATION SUITE FOR GL-SBI")
    print("=" * 70)
    
    print("\n[Configuration]")
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  η range: [{ETA_MIN}, {ETA_MAX}]")
    print(f"  Training: {EPOCHS} epochs, LR={LEARNING_RATE}")
    print(f"  Checkpoint: {CKPT_DIR}\n")
    
    # Load model
    state = load_trained_model()
    if state is None:
        return
    
    # Run analyses
    print("Running evaluations...\n")
    
    # Analysis 1: Posterior recovery
    all_stats = run_posterior_recovery_tests(state)
    
    # Analysis 2: SBC calibration
    ranks = run_sbc_analysis(state, n_samples=100)
    
    print("=" * 70)
    print("  ✓ ALL EVALUATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - {ASSETS_DIR}/posterior_recovery.png")
    print(f"  - {RESULTS_DIR}/performance_metrics.csv")
    print(f"  - {RESULTS_DIR}/metrics_table.tex")
    print(f"  - {RESULTS_DIR}/sbc_calibration.png")
    print(f"  - {RESULTS_DIR}/sbc_ranks.npy")
    print("\nNext step: Run 'python diagnose_calibration.py' for detailed diagnostics")
    print("=" * 70)

if __name__ == "__main__":
    main()