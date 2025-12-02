"""
scripts/run_comprehensive_tests.py

Comprehensive evaluation script for NRE performance.
Generates:
- Fig. 3B: Multi-panel posterior recovery
- Table I: Quantitative performance metrics
- Coverage calibration plot
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from flax.training import train_state, checkpoints
import optax
from tqdm import tqdm
import pandas as pd

from src.gl_jax import GLSolverJAX, SimConfig
from src.model import NREClassifier
from src.sim_config import (
    GRID_SIZE, EVOLVE_STEPS,
    ETA_MIN, ETA_MAX,
    TEST_ETAS, TEST_B_FIXED, TEST_CASE_NU
)

CKPT_DIR = os.path.abspath("checkpoints")
RESULTS_DIR = "results"
ASSETS_DIR = "assets"

# Ensure directories exist
for d in [RESULTS_DIR, ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)


def create_train_state(rng, input_shape):
    """Rebuild model with trained weights"""
    model = NREClassifier()
    dummy_x = jnp.ones((1, *input_shape))
    dummy_theta = jnp.ones((1, 3))
    variables = model.init(rng, dummy_x, dummy_theta)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(1e-3),
    )


def generate_test_observation(eta_true, B_true, nu_true=0.0):
    """Generate a single test observation"""
    key_sim = jax.random.PRNGKey(int(eta_true * 10000))  # Deterministic but varied
    config = SimConfig(eta=eta_true, B=B_true, nu=nu_true, N=GRID_SIZE)
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


def compute_posterior_1d(state, obs_img, B_fixed, nu_fixed=0.0, resolution=100):
    """Compute 1D posterior over eta"""
    test_etas = jnp.linspace(ETA_MIN, ETA_MAX, resolution)
    logits_list = []
    
    for test_eta in test_etas:
        theta_test = jnp.array([[test_eta, B_fixed, nu_fixed]])
        logit = state.apply_fn({'params': state.params}, obs_img, theta_test)
        logits_list.append(logit[0, 0])
    
    logits = jnp.array(logits_list)
    probs = jax.nn.sigmoid(logits)
    
    # Normalize to integrate to 1 (approximate posterior)
    probs_normalized = probs / jnp.sum(probs)
    
    return test_etas, probs_normalized


def compute_credible_interval(etas, probs, confidence=0.68):
    """Compute credible interval from discrete posterior"""
    # Sort by probability
    sorted_indices = jnp.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_etas = etas[sorted_indices]
    
    # Accumulate probability mass
    cumsum = jnp.cumsum(sorted_probs)
    
    # Find indices within confidence level
    mask = cumsum <= confidence
    credible_etas = sorted_etas[mask]
    
    if len(credible_etas) == 0:
        return etas[jnp.argmax(probs)], etas[jnp.argmax(probs)]
    
    return float(jnp.min(credible_etas)), float(jnp.max(credible_etas))


def compute_posterior_statistics(etas, probs, eta_true):
    """Compute all statistics for a single posterior"""
    # Posterior mean
    mean = float(jnp.sum(etas * probs))
    
    # Posterior std
    variance = float(jnp.sum((etas - mean)**2 * probs))
    std = float(jnp.sqrt(variance))
    
    # Credible intervals
    ci_68_low, ci_68_high = compute_credible_interval(etas, probs, confidence=0.68)
    ci_95_low, ci_95_high = compute_credible_interval(etas, probs, confidence=0.95)
    
    # Error metrics
    mae = float(jnp.abs(mean - eta_true))
    
    # Check coverage
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


def generate_multipanel_figure(state, test_etas, B_fixed=TEST_B_FIXED):
    """Generate Fig. 3B: Multi-panel posterior recovery"""
    n_tests = len(test_etas)
    fig, axes = plt.subplots(1, n_tests, figsize=(4*n_tests, 3.5))
    
    if n_tests == 1:
        axes = [axes]
    
    all_stats = []
    
    for idx, (ax, eta_true) in enumerate(zip(axes, test_etas)):
        print(f"  Processing test case {idx+1}/{n_tests}: η={eta_true:.2f}")
        
        # Generate observation
        obs_img = generate_test_observation(eta_true, B_fixed)
        
        # Compute posterior
        etas, probs = compute_posterior_1d(state, obs_img, B_fixed)
        
        # Compute statistics
        stats = compute_posterior_statistics(etas, probs, eta_true)
        all_stats.append(stats)
        
        # Plot
        ax.plot(etas, probs, 'b-', linewidth=2, label='Posterior')
        ax.axvline(eta_true, color='red', linestyle='--', linewidth=2, 
                   label=f'True η={eta_true:.2f}')
        
        # Shade 68% CI
        ax.axvspan(stats['ci_68_low'], stats['ci_68_high'], 
                   alpha=0.2, color='blue', label='68% CI')
        
        ax.set_xlabel(r'$\eta$', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'Ground Truth: η={eta_true:.2f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, 'posterior_recovery_multipanel.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved multi-panel figure to {save_path}")
    plt.close()
    
    return all_stats


def generate_metrics_table(all_stats):
    """Generate Table I: Performance metrics"""
    df = pd.DataFrame(all_stats)
    
    # Compute aggregate statistics
    summary = {
        'Mean Absolute Error (MAE)': f"{df['mae'].mean():.3f} ± {df['mae'].std():.3f}",
        'Root Mean Square Error (RMSE)': f"{np.sqrt((df['mae']**2).mean()):.3f}",
        'Mean Posterior Std': f"{df['posterior_std'].mean():.3f}",
        'Coverage (68% CI)': f"{df['in_68_ci'].mean()*100:.1f}%",
        'Coverage (95% CI)': f"{df['in_95_ci'].mean()*100:.1f}%",
    }
    
    # Print LaTeX table
    print("\n" + "="*70)
    print("TABLE I: Quantitative Performance Metrics")
    print("="*70)
    print("\nIndividual Test Cases:")
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    
    print("\n\nAggregate Statistics:")
    for key, val in summary.items():
        print(f"  {key:30s}: {val}")
    
    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, 'performance_metrics.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✅ Saved detailed metrics to {csv_path}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(df)
    latex_path = os.path.join(RESULTS_DIR, 'table_metrics.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✅ Saved LaTeX table to {latex_path}")
    
    return df, summary


def generate_latex_table(df):
    """Generate LaTeX table code"""
    latex = r"""
\begin{table}[h]
\centering
\caption{Quantitative Performance Metrics for Parameter Recovery}
\label{tab:metrics}
\begin{tabular}{cccccc}
\hline
Ground Truth $\eta$ & Post. Mean & Post. Std & MAE & 68\% CI Width & 95\% CI Width \\
\hline
"""
    
    for _, row in df.iterrows():
        latex += f"{row['eta_true']:.2f} & {row['posterior_mean']:.3f} & "
        latex += f"{row['posterior_std']:.3f} & {row['mae']:.3f} & "
        latex += f"{row['ci_68_width']:.3f} & {row['ci_95_width']:.3f} \\\\\n"
    
    latex += r"""\hline
\multicolumn{6}{l}{Mean Absolute Error (MAE): """ + f"{df['mae'].mean():.3f}" + r"""} \\
\multicolumn{6}{l}{Root Mean Square Error (RMSE): """ + f"{np.sqrt((df['mae']**2).mean()):.3f}" + r"""} \\
\multicolumn{6}{l}{Coverage (68\% CI): """ + f"{df['in_68_ci'].mean()*100:.1f}\%" + r"""} \\
\multicolumn{6}{l}{Coverage (95\% CI): """ + f"{df['in_95_ci'].mean()*100:.1f}\%" + r"""} \\
\hline
\end{tabular}
\end{table}
"""
    return latex


def main():
    print("="*70)
    print("COMPREHENSIVE EVALUATION: NRE PERFORMANCE ANALYSIS")
    print("="*70)
    
    # 1. Load trained model
    print("\n[1/4] Loading trained model...")
    input_shape = (GRID_SIZE, GRID_SIZE, 3)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, input_shape)
    
    if os.path.exists(CKPT_DIR):
        state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
        print(f"     ✅ Loaded checkpoint from {CKPT_DIR}")
    else:
        print(f"     ❌ No checkpoint found at {CKPT_DIR}")
        print("     Please run 'python src/train_offline.py' first!")
        return
    
    # 2. Generate multi-panel figure
    print("\n[2/4] Generating multi-panel posterior recovery figure...")
    all_stats = generate_multipanel_figure(state, TEST_ETAS, TEST_B_FIXED)
    
    # 3. Generate metrics table
    print("\n[3/4] Computing performance metrics...")
    df, summary = generate_metrics_table(all_stats)
    
    # 4. Generate additional diagnostic plots
    print("\n[4/4] Generating diagnostic plots...")
    generate_diagnostic_plots(df)
    
    print("\n" + "="*70)
    print("✅ ALL EVALUATIONS COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - {ASSETS_DIR}/posterior_recovery_multipanel.png")
    print(f"  - {RESULTS_DIR}/performance_metrics.csv")
    print(f"  - {RESULTS_DIR}/table_metrics.tex")
    print(f"  - {ASSETS_DIR}/error_vs_eta.png")
    print(f"  - {ASSETS_DIR}/ci_width_vs_eta.png")


def generate_diagnostic_plots(df):
    """Generate additional diagnostic visualizations"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: MAE vs eta
    ax = axes[0]
    ax.plot(df['eta_true'], df['mae'], 'o-', markersize=8, linewidth=2, color='steelblue')
    ax.axhline(df['mae'].mean(), color='red', linestyle='--', 
               label=f"Mean MAE = {df['mae'].mean():.3f}")
    ax.set_xlabel(r'Ground Truth $\eta$', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Parameter Recovery Accuracy', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: CI width vs eta
    ax = axes[1]
    ax.plot(df['eta_true'], df['ci_68_width'], 'o-', markersize=8, 
            linewidth=2, color='orange', label='68% CI')
    ax.plot(df['eta_true'], df['ci_95_width'], 's-', markersize=8, 
            linewidth=2, color='purple', label='95% CI')
    ax.set_xlabel(r'Ground Truth $\eta$', fontsize=12)
    ax.set_ylabel('Credible Interval Width', fontsize=12)
    ax.set_title('Uncertainty Quantification', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, 'diagnostic_plots.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved diagnostic plots to {save_path}")
    plt.close()


if __name__ == "__main__":
    main()