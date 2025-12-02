"""
scripts/calibration_analysis.py

Simulation-Based Calibration (SBC) analysis to verify posterior reliability.
Generates Fig. B1 (Appendix): Coverage calibration curve
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from src.gl_jax import GLSolverJAX, SimConfig
from src.model import NREClassifier
from flax.training import train_state, checkpoints
import optax

from src.sim_config import (
    GRID_SIZE, EVOLVE_STEPS,
    ETA_MIN, ETA_MAX, TEST_B_FIXED
)

CKPT_DIR = os.path.abspath("checkpoints")
RESULTS_DIR = "results"


def create_train_state(rng, input_shape):
    model = NREClassifier()
    dummy_x = jnp.ones((1, *input_shape))
    dummy_theta = jnp.ones((1, 3))
    variables = model.init(rng, dummy_x, dummy_theta)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(1e-3),
    )


def compute_posterior_cdf_at_true(state, eta_true, B_fixed, resolution=200):
    """Compute P(η < η_true | x) for calibration"""
    # Generate observation
    key_sim = jax.random.PRNGKey(int(eta_true * 10000 + np.random.randint(1000)))
    config = SimConfig(eta=eta_true, B=B_fixed, nu=0.0, N=GRID_SIZE)
    solver = GLSolverJAX(config)
    
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    p1_f, p2_f = solver.evolve(psi1, psi2, EVOLVE_STEPS)
    
    rho1 = jnp.abs(p1_f)**2
    rho2 = jnp.abs(p2_f)**2
    Jx, Jy = GLSolverJAX.compute_current(p1_f, config)
    curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config)
    
    obs_img = jnp.stack([rho1, rho2, curl_J], axis=-1)
    obs_img = jnp.expand_dims(obs_img, axis=0)
    
    # Compute posterior
    test_etas = jnp.linspace(ETA_MIN, ETA_MAX, resolution)
    logits_list = []
    
    for test_eta in test_etas:
        theta_test = jnp.array([[test_eta, B_fixed, 0.0]])
        logit = state.apply_fn({'params': state.params}, obs_img, theta_test)
        logits_list.append(logit[0, 0])
    
    logits = jnp.array(logits_list)
    probs = jax.nn.sigmoid(logits)
    probs_normalized = probs / jnp.sum(probs)
    
    # Compute CDF at true value
    mask = test_etas <= eta_true
    cdf_at_true = float(jnp.sum(probs_normalized[mask]))
    
    return cdf_at_true


def run_sbc_analysis(state, n_samples=100):
    """Run Simulation-Based Calibration"""
    print(f"\nRunning SBC with {n_samples} samples...")
    
    # Sample etas uniformly
    np.random.seed(42)
    test_etas = np.random.uniform(ETA_MIN, ETA_MAX, n_samples)
    
    ranks = []
    for i, eta_true in enumerate(tqdm(test_etas, desc="SBC Progress")):
        rank = compute_posterior_cdf_at_true(state, eta_true, TEST_B_FIXED)
        ranks.append(rank)
    
    return np.array(ranks)


def plot_calibration(ranks):
    """Plot calibration curve (empirical CDF vs expected)"""
    sorted_ranks = np.sort(ranks)
    empirical_cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    
    # Compute MAD (Mean Absolute Deviation from diagonal)
    expected_cdf = sorted_ranks  # If perfect, empirical should equal expected
    mad = np.mean(np.abs(empirical_cdf - expected_cdf))
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
    
    # Plot empirical CDF
    ax.plot(sorted_ranks, empirical_cdf, 'b-', linewidth=2.5, label='Empirical CDF')
    
    # Shade confidence band (approximate)
    n = len(ranks)
    epsilon = 1.36 / np.sqrt(n)  # Kolmogorov-Smirnov 95% band
    ax.fill_between([0, 1], [0-epsilon, 1-epsilon], [0+epsilon, 1+epsilon], 
                     alpha=0.2, color='gray', label='95% Confidence Band')
    
    ax.set_xlabel('Expected Confidence Level', fontsize=13)
    ax.set_ylabel('Empirical Coverage', fontsize=13)
    ax.set_title(f'Coverage Calibration (MAD = {mad:.3f})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'calibration_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved calibration plot to {save_path}")
    plt.close()
    
    # Save ranks for paper appendix
    np.save(os.path.join(RESULTS_DIR, 'sbc_ranks.npy'), ranks)
    
    return mad


def main():
    print("="*70)
    print("SIMULATION-BASED CALIBRATION ANALYSIS")
    print("="*70)
    
    # Load model
    input_shape = (GRID_SIZE, GRID_SIZE, 3)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, input_shape)
    
    if os.path.exists(CKPT_DIR):
        state = checkpoints.restore_checkpoint(ckpt_dir=CKPT_DIR, target=state)
        print(f"✅ Loaded checkpoint from {CKPT_DIR}\n")
    else:
        print(f"❌ No checkpoint found!")
        return
    
    # Run SBC (this takes ~10-20 minutes for 100 samples)
    ranks = run_sbc_analysis(state, n_samples=100)
    
    # Plot results
    mad = plot_calibration(ranks)
    
    print("\n" + "="*70)
    print("SBC ANALYSIS COMPLETE")
    print("="*70)
    print(f"Mean Absolute Deviation (MAD): {mad:.4f}")
    print(f"Interpretation:")
    if mad < 0.05:
        print("  ✅ Excellent calibration! Uncertainty estimates are reliable.")
    elif mad < 0.10:
        print("  ✅ Good calibration. Minor deviation is acceptable.")
    else:
        print("  ⚠️  Poor calibration. Consider retraining or adjusting priors.")


if __name__ == "__main__":
    main()