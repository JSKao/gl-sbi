"""
NRE vs. Baseline Comparison Experiment

Demonstrates the advantage of deep learning (NRE) over traditional
statistical methods (S(k)-based inference) for parameter estimation.

This is a methodological validation experiment for the paper.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
import os

from src.gl_jax import GLSolverJAX, SimConfig
from src.model import CNNEncoder
from src.sim_config import (
    GRID_SIZE, L_SIZE, DT, EVOLVE_STEPS,
    DEFAULT_ETA, DEFAULT_B,
    NU_MIN, NU_MAX,
    ALPHA1, BETA1, D1, ALPHA2, BETA2, D2
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================================
# STRUCTURE FACTOR ANALYZER
# ========================================

class StructureFactorAnalyzer:
    """Compute S(k) for baseline method"""
    
    def __init__(self, N, L):
        self.N = N
        self.L = L
        dx = L / N
        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.K_mag = np.sqrt(KX**2 + KY**2)
        self.k_bins = np.linspace(0, np.max(self.K_mag)/2, N//2)
    
    def compute_sk_1d(self, density_field):
        """Returns 1D S(k) vector for ML input"""
        delta_rho = density_field - jnp.mean(density_field)
        rho_k = jnp.fft.fft2(delta_rho)
        S_k_2d = (jnp.abs(rho_k)**2) / (self.N**2)
        
        # Radial binning
        S_k_2d_np = np.array(S_k_2d)
        digitized = np.digitize(self.K_mag.ravel(), self.k_bins)
        
        sk_1d = []
        for i in range(1, len(self.k_bins)):
            mask = digitized == i
            if mask.sum() > 0:
                sk_1d.append(S_k_2d_np.ravel()[mask].mean())
            else:
                sk_1d.append(0.0)
        
        return np.array(sk_1d)

# ========================================
# DATA GENERATION
# ========================================

def generate_comparison_dataset(n_samples=200):
    """
    Generate dataset with varying nu parameter.
    Returns both raw images and S(k) features for fair comparison.
    """
    print(f"Generating {n_samples} samples...")
    print(f"  ν range: [{NU_MIN}, {NU_MAX}]")
    print(f"  Fixed: η={DEFAULT_ETA}, B={DEFAULT_B}")
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE}\n")
    
    analyzer = StructureFactorAnalyzer(GRID_SIZE, L_SIZE)
    
    X_images = []
    X_sk = []
    Y_nu = []
    
    key = jax.random.PRNGKey(999)
    
    for i in tqdm(range(n_samples), desc="Simulating"):
        key, subk = jax.random.split(key)
        
        # Sample nu uniformly
        nu_val = float(jax.random.uniform(subk, minval=NU_MIN, maxval=NU_MAX))
        
        # Create config
        cfg = SimConfig(
            N=GRID_SIZE, L=L_SIZE, dt=DT,
            eta=DEFAULT_ETA, B=DEFAULT_B, nu=nu_val,
            alpha1=ALPHA1, beta1=BETA1, D1=D1,
            alpha2=ALPHA2, beta2=BETA2, D2=D2
        )
        solver = GLSolverJAX(cfg)
        
        # Run simulation
        p1, p2 = GLSolverJAX.initialize_state(cfg, key)
        p1, p2 = solver.evolve(p1, p2, steps=EVOLVE_STEPS)
        
        # Extract features
        rho1 = jnp.abs(p1)**2
        rho2 = jnp.abs(p2)**2
        rho_total = rho1 + rho2
        
        # Image for CNN (2 channels: rho1, rho2)
        img = jnp.stack([rho1, rho2], axis=-1)
        
        # S(k) for baseline
        sk_1d = analyzer.compute_sk_1d(rho_total)
        
        X_images.append(img)
        X_sk.append(sk_1d)
        Y_nu.append(nu_val)
    
    return jnp.array(X_images), jnp.array(X_sk), jnp.array(Y_nu)

# ========================================
# MODELS
# ========================================

class SkRegressor(nn.Module):
    """Baseline: Regressor using S(k) features"""
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class ImageRegressor(nn.Module):
    """Deep Learning: CNN-based regressor (similar to NRE)"""
    
    @nn.compact
    def __call__(self, x):
        # Use existing CNN encoder
        x = CNNEncoder(output_dim=128)(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# ========================================
# TRAINING
# ========================================

def train_regressor(model, x_train, y_train, x_val, y_val, 
                   epochs=300, lr=1e-3, patience=30):
    """Train regression model with early stopping"""
    
    key = jax.random.PRNGKey(0)
    params = model.init(key, x_train[:1])
    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=tx
    )
    
    @jax.jit
    def train_step(state, batch_x, batch_y):
        def loss_fn(p):
            pred = state.apply_fn(p, batch_x)
            return jnp.mean((pred - batch_y)**2)
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss
    
    @jax.jit
    def eval_step(state, x, y):
        pred = state.apply_fn(state.params, x)
        return jnp.mean((pred - y)**2)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        state, train_loss = train_step(state, x_train, y_train)
        val_loss = eval_step(state, x_val, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
        
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: Train={train_loss:.5f}, Val={val_loss:.5f}")
    
    return best_state

# ========================================
# EVALUATION & VISUALIZATION
# ========================================

def evaluate_and_plot(state_sk, state_cnn, test_sk, test_img, test_y):
    """Compare both methods and generate publication figure"""
    
    # Predictions
    pred_sk = state_sk.apply_fn(state_sk.params, test_sk)
    pred_cnn = state_cnn.apply_fn(state_cnn.params, test_img)
    
    # Metrics
    mse_sk = float(jnp.mean((pred_sk - test_y)**2))
    rmse_sk = float(jnp.sqrt(mse_sk))
    mae_sk = float(jnp.mean(jnp.abs(pred_sk - test_y)))
    
    mse_cnn = float(jnp.mean((pred_cnn - test_y)**2))
    rmse_cnn = float(jnp.sqrt(mse_cnn))
    mae_cnn = float(jnp.mean(jnp.abs(pred_cnn - test_y)))
    
    # R² scores
    ss_tot = jnp.sum((test_y - jnp.mean(test_y))**2)
    ss_res_sk = jnp.sum((test_y - pred_sk)**2)
    ss_res_cnn = jnp.sum((test_y - pred_cnn)**2)
    r2_sk = float(1 - ss_res_sk / ss_tot)
    r2_cnn = float(1 - ss_res_cnn / ss_tot)
    
    # Print results
    print("\n" + "="*70)
    print("  COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Metric':<20} {'S(k) Baseline':<20} {'CNN (Ours)':<20}")
    print("-"*70)
    print(f"{'MSE':<20} {mse_sk:<20.6f} {mse_cnn:<20.6f}")
    print(f"{'RMSE':<20} {rmse_sk:<20.6f} {rmse_cnn:<20.6f}")
    print(f"{'MAE':<20} {mae_sk:<20.6f} {mae_cnn:<20.6f}")
    print(f"{'R²':<20} {r2_sk:<20.4f} {r2_cnn:<20.4f}")
    print("-"*70)
    print(f"{'Improvement':<20} {'-':<20} {f'{mse_sk/mse_cnn:.2f}x better':<20}")
    print("="*70 + "\n")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: S(k) Baseline
    ax = axes[0]
    ax.scatter(test_y, pred_sk, alpha=0.6, s=50, 
               color='steelblue', edgecolor='black', linewidth=0.5)
    ax.plot([NU_MIN, NU_MAX], [NU_MIN, NU_MAX], 'k--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel(r'True $\nu$', fontsize=13)
    ax.set_ylabel(r'Predicted $\nu$', fontsize=13)
    ax.set_title(f'Baseline: S(k) Features\nRMSE={rmse_sk:.4f}, R²={r2_sk:.3f}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([NU_MIN-0.02, NU_MAX+0.02])
    ax.set_ylim([NU_MIN-0.02, NU_MAX+0.02])
    
    # Plot 2: CNN (Ours)
    ax = axes[1]
    ax.scatter(test_y, pred_cnn, alpha=0.6, s=50,
               color='orange', edgecolor='black', linewidth=0.5)
    ax.plot([NU_MIN, NU_MAX], [NU_MIN, NU_MAX], 'k--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel(r'True $\nu$', fontsize=13)
    ax.set_ylabel(r'Predicted $\nu$', fontsize=13)
    ax.set_title(f'Deep Learning: Raw Image\nRMSE={rmse_cnn:.4f}, R²={r2_cnn:.3f}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([NU_MIN-0.02, NU_MAX+0.02])
    ax.set_ylim([NU_MIN-0.02, NU_MAX+0.02])
    
    # Add config box
    config_text = (
        f"Comparison Experiment\n"
        f"Grid: {GRID_SIZE}×{GRID_SIZE}\n"
        f"η={DEFAULT_ETA}, B={DEFAULT_B}\n"
        f"ν ∈ [{NU_MIN}, {NU_MAX}]\n"
        f"Test samples: {len(test_y)}\n"
        f"Improvement: {mse_sk/mse_cnn:.2f}×"
    )
    
    fig.text(0.02, 0.98, config_text, transform=fig.transFigure,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(RESULTS_DIR, 'nre_vs_baseline.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison figure: {save_path}")
    plt.close()
    
    # Save metrics to file
    metrics_path = os.path.join(RESULTS_DIR, 'comparison_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("NRE vs Baseline Comparison\n")
        f.write("="*50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Grid: {GRID_SIZE}×{GRID_SIZE}\n")
        f.write(f"  η={DEFAULT_ETA}, B={DEFAULT_B}\n")
        f.write(f"  ν range: [{NU_MIN}, {NU_MAX}]\n\n")
        f.write(f"Results:\n")
        f.write(f"  S(k) Baseline:\n")
        f.write(f"    MSE:  {mse_sk:.6f}\n")
        f.write(f"    RMSE: {rmse_sk:.6f}\n")
        f.write(f"    MAE:  {mae_sk:.6f}\n")
        f.write(f"    R²:   {r2_sk:.4f}\n\n")
        f.write(f"  CNN (Deep Learning):\n")
        f.write(f"    MSE:  {mse_cnn:.6f}\n")
        f.write(f"    RMSE: {rmse_cnn:.6f}\n")
        f.write(f"    MAE:  {mae_cnn:.6f}\n")
        f.write(f"    R²:   {r2_cnn:.4f}\n\n")
        f.write(f"  Improvement: {mse_sk/mse_cnn:.2f}× lower MSE\n")
    
    print(f"✓ Saved metrics: {metrics_path}\n")

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    print("\n" + "="*70)
    print("  METHODOLOGICAL COMPARISON: NRE vs. S(k) Baseline")
    print("="*70)
    
    # Configuration
    N_SAMPLES = 300  # Increase for paper quality
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    # TEST_RATIO = 0.15 (implicit)
    
    print(f"\n[Configuration]")
    print(f"  Total samples: {N_SAMPLES}")
    print(f"  Train/Val/Test: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{1-TRAIN_RATIO-VAL_RATIO:.0%}")
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Evolution steps: {EVOLVE_STEPS}\n")
    
    # 1. Generate dataset
    print("[1/4] Generating dataset...")
    X_img, X_sk, Y_nu = generate_comparison_dataset(N_SAMPLES)
    Y_nu = Y_nu.reshape(-1, 1)
    
    # 2. Split data
    n_train = int(N_SAMPLES * TRAIN_RATIO)
    n_val = int(N_SAMPLES * VAL_RATIO)
    
    train_img = X_img[:n_train]
    val_img = X_img[n_train:n_train+n_val]
    test_img = X_img[n_train+n_val:]
    
    train_sk = X_sk[:n_train]
    val_sk = X_sk[n_train:n_train+n_val]
    test_sk = X_sk[n_train+n_val:]
    
    train_y = Y_nu[:n_train]
    val_y = Y_nu[n_train:n_train+n_val]
    test_y = Y_nu[n_train+n_val:]
    
    print(f"\n  Split: Train={len(train_y)}, Val={len(val_y)}, Test={len(test_y)}\n")
    
    # 3. Train S(k) baseline
    print("[2/4] Training S(k) Baseline...")
    model_sk = SkRegressor()
    state_sk = train_regressor(model_sk, train_sk, train_y, val_sk, val_y)
    print("  ✓ Baseline trained\n")
    
    # 4. Train CNN
    print("[3/4] Training CNN (Deep Learning)...")
    model_cnn = ImageRegressor()
    state_cnn = train_regressor(model_cnn, train_img, train_y, val_img, val_y)
    print("  ✓ CNN trained\n")
    
    # 5. Evaluate and visualize
    print("[4/4] Evaluating and generating figures...")
    evaluate_and_plot(state_sk, state_cnn, test_sk, test_img, test_y)
    
    print("="*70)
    print("  ✓ COMPARISON COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()