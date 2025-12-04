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
from src.model import CNNEncoder # Use your existing encoder

# --- 1. Data Generation ---
class DataGen:
    def __init__(self, N=64):
        self.N = N
        # Pre-compute k-bins for S(k)
        L = 20.0
        dx = L/N
        kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.K_mag = np.sqrt(KX**2 + KY**2)
        self.k_bins = np.linspace(0, np.max(self.K_mag)/2, N//2)

    def get_batch(self, batch_size, key):
        # Sample nu only, fix eta and B
        k_nu, k_sim = jax.random.split(key)
        nu = jax.random.uniform(k_nu, (batch_size, 1), minval=0.0, maxval=0.3)
        
        # We need to simulate loop (simplified for speed here, ideally use vmap or pre-gen)
        # For demonstration, let's assume we have a way to get X from nu quickly
        # Or we generate a small dataset first.
        return nu

def generate_dataset(n_samples=500):
    print(f"Generating {n_samples} samples for comparison...")
    config_base = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04)
    solver = GLSolverJAX(config_base)
    analyzer = DataGen(64)
    
    X_images = []
    X_sk = []
    Y_nu = []
    
    key = jax.random.PRNGKey(999)
    
    for i in tqdm(range(n_samples)):
        key, subk = jax.random.split(key)
        nu_val = jax.random.uniform(subk, minval=0.0, maxval=0.3)
        
        # Update config
        cfg = SimConfig(N=64, L=20.0, dt=0.002, eta=0.8, B=0.04, nu=nu_val)
        solver.cfg = cfg # Hacky update
        
        # Init and Evolve
        p1, p2 = GLSolverJAX.initialize_state(cfg, key)
        p1, p2 = solver.evolve(p1, p2, steps=2000)
        
        # Image
        rho = jnp.abs(p1)**2 + jnp.abs(p2)**2
        # Simple normalization for CNN
        img = jnp.stack([jnp.abs(p1)**2, jnp.abs(p2)**2], axis=-1)
        
        # S(k)
        delta_rho = rho - jnp.mean(rho)
        rho_k = jnp.fft.fft2(delta_rho)
        S_k_2d = (jnp.abs(rho_k)**2) / (64**2)
        S_k_2d = np.array(S_k_2d)
        digitized = np.digitize(analyzer.K_mag.ravel(), analyzer.k_bins)
        sk_1d = [S_k_2d.ravel()[digitized == j].mean() for j in range(1, len(analyzer.k_bins))]
        
        X_images.append(img)
        X_sk.append(np.array(sk_1d))
        Y_nu.append(nu_val)
        
    return jnp.array(X_images), jnp.array(X_sk), jnp.array(Y_nu)

# --- 2. Models ---
class SkRegressor(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x) # Predict nu directly
        return x

class CNNRegressor(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Reuse your CNN structure
        x = CNNEncoder(output_dim=64)(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# --- 3. Training Loop Helper ---
def train_model(model, x, y, epochs=200, lr=1e-3):
    key = jax.random.PRNGKey(0)
    params = model.init(key, x[:1])
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    @jax.jit
    def step(state, batch_x, batch_y):
        def loss_fn(p):
            pred = state.apply_fn(p, batch_x)
            return jnp.mean((pred - batch_y)**2)
        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)
    
    # Simple full batch training for demo
    for _ in range(epochs):
        state = step(state, x, y)
    return state

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists("results"): os.makedirs("results")
    
    # 1. Generate Data (Small set for speed demo)
    # Increase to 500-1000 for paper quality results
    N_SAMPLES = 200 
    print(f"Running Comparison on {N_SAMPLES} samples...")
    
    X_img, X_sk, Y_nu = generate_dataset(N_SAMPLES)
    Y_nu = Y_nu.reshape(-1, 1)
    
    # Split
    split = int(0.8 * N_SAMPLES)
    train_img, test_img = X_img[:split], X_img[split:]
    train_sk, test_sk = X_sk[:split], X_sk[split:]
    train_y, test_y = Y_nu[:split], Y_nu[split:]
    
    # 2. Train S(k) Regressor
    print("Training S(k) Baseline...")
    model_sk = SkRegressor()
    state_sk = train_model(model_sk, train_sk, train_y)
    pred_sk = state_sk.apply_fn(state_sk.params, test_sk)
    
    # 3. Train CNN (NRE-like) Regressor
    print("Training CNN (Deep Learning)...")
    model_cnn = CNNRegressor()
    state_cnn = train_model(model_cnn, train_img, train_y)
    pred_cnn = state_cnn.apply_fn(state_cnn.params, test_img)
    
    # 4. Analysis
    mse_sk = np.mean((pred_sk - test_y)**2)
    mse_cnn = np.mean((pred_cnn - test_y)**2)
    
    print(f"\nResults:")
    print(f"  S(k) Baseline MSE: {mse_sk:.5f}")
    print(f"  CNN (Image) MSE:   {mse_cnn:.5f}")
    print(f"  Improvement:       {mse_sk/mse_cnn:.2f}x lower error")
    
    # 5. Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(test_y, pred_sk, alpha=0.6, label=f'S(k) (MSE={mse_sk:.4f})')
    plt.plot([0, 0.3], [0, 0.3], 'k--')
    plt.title("Baseline: Inference from S(k)")
    plt.xlabel("True $\\nu$")
    plt.ylabel("Predicted $\\nu$")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(test_y, pred_cnn, alpha=0.6, color='orange', label=f'CNN (MSE={mse_cnn:.4f})')
    plt.plot([0, 0.3], [0, 0.3], 'k--')
    plt.title("Ours: Inference from Raw Image")
    plt.xlabel("True $\\nu$")
    plt.ylabel("Predicted $\\nu$")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/nre_vs_sk_comparison.png")
    print("Comparison plot saved.")