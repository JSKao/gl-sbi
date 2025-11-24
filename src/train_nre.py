import jax
import jax.numpy as jnp
from flax.training import train_state
import optax 
from dataclasses import dataclass

from src.model import NREClassifier 
from src.simulator import DataGenerator 
from src.gl_jax import GLSolverJAX, SimConfig

@dataclass
class TrainConfig:
    batch_size: int = 8       # Fixed: removed float dot
    n_samples: int = 1000     
    learning_rate: float = 1e-3
    seed: int = 42
    grid_size: int = 32       # Low res for online training
    evolve_steps: int = 100

def create_train_state(rng, learning_rate, grid_size):
    """Initializes model and optimizer."""
    model = NREClassifier()
    
    dummy_x = jnp.ones((1, grid_size, grid_size, 2))
    dummy_theta = jnp.ones((1, 2))
    
    variables = model.init(rng, dummy_x, dummy_theta)
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

@jax.jit
def train_step(state, batch_x, batch_theta):
    """
    Contrastive training step. 
    Fixed: Removed unused key_shuffle argument.
    """
    
    # Positive samples (Joint)
    pos_x = batch_x
    pos_theta = batch_theta
    pos_labels = jnp.ones((batch_x.shape[0], 1))
    
    # Negative samples (Marginal) via permutation
    neg_x = batch_x
    neg_theta = jnp.roll(batch_theta, shift=1, axis=0) 
    neg_labels = jnp.zeros((batch_x.shape[0], 1))
    
    # Concatenate
    train_x = jnp.concatenate([pos_x, neg_x], axis=0)
    train_theta = jnp.concatenate([pos_theta, neg_theta], axis=0)
    train_labels = jnp.concatenate([pos_labels, neg_labels], axis=0)
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, train_x, train_theta)
        loss = optax.sigmoid_binary_cross_entropy(logits, train_labels).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)   
    return new_state, loss

if __name__ == "__main__":
    cfg = TrainConfig()
    print(f"🚀 Starting NRE Training | Batch: {cfg.batch_size} | Grid: {cfg.grid_size}")
    
    # 1. Setup
    master_key = jax.random.PRNGKey(cfg.seed)
    key_init, key_train = jax.random.split(master_key)
    
    state = create_train_state(key_init, cfg.learning_rate, cfg.grid_size)
    
    # 2. Initialize Centralized Data Generator
    data_gen = DataGenerator(grid_size=cfg.grid_size, evolve_steps=cfg.evolve_steps)
    batched_simulator = jax.vmap(data_gen.sample_batch) # Vectorize the method
    
    # 3. Loop
    key_current = key_train
    for step in range(cfg.n_samples):
        key_current, key_sim = jax.random.split(key_current)
        sim_keys = jax.random.split(key_sim, cfg.batch_size)
        
        # Generate data
        batch_theta, batch_x = batched_simulator(sim_keys)
        
        # Train (Notice: removed key_shuffle)
        state, loss = train_step(state, batch_x, batch_theta)
        
        if step % 10 == 0:
            print(f"Step {step:04d} | Loss: {loss:.4f}")

    print("\n✅ Training Complete.")
    
   # --- Inference Demo ---
    print("\n🔍 Running Inference Demo...")
    import matplotlib.pyplot as plt
    import numpy as np

    # Ground Truth Setup
    true_eta = 0.8
    true_B = 0.01
    key_infer, key_sim = jax.random.split(jax.random.PRNGKey(999))
    
    print(f"Generating ground truth (eta={true_eta}, Grid={cfg.grid_size})...")
    
    config = SimConfig(eta=true_eta, B=true_B, N=cfg.grid_size)
    solver = GLSolverJAX(config)
    psi1, psi2 = GLSolverJAX.initialize_state(config, key_sim)
    
    p1_f, p2_f = solver.evolve(psi1, psi2, cfg.evolve_steps)
    
    obs_img = jnp.stack([jnp.abs(p1_f)**2, jnp.abs(p2_f)**2], axis=-1)
    obs_img_batch = jnp.expand_dims(obs_img, axis=0)

    # 1D Parameter Scan
    print("Scanning parameter space...")
    test_etas = jnp.linspace(0.0, 1.5, 100)
    scores = []
    
    for test_eta in test_etas:
        theta_test = jnp.array([[test_eta, true_B]])
        logit = state.apply_fn({'params': state.params}, obs_img_batch, theta_test)
        prob = jax.nn.sigmoid(logit)
        scores.append(prob[0, 0])
    
    # Visualization
    scores = np.array(scores)
    plt.figure(figsize=(10, 6))
    plt.plot(test_etas, scores, label='Posterior Proxy (AI)', color='blue', linewidth=2)
    plt.axvline(x=true_eta, color='red', linestyle='--', label=f'Ground Truth ({true_eta})')
    plt.title(f"NRE Inference (Grid {cfg.grid_size}x{cfg.grid_size})", fontsize=14)
    plt.xlabel("Coupling Strength $\eta$", fontsize=12)
    plt.ylabel("Ratio Score", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("inference_result.png")
    print("✅ Result saved to 'inference_result.png'")