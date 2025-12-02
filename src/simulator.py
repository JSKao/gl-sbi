# src/simulator.py
import jax
import jax.numpy as jnp
from src.gl_jax import GLSolverJAX, SimConfig

from src.sim_config import (
    ETA_MIN, ETA_MAX,
    B_MIN, B_MAX,
    NU_MIN, NU_MAX,
    GRID_SIZE, EVOLVE_STEPS
)

class DataGenerator:
    """
    Centralized data factory for creating (parameter, observation) pairs.
    """
    
    def __init__(self, grid_size=GRID_SIZE, evolve_steps=EVOLVE_STEPS):
        self.grid_size = grid_size
        self.evolve_steps = evolve_steps    

    def sample_batch(self, key):
        """
        Generates a single training sample for Model Selection task.
        Returns: 
            params: [eta, B, nu]
            label: 0.0 (Model A) or 1.0 (Model B)
            x: (H, W, 3) density + magnetic field map
        """
        # Split keys
        k_eta, k_B, k_nu, k_label, k_sim = jax.random.split(key, 5)
        
        # Sample Physics Parameters
        # Basic parameters (Always active)
        eta = jax.random.uniform(k_eta, minval=ETA_MIN, maxval=ETA_MAX)
        B   = jax.random.uniform(k_B,   minval=B_MIN,   maxval=B_MAX)
        # Drag parameter candidate (Only active if label == 1)
        raw_nu = jax.random.uniform(k_nu, minval=NU_MIN, maxval=NU_MAX)
        
        # Sample Label (Coin Flip)
        # Generate a float 0.0 or 1.0 directly
        label = jax.random.bernoulli(k_label, p=0.5).astype(jnp.float32)
        
        # Apply Logic Switch (The Mathematical Switch)
        # If label is 1, use raw_nu; if label is 0, force nu to 0.0
        nu = jnp.where(label > 0.5, raw_nu, 0.0)
        
        # ---- Simulation ----
        # Initialize Config with the specific nu
        config = SimConfig(eta=eta, B=B, nu=nu, N=self.grid_size)
        solver = GLSolverJAX(config)
        
        psi1_init, psi2_init = GLSolverJAX.initialize_state(config, k_sim)
        psi1_final, psi2_final = solver.evolve(psi1_init, psi2_init, self.evolve_steps)
        
        # Observation Engineering
        rho1 = jnp.abs(psi1_final) ** 2
        rho2 = jnp.abs(psi2_final) ** 2
        
        # Calculate magnetic features (Curl J)
        Jx, Jy = GLSolverJAX.compute_current(psi1_final, config)
        curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config)
        
        # Stack channels: (Density1, Density2, Curl_J)
        x = jnp.stack([rho1, rho2, curl_J], axis=-1)
        
        # --- Pack and Return ---
        # We include nu in params so we can verify the ground truth later if needed
        params = jnp.array([eta, B, nu])
        
        # Structure: ((Physics Params, Class Label), Image)
        # This structure allows flexible unpacking in the training loop
        return (params, label), x