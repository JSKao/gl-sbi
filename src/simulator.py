import jax
import jax.numpy as jnp
from src.gl_jax import GLSolverJAX, SimConfig

class DataGenerator:
    """
    Centralized data factory for creating (parameter, observation) pairs.
    This ensures consistency between offline generation and online training.
    """
    
    def __init__(self, grid_size=32, evolve_steps=100):
        self.grid_size = grid_size
        self.evolve_steps = evolve_steps

    def sample_batch(self, key):
        """
        Single sample generation function to be vmapped.
        Returns: (theta, x)
        """
        key_eta, key_B, key_sim = jax.random.split(key, 3)
        
        # 1. Prior Sampling
        eta = jax.random.uniform(key_eta, minval=0.0, maxval=1.5)
        B = jax.random.uniform(key_B, minval=0.0, maxval=0.02)
        
        # 2. Simulation
        config = SimConfig(eta=eta, B=B, N=self.grid_size)
        solver = GLSolverJAX(config)
        
        psi1_init, psi2_init = GLSolverJAX.initialize_state(config, key_sim)
        psi1_final, psi2_final = solver.evolve(psi1_init, psi2_init, self.evolve_steps)
        
        # 3. Observation (Density)
        rho1 = jnp.abs(psi1_final) ** 2
        rho2 = jnp.abs(psi2_final) ** 2
        
        # Stack channels last: (H, W, C)
        density = jnp.stack([rho1, rho2], axis=-1)
        
        params = jnp.array([eta, B])
        return params, density