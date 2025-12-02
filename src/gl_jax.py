# src/gl_jax.py
import jax
import jax.numpy as jnp
import numpy as np 
import time
from dataclasses import dataclass
from functools import partial
from jax.tree_util import register_pytree_node

# Import all defaults from the single source of truth
from src.sim_config import (
    GRID_SIZE, L_SIZE, DT, STEPS_PER_FRAME,
    DEFAULT_ETA, DEFAULT_B, DEFAULT_NU,
    ALPHA1, BETA1, D1,
    ALPHA2, BETA2, D2
)


# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)


@dataclass
class SimConfig:
    """
    Immutable configuration object for JAX kernels.
    Defaults are pulled from src.sim_config.py
    """
    # Grid and Time (Static / Aux data)
    N: int = GRID_SIZE           
    L: float = L_SIZE        
    dt: float = DT
    steps_per_frame: int = STEPS_PER_FRAME

    # Physics Parameters (Dynamic / Children)
    eta: float = DEFAULT_ETA       
    B: float = DEFAULT_B
    nu: float = DEFAULT_NU      
    
    # Band 1 Parameters (Dynamic)
    alpha1: float = ALPHA1
    beta1: float = BETA1
    D1: float = D1        

    # Band 2 Parameters (Dynamic)
    alpha2: float = ALPHA2
    beta2: float = BETA2
    D2: float = D2

    @property
    def dx(self): 
        return self.L / self.N
    
    def compute_cfl_limit(self):
        """Compute CFL limit dt for diffusion equation"""
        # Find max between D1 and D2
        D_max = jnp.maximum(self.D1 + jnp.abs(self.nu), self.D2 + jnp.abs(self.nu))
    
        # Courant-Friedrichs-Lewy condition
        dt_limit = self.dx**2 / 4 / D_max   # dx^2 / (4 * D_max)
        return dt_limit

    def validate(self):
        """
        Set constraints to parameters for safety GL simulation.
        Only runs on concrete values (Python floats), skips JAX Tracers.
        """
        # Tracer Check
        if not isinstance(self.nu, (float, int, np.floating, np.integer)):
            return

        # --- 1: Resolution Constraint  ---
        # dx must be small enough to resolve vortices (coherence length ~ 1)
        if self.dx > 0.5:
            print(f" [Warning] Low Resolution: dx={self.dx:.3f} (Recommended <= 0.5). Vortices may look pixelated.")
        elif self.dx < 0.1:
            print(f" [Warning] Super High Resolution: dx={self.dx:.3f}. Compute might be wasteful.")

        # --- 2: Courant-Friedrichs-Lewy condition ---
        # Rule for numerical stability
        dt_limit = float(self.compute_cfl_limit()) # Force conversion to float
        safety_factor = 0.9  
        if self.dt > dt_limit * safety_factor:
            raise ValueError(
                f"  [CFL Violation] Time step dt={self.dt} is unsafe!\n"
                f"   Max allowed: {dt_limit:.5f} (with safety factor {safety_factor})\n"
                f"   Solution: Reduce DT or Increase dx (L/N)."
            )

        # --- 3: Topological Capacity ---
        # Check flux quantization)
        # Phi = B * L^2. One vortex quantum = 2*pi
        total_flux = self.B * (self.L ** 2)
        if total_flux < 2 * np.pi:
             # Only warn if B is not intentionally zero
             if self.B > 1e-9:
                 print(f"[Warning] Low Magnetic Flux: Phi={total_flux:.2f}. Box might be too small to hold even one vortex.")
    def __post_init__(self):      
        self.validate()


# --- JAX Pytree Registration ---

def _sim_config_flatten(config):
    # Dynamic children (can be traced/batched)
    children = (
        config.eta, config.B, config.nu,
        config.alpha1, config.beta1, config.D1,
        config.alpha2, config.beta2, config.D2
    )
    # Static auxiliary data (integers, grid size, etc.)
    aux_data = (
        config.N, config.L, config.dt, config.steps_per_frame
    )
    return children, aux_data

def _sim_config_unflatten(aux_data, children):
    N, L, dt, steps_per_frame = aux_data
    eta, B, nu, a1, b1, d1, a2, b2, d2 = children
    
    return SimConfig(
        N=N, L=L, dt=dt, steps_per_frame=steps_per_frame,
        eta=eta, B=B, nu=nu,
        alpha1=a1, beta1=b1, D1=d1,
        alpha2=a2, beta2=b2, D2=d2
    )

register_pytree_node(SimConfig, _sim_config_flatten, _sim_config_unflatten)


class GLSolverJAX:
    """
    Stateless, JAX-accelerated solver for Time-Dependent Ginzburg-Landau (TDGL) equations.
    Implements a finite-difference scheme with Peierls substitution for gauge invariance.
    """
    
    @staticmethod
    def initialize_state(config, key):
        """
        Generates a random initial state representing a high-temperature quench.
        
        Args:
            config: SimConfig instance.
            key: JAX PRNGKey.
            
        Returns:
            psi1, psi2: Randomized complex order parameters.
        """
        
        # split into 4 keys random arrays: psi1_real, psi1_imag, psi2_real, psi2_imag
        k1, k2, k3, k4 = jax.random.split(key, 4)
        
        # Initialize with uniform random noise in [-0.5, 0.5] for both real and imag parts
        r1 = jax.random.uniform(k1, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        i1 = jax.random.uniform(k2, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        psi1 = r1 + 1j * i1
        
        r2 = jax.random.uniform(k3, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        i2 = jax.random.uniform(k4, shape=(config.N, config.N), minval=-0.5, maxval=0.5)
        psi2 = r2 + 1j * i2
        
        return psi1, psi2

    
    def __init__(self, config: SimConfig):
        self.cfg = config 

    # --- (Physics Kernel) ---
    @staticmethod 
    def compute_peierls_phase(psi, config): 
        
        rows, cols = psi.shape
        j_indices = jnp.arange(cols)
        
        # Peierls phase factor for the Landau gauge A = (0, Bx, 0)
        flux_per_cell = config.B * (config.dx ** 2)
        uy = jnp.exp(-1j * flux_per_cell * j_indices) 
        
        # Broadcast uy to (rows, cols)
        U = jnp.tile(uy, (rows, 1))
        
        return U
    
    
    @staticmethod 
    @jax.jit
    def laplacian(psi, config):
        """
        Computes the discrete gauge-covariant Laplacian: (nabla - i*A)^2 * psi.
        Uses Peierls substitution U_ij = exp(-i * integral(A dl)) to maintain gauge invariance.
        """
        U = GLSolverJAX.compute_peierls_phase(psi, config)
        
        # Shift and apply phase factors
        # Axis 0 (y-direction): Apply Peierls phase U
        ip1 = jnp.roll(psi, shift=-1, axis=0) * U           # psi(y+1)
        im1 = jnp.roll(psi, shift=1, axis=0) * jnp.conj(U)  # psi(y-1)
        
        # Axis 1 (x-direction): No A_x component in Landau gauge
        jp1 = jnp.roll(psi, shift=-1, axis=1) 
        jm1 = jnp.roll(psi, shift=1, axis=1) 
        
        center = psi
        
        # (Discrete Laplacian)
        # (up + down + left + right - 4*center) / dx^2
        lap = (ip1 + im1 + jp1 + jm1 - 4 * center) / (config.dx**2)
        
        return lap
    
    @staticmethod
    @jax.jit
    def potential_force(psi, alpha, beta):
        """
        Calculates the force derived from the local Ginzburg-Landau potential:
        F_pot = - dV/d(psi*) = -(alpha*psi + beta*|psi|^2*psi)
        """
        density = jnp.abs(psi)**2
        force = - (alpha * psi + beta * density * psi) 
        return force
    
    @staticmethod
    @jax.jit
    def interaction_force(psi_other, config):
        """
        Calculates the Josephson interband coupling force:
        F_int = eta * psi_other
        """
        return config.eta * psi_other
    
    @staticmethod
    def compute_free_energy(psi1, psi2, config):
        """
        Calculates the total Ginzburg-Landau free energy.
        """
        # --- 1. Kinetic Energy (with Peierls Phase) ---
        # Use forward difference (|D_x|^2 + |D_y|^2)
        
        # Peierls phase U
        U1 = GLSolverJAX.compute_peierls_phase(psi1, config)
        U2 = GLSolverJAX.compute_peierls_phase(psi2, config)

        def kinetic_density(psi, U):
            # X-direction (No phase)
            d_psi_x = (jnp.roll(psi, shift=-1, axis=1) - psi) / config.dx
            # Y-direction (With phase U)
            # D_y psi ~ (U * psi(y+1) - psi(y)) / dx
            d_psi_y = (jnp.roll(psi, shift=-1, axis=0) * U - psi) / config.dx
            return jnp.abs(d_psi_x)**2 + jnp.abs(d_psi_y)**2

        f_kin1 = config.D1 * kinetic_density(psi1, U1) 
        f_kin2 = config.D2 * kinetic_density(psi2, U2)

        # --- 2. Potential Energy ---
        def potential_density(psi, alpha, beta):
            rho = jnp.abs(psi)**2
            return alpha * rho + (beta / 2) * rho**2

        f_pot1 = potential_density(psi1, config.alpha1, config.beta1)
        f_pot2 = potential_density(psi2, config.alpha2, config.beta2)

        # --- 3. Interaction Energy (Josephson) ---
        # -eta * (psi1 * psi2* + c.c.)
        f_int = -config.eta * (psi1 * jnp.conj(psi2) + jnp.conj(psi1) * psi2)
        f_int = jnp.real(f_int) # In case JAX pick imaginary part 0j

        # --- Total energy functional ---
        f_density = f_kin1 + f_kin2 + f_pot1 + f_pot2 + f_int
        return jnp.sum(f_density) * (config.dx**2)
    
    @staticmethod
    @jax.jit
    def compute_current(psi, config):
        """
        Calculates the gauge-invariant superconducting current density J.
        
        Formula: J_k ~ Im(psi* D_k psi), where D_k is the covariant derivative.
        We use Peierls phase factors to ensure gauge invariance on the discrete lattice.
        
        Args:
            psi: Complex order parameter field.
            dx: Grid spacing.
            B: Magnetic field strength.
            
        Returns:
            J_x, J_y: Current density components.
        """
        
        # Peierls phase U
        U = GLSolverJAX.compute_peierls_phase(psi, config)
        
        # J_y component (Vertical links)
        # Connection involves phase U: psi(y) -> psi(y+1)
        ip1 = jnp.roll(psi, shift=-1, axis=0)
        J_y = jnp.imag(jnp.conj(psi) * U * ip1) / config.dx
        
        # J_x component (Horizontal links)
        # In Landau gauge A_x = 0, so the link variable is unity
        jp1 = jnp.roll(psi, shift=-1, axis=1)
        J_x = jnp.imag(jnp.conj(psi) * jp1) / config.dx
        
        return J_x, J_y
    
    @staticmethod
    @jax.jit
    def compute_curl_J(Jx, Jy, config):
        """
        Computes the discrete curl of the supercurrent: (curl J)_z = dJy/dx - dJx/dy.
        
        Physical Significance:
            This quantity serves as a proxy for the local magnetic field distribution.
            It peaks sharply at vortex cores, making it excellent for visualization
            and vortex counting algorithms.
        """
        # Calculate dJy/dx using forward difference
        # Shift -1 on axis 1 corresponds to x -> x+1
        jp1_y = jnp.roll(Jy, shift=-1, axis=1)
        dJy_dx = (jp1_y - Jy) / config.dx
        
        # Calculate dJx/dy using forward difference
        # Shift -1 on axis 0 corresponds to y -> y+1
        ip1_x = jnp.roll(Jx, shift=-1, axis=0)
        dJx_dy = (ip1_x - Jx) / config.dx
        
        # Curl_z definition
        curl_z = dJy_dx - dJx_dy
        
        return curl_z
    
    @staticmethod
    @jax.jit
    def update_step(psi1, psi2, config):
        """
        Performs a single Euler integration step for the coupled TDGL equations.
        """
        # 1. Kinetic term (Covariant Laplacian)
        kin1 = config.D1 * GLSolverJAX.laplacian(psi1, config)
        kin2 = config.D2 * GLSolverJAX.laplacian(psi2, config)

        # 2. Potential term (Condensation energy)
        pot1 = GLSolverJAX.potential_force(psi1, config.alpha1, config.beta1)
        pot2 = GLSolverJAX.potential_force(psi2, config.alpha2, config.beta2)

        # 3. Interaction term (Josephson coupling)
        coup1 = GLSolverJAX.interaction_force(psi2, config)
        coup2 = GLSolverJAX.interaction_force(psi1, config)
        
        # 4. Drag term 
        drag1 = config.nu * GLSolverJAX.laplacian(psi2, config)
        drag2 = config.nu * GLSolverJAX.laplacian(psi1, config)

        # Time integration
        new_psi1 = psi1 + config.dt * (kin1 + pot1 + coup1 + drag1)
        new_psi2 = psi2 + config.dt * (kin2 + pot2 + coup2 + drag2)

        return new_psi1, new_psi2

    
    def evolve(self, psi1, psi2, steps):
        """
        Evolves the system for a fixed number of steps using jax.lax.scan.
        This compiles the temporal loop into a single XLA kernel for efficiency.
        """
        config = self.cfg  
        
        def body_fun(carry, _):
            p1, p2 = carry
            new_p1, new_p2 = GLSolverJAX.update_step(p1, p2, config)
            return (new_p1, new_p2), None 
        
        init_carry = (psi1, psi2)
        final_state, _ = jax.lax.scan(body_fun, init_carry, jnp.arange(steps))
    
        return final_state    
        

# ---test---
if __name__ == "__main__":
    test_config = SimConfig(N=64, B=0.01, D1=2.0)
    
    sim = GLSolverJAX(test_config)
    
    key = jax.random.PRNGKey(42)
    
    psi1, psi2 = GLSolverJAX.initialize_state(test_config, key)
    
    print(f"Running simulation with B={sim.cfg.B}, D1={sim.cfg.D1}")
    
    final_psi1, final_psi2 = sim.evolve(psi1, psi2, steps=100)
    
    print(f"Done. Final shape: {final_psi1.shape}")