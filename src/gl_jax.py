import jax
import jax.numpy as jnp
import numpy as np 
import time
from dataclasses import dataclass

# Enable 64-bit precision for numerical stability in physical simulations
jax.config.update("jax_enable_x64", True)


@dataclass
class SimConfig:
    """
    Configuration dataclass for Multicomponent Ginzburg-Landau simulations.
    
    Physical Parameters:
        eta (float): Intercomponent Josephson coupling strength.
        B (float): External magnetic field strength (flux per plaquette).
        alpha, beta (float): Ginzburg-Landau phenomenological parameters.
        D (float): Diffusion coefficient (related to relaxation time).
    
    Grid Parameters:
        N (int): Lattice size (N x N).
        L (float): Physical system size.
        dt (float): Time step for integration.
    """
    # Grid and Time
    N: int = 128           
    L: float = 64.0        
    dt: float = 0.01       
    steps_per_frame: int = 20

    # Global parameters
    eta: float = 0.8       
    B: float = 0.005       

    # Band 1 Parameters (Type-I like behavior)
    alpha1: float = -1.0
    beta1: float = 1.0
    D1: float = 4.0        

    # Band 2 Parameters (Type-II like behavior)
    alpha2: float = -1.0
    beta2: float = 1.0
    D2: float = 1.0

    @property
    def dx(self): 
        return self.L / self.N


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
        self.cfg = config  # 儲存配置，隨時可用

    # --- (Physics Kernel) ---
    @staticmethod 
    @jax.jit      
    def laplacian(field, dx, B):
        """
        Computes the discrete gauge-covariant Laplacian: (nabla - i*A)^2 * psi.
        Uses Peierls substitution U_ij = exp(-i * integral(A dl)) to maintain gauge invariance.
        """
        
        rows, cols = field.shape
        j_indices = jnp.arange(cols)
        
        # Peierls phase factor for the Landau gauge A = (0, Bx, 0)
        flux_per_cell = B * (dx**2)
        uy = jnp.exp(-1j * flux_per_cell * j_indices)
        
        # Broadcast uy to (rows, cols)
        U = jnp.tile(uy, (rows, 1))
        
        # Shift and apply phase factors
        # Axis 0 (y-direction): Apply Peierls phase U
        ip1 = jnp.roll(field, shift=-1, axis=0) * U           # psi(y+1)
        im1 = jnp.roll(field, shift=1, axis=0) * jnp.conj(U)  # psi(y-1)
        
        # Axis 1 (x-direction): No A_x component in Landau gauge
        jp1 = jnp.roll(field, shift=-1, axis=1) 
        jm1 = jnp.roll(field, shift=1, axis=1) 
        
        center = field
        
        # (Discrete Laplacian)
        # (up + down + left + right - 4*center) / dx^2
        lap = (ip1 + im1 + jp1 + jm1 - 4 * center) / (dx**2)
        
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
    def interaction_force(psi_self, psi_other, eta):
        """
        Calculates the Josephson interband coupling force:
        F_int = eta * psi_other
        """
        return eta * psi_other
    
    @staticmethod
    @jax.jit
    def compute_current(psi, dx, B):
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
        rows, cols = psi.shape
        j_indices = jnp.arange(cols)
        # Peierls phase factor for y-hopping in Landau gauge A = (0, Bx, 0)
        flux = B * (dx**2)
        uy = jnp.exp(-1j * flux * j_indices)
        U = jnp.tile(uy, (rows, 1))
        
        # J_y component (Vertical links)
        # Connection involves phase U: psi(y) -> psi(y+1)
        ip1 = jnp.roll(psi, shift=-1, axis=0)
        J_y = jnp.imag(jnp.conj(psi) * U * ip1) / dx
        
        # J_x component (Horizontal links)
        # In Landau gauge A_x = 0, so the link variable is unity
        jp1 = jnp.roll(psi, shift=-1, axis=1)
        J_x = jnp.imag(jnp.conj(psi) * jp1) / dx
        
        return J_x, J_y
    
    @staticmethod
    @jax.jit
    def compute_curl_J(Jx, Jy, dx):
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
        dJy_dx = (jp1_y - Jy) / dx
        
        # Calculate dJx/dy using forward difference
        # Shift -1 on axis 0 corresponds to y -> y+1
        ip1_x = jnp.roll(Jx, shift=-1, axis=0)
        dJx_dy = (ip1_x - Jx) / dx
        
        # Curl_z definition
        curl_z = dJy_dx - dJx_dy
        
        return curl_z
    
    @staticmethod
    @jax.jit
    def update_step(psi1, psi2, D1, D2, alpha1, beta1, alpha2, beta2, eta, dx, dt, B):
        """
        Performs a single Euler integration step for the coupled TDGL equations.
        """
        # 1. Kinetic term (Covariant Laplacian)
        kin1 = D1 * GLSolverJAX.laplacian(psi1, dx, B)
        kin2 = D2 * GLSolverJAX.laplacian(psi2, dx, B)

        # 2. Potential term (Condensation energy)
        pot1 = GLSolverJAX.potential_force(psi1, alpha1, beta1)
        pot2 = GLSolverJAX.potential_force(psi2, alpha2, beta2)

        # 3. Interaction term (Josephson coupling)
        coup1 = GLSolverJAX.interaction_force(psi1, psi2, eta)
        coup2 = GLSolverJAX.interaction_force(psi2, psi1, eta)

        # Time integration
        new_psi1 = psi1 + dt * (kin1 + pot1 + coup1)
        new_psi2 = psi2 + dt * (kin2 + pot2 + coup2)

        return new_psi1, new_psi2

    
    def evolve(self, psi1, psi2, steps):
        """
        Evolves the system for a fixed number of steps using jax.lax.scan.
        This compiles the temporal loop into a single XLA kernel for efficiency.
        """
        c = self.cfg  
        params = (c.D1, c.D2, c.alpha1, c.beta1, 
                  c.alpha2, c.beta2, c.eta, c.dx, c.dt, c.B)
        
        def body_fun(carry, _):
            p1, p2 = carry
            new_p1, new_p2 = self.update_step(p1, p2, *params)
            return (new_p1, new_p2), None 
        
        init_carry = (psi1, psi2)
        final_state, _ = jax.lax.scan(body_fun, init_carry, jnp.arange(steps))
    
        return final_state    
        

# ---test---
if __name__ == "__main__":
    test_config = SimConfig(N=64, B=0.05, D1=2.0)
    
    sim = GLSolverJAX(test_config)
    
    key = jax.random.PRNGKey(42)
    
    psi1, psi2 = GLSolverJAX.initialize_state(test_config, key)
    
    print(f"Running simulation with B={sim.cfg.B}, D1={sim.cfg.D1}")
    
    final_psi1, final_psi2 = sim.evolve(psi1, psi2, steps=100)
    
    print(f"Done. Final shape: {final_psi1.shape}")