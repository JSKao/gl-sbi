import jax
import jax.numpy as jnp
from src.gl_jax import GLSolverJAX, SimConfig

'''
flat_params (1D array, length 4N^2) 
[ ------------------ | ------------------ | ------------------ | ------------------ ]
   1st: RE[psi1]      2nd: Im[psi1]         3rd: Re[psi2]          4th: Im[psi2]
   (length N^2)           (length N^2)         (length N^2)         (length N^2)
'''   

def energy_wrapper(flat_params, config):
    
    N = config.N
    size = N * N
    # Slicing
    p1_r = flat_params[0 : size]
    p1_i = flat_params[size : 2*size]
    p2_r = flat_params[ 2*size : 3*size ]
    p2_i = flat_params[ 3*size : 4*size ]

    # Recombine & Reshape
    psi1 = (p1_r + 1j * p1_i).reshape((N, N))
    psi2 = (p2_r + 1j * p2_i).reshape((N, N)) 
    
    # 4. Call Physics Engine
    return GLSolverJAX.compute_free_energy(psi1, psi2, config) 

def compute_hessian_eig(psi1, psi2, config):
    print(f"Computing Hessian for Grid N={config.N}...")
    
    # Packing 
    flat_params = jnp.concatenate([
        jnp.real(psi1).ravel(),
        jnp.imag(psi1).ravel(),
        jnp.real(psi2).ravel(),
        jnp.imag (psi2).ravel()
    ])


    # Define Target Function
    # Create a function only see p as variables, and keep config fixed
    loss_fn = lambda p: energy_wrapper(p, config)
    
    # Compute Hessian
    # jax.hessian returns a function, we call it immediately with flat_params
    print("  JIT Compiling Hessian function... (this may take a moment)")
    H_fn = jax.jit(jax.hessian(loss_fn))
    H = H_fn(flat_params)
    
    print(f"  Hessian Computed. Shape: {H.shape}. Eigenodecomposition starting...")
 
    # Compute Eigenvalues
    # jnp.linalg.eigvalsh is for symmetric matrices (faster & more stable)
    eigvals = jnp.linalg.eigvalsh(H)
    
    return eigvals