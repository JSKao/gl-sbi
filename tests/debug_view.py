import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from src.gl_jax import GLSolverJAX, SimConfig

def main():
    print("🔍 Debugging Physics Features...")
    
    #  Set parameters same as ground truth 
    eta = 0.8
    B = 0.015  # the same value in the inference
    N = 128
    
    config = SimConfig(eta=eta, B=B, N=N)
    solver = GLSolverJAX(config)
    
    #  simulate
    key = jax.random.PRNGKey(42)
    print("   Running Simulation...")
    psi1, psi2 = GLSolverJAX.initialize_state(config, key)
    psi1, psi2 = solver.evolve(psi1, psi2, 1000) # 跑久一點確保有東西
    
    #  Compute character
    rho1 = jnp.abs(psi1)**2
    rho2 = jnp.abs(psi2)**2
    
    # Compute B-field character
    Jx, Jy = GLSolverJAX.compute_current(psi1, config.dx, config.B)
    curl_J = GLSolverJAX.compute_curl_J(Jx, Jy, config.dx)
    
    print(f"   Stats | Density Max: {rho1.max():.2f} | Curl Max: {curl_J.max():.2f}")
    
    # plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Channel 0: Density 1
    im0 = axs[0].imshow(rho1, origin='lower', cmap='inferno')
    axs[0].set_title(f"Ch 0: Density 1 (eta={eta})")
    plt.colorbar(im0, ax=axs[0])
    
    # Channel 1: Density 2
    im1 = axs[1].imshow(rho2, origin='lower', cmap='inferno')
    axs[1].set_title("Ch 1: Density 2")
    plt.colorbar(im1, ax=axs[1])
    
    # Channel 2: Curl J (Magnetic Field) -> important
    # Use seismic colormap in case curl might have positive and negative sign
    im2 = axs[2].imshow(curl_J, origin='lower', cmap='seismic')
    axs[2].set_title(f"Ch 2: Curl J (B={B})")
    plt.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    plt.savefig("debug_features.png")
    print("✅ Saved debug image to 'debug_features.png'. Go check it!")

if __name__ == "__main__":
    main()