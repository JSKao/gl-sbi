"""
experimental_noise.py

This module implements the "synthetic-to-real" transformation pipelines described
in Section VI.E of the paper. It applies instrumental noise and resolution 
constraints to ideal simulation data to bridge the gap between TDGL simulations 
and experimental measurements (STM, Lorentz TEM, Neutron Scattering).

References:
    - STM: Convolve with Gaussian kernel + Poisson noise.
    - TEM: Integrated magnetic flux + depth-of-field blur.
    - S(k): Instrumental resolution function in k-space.
"""

import numpy as np
import scipy.ndimage as ndimage

class ExperimentalTransform:
    def __init__(self, pixel_size_nm=1.0):
        """
        Args:
            pixel_size_nm (float): The physical size of one simulation pixel in nanometers.
                                   Used to calibrate resolution kernels.
        """
        self.pixel_size_nm = pixel_size_nm

    def simulate_stm(self, density_field, tip_resolution_nm=1.0, noise_level=0.05):
        """
        Simulates an STM (Scanning Tunneling Microscopy) measurement.
        
        Physics:
            1. Convolves the local density of states (LDOS ~ |psi|^2) with a Gaussian
               kernel representing the STM tip point-spread function.
            2. Adds Poisson (shot) noise to simulate photon/electron counting statistics.
        
        Args:
            density_field (np.ndarray): 2D array of superconductor density |psi|^2.
            tip_resolution_nm (float): Sigma of the Gaussian tip convolution (approx. resolution).
            noise_level (float): Scale factor for the Poisson noise intensity.
            
        Returns:
            np.ndarray: The "measured" STM image.
        """
        # 1. Gaussian Convolution (Tip Geometry)
        sigma_pixels = tip_resolution_nm / self.pixel_size_nm
        blurred_field = ndimage.gaussian_filter(density_field, sigma=sigma_pixels)
        
        # 2. Poisson Noise (Measurement Statistics)
        # We normalize to a "count" range to apply poisson, then scale back
        # This is a heuristic approximation for generic signal-to-noise control
        if noise_level > 0:
            # Avoid negative values for Poisson
            safe_field = np.maximum(blurred_field, 0)
            # Scale up to simulate "counts", add noise, scale down
            scale = 1.0 / noise_level
            noisy_field = np.random.poisson(safe_field * scale) / scale
            return noisy_field
        
        return blurred_field

    def simulate_lorentz_tem(self, b_field_z, film_thickness_nm=50.0, blur_sigma=0.5):
        """
        Simulates a Lorentz TEM (Transmission Electron Microscopy) measurement.
        
        Physics:
            1. Integrates the magnetic flux along the z-axis (thickness).
            2. Applies a slight blur to account for finite depth-of-field or lens aberration.
        
        Args:
            b_field_z (np.ndarray): 2D array of local magnetic field (curl J).
            film_thickness_nm (float): Thickness of the sample.
            blur_sigma (float): Instrumental blurring factor (pixels).
            
        Returns:
            np.ndarray: The phase shift or integrated flux map.
        """
        # 1. Integration (Projection)
        # In a 2D simulation, this is effectively a scalar multiplication 
        # assuming the vortex is cylindrical through the film.
        projected_flux = b_field_z * film_thickness_nm
        
        # 2. Instrumental Blur
        if blur_sigma > 0:
            return ndimage.gaussian_filter(projected_flux, sigma=blur_sigma)
        
        return projected_flux

    def apply_sk_resolution(self, structure_factor, k_resolution_percent=0.05):
        """
        Applies instrumental resolution to a calculated Structure Factor S(k).
        
        Physics:
            Neutron scattering instruments have a finite k-space resolution,
            often modeled as a Gaussian convolution in radial k-space.
            
        Args:
            structure_factor (np.ndarray): 1D array of S(k).
            k_resolution_percent (float): Width of resolution function (Delta k / k).
            
        Returns:
            np.ndarray: Broadened S(k).
        """
        # Simple approximation: apply a varying Gaussian blur that grows with k
        # or a constant blur if k-dependence is negligible for the range.
        # Here we use a constant small blur for demonstration.
        
        # Assuming input is a 1D radial average
        sigma = len(structure_factor) * k_resolution_percent
        return ndimage.gaussian_filter(structure_factor, sigma=sigma)

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy synthetic data (64x64 grid)
    N = 64
    x = np.linspace(-3, 3, N)
    y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(x, y)
    
    # Generate a fake "vortex" (Gaussian dip)
    r = np.sqrt(X**2 + Y**2)
    psi_squared = 1.0 - np.exp(-r**2)
    b_field = np.exp(-r**2) # Vortex core carries flux

    # Initialize transformer
    transformer = ExperimentalTransform(pixel_size_nm=2.0)

    # 1. Generate STM View
    stm_view = transformer.simulate_stm(psi_squared, tip_resolution_nm=1.0, noise_level=0.02)
    print(f"Generated STM view with shape {stm_view.shape} and noise stats.")

    # 2. Generate TEM View
    tem_view = transformer.simulate_lorentz_tem(b_field, film_thickness_nm=100.0)
    print(f"Generated TEM view with max flux: {tem_view.max():.4f}")

    print("Success: experimental_noise.py is running correctly.")