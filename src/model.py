import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class CNNEncoder(nn.Module):
    """
    Convolutional Encoder for extracting geometric features from physical lattice systems.
    Utilizes Global Average Pooling (GAP) to enforce translational invariance.
    """
    features: Sequence[int] = (32, 64, 64)
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        # Feature extraction
        for feat in self.features:
            x = nn.Conv(features=feat, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            
        # Global Average Pooling: (B, H, W, C) -> (B, C)
        x = jnp.mean(x, axis=(1, 2))
        
        # Projection
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.relu(x)
        return x

class ParameterEmbedding(nn.Module):
    """
    High-dimensional embedding for scalar physical parameters.
    """
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.relu(x)
        return x

class NREClassifier(nn.Module):
    """
    Neural Ratio Estimator (NRE) architecture.
    Fuses visual observables with hypothesis parameters to estimate likelihood ratios.
    """
    
    @nn.compact
    def __call__(self, x, theta):
        # Left Tower: Visual features
        h_img = CNNEncoder(output_dim=64)(x)
        
        # Right Tower: Parameter features
        h_param = ParameterEmbedding(output_dim=64)(theta)
        
        # Fusion
        h_joint = jnp.concatenate([h_img, h_param], axis=-1)
        
        # Decision Head
        h = nn.Dense(features=64)(h_joint)
        h = nn.relu(h)
        h = nn.Dense(features=64)(h)
        h = nn.relu(h)
        
        # Logit output (Linear)
        logit = nn.Dense(features=1)(h)
        return logit

if __name__ == "__main__":
    # Architecture Verification
    dummy_image = jnp.ones((5, 128, 128, 2))
    dummy_theta = jnp.ones((5, 2))
    
    model = NREClassifier()
    key = jax.random.PRNGKey(0)
    variables = model.init(key, dummy_image, dummy_theta)
    
    print(f"Model Initialized. Parameters: {list(variables.keys())}")
    
    output = model.apply(variables, dummy_image, dummy_theta)
    assert output.shape == (5, 1)
    print("✅ Architecture Check Passed.")