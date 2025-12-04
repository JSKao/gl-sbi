# src/train_config.py

"""
Hyperparameters for the Training Loop.
Adjust these to optimize convergence speed and stability.
"""

# Optimization
BATCH_SIZE = 32
LEARNING_RATE = 2e-4  # Adam default
EPOCHS = 25           

# Data Management
VAL_SPLIT = 0.1       # 10% for validation
SEED = 42             # Reproducibility

# Checkpointing
CKPT_DIR = "checkpoints"
KEEP_CHECKPOINTS = 3  # Keep top 3 best models