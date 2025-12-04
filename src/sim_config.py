# src/sim_config.py

"""
Global Configuration for GL-SBI Project.
Defines physics priors, simulation defaults, and hardware settings.
"""

# ---  Parameter Priors (Data Generation) ---
# Ranges for uniform sampling
ETA_MIN = 0.4
ETA_MAX = 1.2

B_MIN = 0.04
B_MAX = 0.10

NU_MIN = 0.05
NU_MAX = 0.25  # For model selection task

# ---  Simulation Defaults (The "Factory Settings") ---
# Grid & Time
GRID_SIZE = 64
L_SIZE = 32.0
DT = 0.002
STEPS_PER_FRAME = 20
EVOLVE_STEPS = 5000

# Physics Constants (Band 1)
ALPHA1 = -1.0
BETA1  = 1.0
D1     = 4.0

# Physics Constants (Band 2)
ALPHA2 = -1.0
BETA2  = 1.0
D2     = 1.0

# Default Global Params (for single run)
DEFAULT_ETA = 0.8
DEFAULT_B   = 0.01
DEFAULT_NU  = 0.0

# Data Settings
N_SAMPLES = 10000
BATCH_SIZE = 16

# Model Settings
CHANNELS = 3  # [rho1, rho2, curl_J]

# --- Test Cases for Paper Figures ---
TEST_CASE_ETA = 0.8
TEST_CASE_B   = 0.04
TEST_CASE_NU  = 0.0


# Evaluation / Test Configs
# 這些是 run_comprehensive_tests.py 需要的常數

# --- Multiple test cases for validation ---
# for Fig. 3B 
TEST_ETAS = [0.2, 0.5, 0.8, 1.1, 1.4] 
# for 1D slice
TEST_B_FIXED = 0.05   

 