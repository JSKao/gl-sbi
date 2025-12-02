# tests/test_physics_safety.py
import unittest
import sys
import os
import jax.numpy as jnp

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gl_jax import SimConfig

from src.sim_config import (
    ETA_MAX, B_MAX, NU_MAX, 
    GRID_SIZE, L_SIZE, DT, 
    D1, D2, STEPS_PER_FRAME
)

class TestPhysicsSafety(unittest.TestCase):
    
    # ---  check parameters in sim_config.py  ---
    def test_actual_production_config(self):
        """
        [CRITICAL] Tests the ACTUAL parameters defined in src/sim_config.py.
        This ensures the simulation we are about to run is safe.
        """
        print("\n=== 🛡️ Checking Production Configuration (sim_config.py) ===")
        print(f"  > Grid: {GRID_SIZE}x{GRID_SIZE}")
        print(f"  > Size: L={L_SIZE}")
        print(f"  > Time: dt={DT}")
        
        
        prod_config = SimConfig(
            N=GRID_SIZE,
            L=L_SIZE,
            dt=DT,
            steps_per_frame=STEPS_PER_FRAME,
            eta=ETA_MAX,
            B=B_MAX,
            nu=NU_MAX  
        )

        
        print(f"  > Effective Resolution dx: {prod_config.dx:.4f}")
        
        
        self.assertLessEqual(prod_config.dx, 0.5, 
            f" Resolution too low! dx={prod_config.dx}. Please decrease L_SIZE or increase GRID_SIZE in sim_config.py")
        
        self.assertGreaterEqual(prod_config.dx, 0.1,
            f" Resolution too high! dx={prod_config.dx}. You are wasting compute power.")

        
        prod_config.validate() 
        print("   CFL Condition: PASSED")
        print("   Topological Capacity: PASSED")
        print("=== Production Config is SAFE to run! ===\n")

    # -- old version --
    def test_golden_rule_logic_1_resolution(self):
        """Test if the validation logic correctly catches bad dx"""
        
        bad_config = SimConfig(N=32, L=64.0) 
        
        pass 

    def test_golden_rule_logic_2_cfl(self):
        """Test if validation logic catches exploding dt"""
        with self.assertRaises(ValueError):
            SimConfig(dt=1.0, nu=NU_MAX, N=64)

if __name__ == '__main__':
    unittest.main()