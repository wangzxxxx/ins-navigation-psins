import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Connect parent module locally 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.shadow_kf import EnhancedKalmanFilter
from psins_py.shadow_manager import ShadowSimulationManager
from psins_py.nav_utils import glv

def main():
    print("Initializing LLM-Enhanced Shadow Observation Pipeline MOCK TEST...")

    # 1. Initialize Mock States
    # State Vector: [fi (3), dvn (3), eb (3), db (3), dK (3)] -> 15 dimensions
    initial_X = np.zeros(15)
    
    # Large Initial Covariance representing huge uncertainty
    initial_P = np.diag([
        0.1, 0.1, 0.1,      # fi
        1.0, 1.0, 1.0,      # dvn
        0.1*glv.dph, 0.1*glv.dph, 0.1*glv.dph,  # eb
        1.0*glv.mg, 1.0*glv.mg, 1.0*glv.mg,     # db
        100*glv.ppm, 100*glv.ppm, 100*glv.ppm   # dK
    ]) ** 2
    
    Q = np.eye(15) * 1e-8
    
    # H_speed represents zero-velocity updates in a 15-state EKF
    H_speed = np.zeros((3, 15))
    H_speed[0:3, 3:6] = np.eye(3) 

    R_speed = np.eye(3) * (0.01 ** 2)

    # 2. Build Extensible KF
    kf = EnhancedKalmanFilter(initial_X, initial_P, Q, H_speed, R_speed)
    
    # 3. Create Intelligent Manager 
    # Provide a tiny window size for testing to trigger the LLM call almost instantly
    hz = 100
    manager = ShadowSimulationManager(kf, window_size=200) # 2 seconds of high freq data
    
    # 4. Mock Loop Simulation
    length = 500 # Simulate 5 seconds 
    P_trace = []
    
    print("Starting Main Loop...")
    for step in range(length):
        # MOCK Physics: ZUPT (Velocity is roughly 0 with some noise)
        z_speed = np.random.randn(3) * 0.01
        
        # MOCK Physics: Identical Phi
        Phi = np.eye(15)
        
        # Simulate different stages to allow context building
        current_action = "Turntable X-Axis 90deg Spin" if step < 200 else "Stationary Level"
        
        # At step=200, window is full, we decide to authorize LLM invocation
        trigger_shadow = (step % 250 == 0 and step > 0)
        
        # Iteration Execution 
        manager.step(z_speed, Phi, current_action=current_action, trigger_shadow_eval=trigger_shadow)
        
        # Log Diag(P)
        P_trace.append(np.diag(kf.P).copy())
        
    P_trace = np.array(P_trace)

    # 5. Output Verification
    print("Simulation Complete. Checking Covariance Convergence...")
    
    plt.figure(figsize=(10, 6))
    # Plotting parameter index 12 (mocking X Gyro Scale Factor Error)
    plt.plot(P_trace[:, 12], label="dK_X Covariance (Index 12)")
    plt.title("Covariance Convergence with Sequential Shadow Injection")
    plt.xlabel("Iteration")
    plt.ylabel("P Variance Value")
    plt.grid(True)
    plt.legend()
    # Save the test plot locally
    plt.savefig('shadow_test_convergence.png')
    print("Saved plot to shadow_test_convergence.png")
    

if __name__ == "__main__":
    main()
