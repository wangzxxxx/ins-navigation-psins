import os
import sys
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_calibration_rl import DualAxisCalibrationEnv
from psins_py.nav_utils import glv

def evaluate_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dual_axis_calib_ppo.zip")
    if not os.path.exists(model_path):
        print(f"Error: Model file NOT found at {model_path}.")
        return

    print(f"Loading trained RL model from: {model_path}")
    model = PPO.load(model_path)
    
    # Run a deterministic evaluation
    env = DualAxisCalibrationEnv(max_time_s=1800.0)
    obs, info = env.reset()
    
    print("\nStarting Deterministic Evaluation Episode...")
    done = False
    step_count = 0
    total_reward = 0.0
    
    action_log = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # scalar action
        action = int(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Log action info
        act_info = env.action_defs[action]
        action_log.append({
            'step': step_count,
            'action_id': action,
            'desc': f"{act_info[0]} {act_info[1]:>3} deg, static: {act_info[3]:>2}s",
            'time': info['time'],
            'worst_red': info['worst_red'],
            'reward': reward
        })
        
        step_count += 1
        total_reward += reward
        print(f"[{info['time']:6.1f}s] Action {action}: {action_log[-1]['desc']} | Worst Reduc: {info['worst_red']:6.2f}%")
        
        # Stop if we hit 100 steps to avoid infinite loops if constraints fail
        if step_count > 100:
            print("[!] Force stopping evaluation at 100 steps.")
            break
            
    print(f"\nEvaluation Complete! Total Steps: {step_count}")
    print(f"Total Time Consumed: {info['time']:.1f} s")
    print(f"Final Worst Reduction: {info['worst_red']:.2f}%")
    print(f"Total Accumulated Reward: {total_reward:.2f}")

    print("\n--- Detailed State Convergence ---")
    STATE_LABELS = [
        # Gyro Biases (6-8)
        'eb_x', 'eb_y', 'eb_z', 
        # Accel Biases (9-11)
        'db_x', 'db_y', 'db_z',
        # Gyro Scale/Cross (12-20)
        'Kg00', 'Kg10', 'Kg20', 'Kg01', 'Kg11', 'Kg21', 'Kg02', 'Kg12', 'Kg22',
        # Accel Scale/Cross (21-26)
        'Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz',
        # Accel Order 2 (27-29)
        'Ka2_x', 'Ka2_y', 'Ka2_z',
        # Inner/Outer Arm lever (30-35)
        'rx_x', 'rx_y', 'rx_z', 'ry_x', 'ry_y', 'ry_z'
    ]
    
    sigma_f = np.sqrt(np.diag(env.kf['Pxk']))[6:36]
    red_percent = (sigma_f / env.sigma0) * 100.0
    
    for i, label in enumerate(STATE_LABELS):
        val = red_percent[i]
        status = "GOOD" if val < 5.0 else ("OK" if val < 20.0 else "POOR")
        print(f"  {label:<8s} {val:6.2f}%  {status}")

if __name__ == "__main__":
    evaluate_model()
