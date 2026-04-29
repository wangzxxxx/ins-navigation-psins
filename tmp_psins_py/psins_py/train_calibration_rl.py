"""
train_rl_calibration.py
---------------------------------------------
Reinforcement Learning (PPO) for Dual-Axis Calibration Path Planning

State Space (45 dims):
  - Current Attitude C_n_b (9 dims flattened)
  - Current Uncertanties (30 dims, log scaled sigma/sigma0)
  - Static Face Coverage (6 dims, tracks how much static time spent on each of the 6 faces)

Action Space (Discrete 7):
  0: Inner Axis (IMU Y) Rotate +90 deg
  1: Inner Axis (IMU Y) Rotate -90 deg
  2: Inner Axis (IMU Y) Rotate +180 deg
  3: Outer Axis (Nav X) Rotate +90 deg
  4: Outer Axis (Nav X) Rotate -90 deg
  5: Outer Axis (Nav X) Rotate +180 deg
  6: Static 15s

Rewards:
  - Dense reward: uncertainty reduction (clamped >= 0 to avoid penalizing rotations).
  - Exploration bonus: +reward for static time on newly visited faces.
  - Redundancy penalty: -reward for excessive static time on the same face.
  - Time penalty: negative reward based on time consumed.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, a2qua
from test_calibration_markov_pruned import get_default_clbt, getFt_36, clbtkfinit_36

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Environment Definitions
class DualAxisCalibrationEnv(gym.Env):
    def __init__(self, max_time_s=1800.0, target_reduction=0.05):
        super().__init__()
        # Actions: 0-6
        self.action_space = spaces.Discrete(7)
        # Obs: C_n_b (9), log(errors) (30), coverage (6) -> Total 45
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(45,), dtype=np.float32)

        self.max_time_s = max_time_s
        self.target_reduction = target_reduction
        self.pos0 = posset(34.0, 0.0, 0.0)
        self.eth = Earth(self.pos0)
        self.wnie = glv.wie * np.array([0, math.cos(self.pos0[0]), math.sin(self.pos0[0])])
        self.Cba = np.eye(3)
        self.ts = 0.01
        self.nn, _, self.nts, _ = nnts(2, self.ts)
        self.frq2 = int(1 / self.ts / 2) - 1
        self.n = 36
        self.ARW = 0.01 * glv.dpsh
        self.VRW = 10.0 * glv.ugpsHz
        self.clbt_t = get_default_clbt()

        # Initialize Q stringently
        qvec = np.zeros(self.n)
        qvec[0:3] = self.ARW
        qvec[3:6] = self.VRW
        self.Qk = np.diag(qvec)**2 * self.nts

        # [type_id, angle_deg, rot_t, static_t]
        self.action_defs = {
            0: ('inner', 90, 9, 0),
            1: ('inner', -90, 9, 0),
            2: ('inner', 180, 18, 0),
            3: ('outer', 90, 9, 0),
            4: ('outer', -90, 9, 0),
            5: ('outer', 180, 18, 0),
            6: ('static', 0, 0.02, 15),  # 0.02s nominal rot_t to avoid div/0 in attrottt
        }

        self.reset()
        
    def get_obs(self):
        C_n_b = q2mat(self.qnb)
        C_n_b_flat = C_n_b.flatten()
        sigma_f = np.sqrt(np.diag(self.kf['Pxk']))[6:36]
        sigma_f = np.maximum(sigma_f, 1e-12)
        
        # log reduction: log(sigma / sigma0)
        log_reduction = np.log(sigma_f / self.sigma0)
        
        # Scale coverage so it stays roughly in [-1, 1] for neural net
        coverage_scaled = np.array(self.coverage) / 100.0
        
        obs = np.concatenate([C_n_b_flat, log_reduction, coverage_scaled]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.att0 = np.array([0.0, 0.0, 0.0]) * glv.deg
        self.qnb = a2qua(self.att0)
        
        self.kf = clbtkfinit_36(self.nts)
        self.kf['Pxk'][:, 2] = 0; self.kf['Pxk'][2, :] = 0
        self.Hk = self.kf['Hk']

        kf0 = clbtkfinit_36(self.nts)
        self.sigma0 = np.sqrt(np.diag(kf0['Pxk']))[6:36]
        self.sigma0 = np.where(self.sigma0 < 1e-30, 1.0, self.sigma0)

        self.clbt = {'Kg': np.eye(3), 'Ka': np.eye(3), 'Ka2': np.zeros(3),
                 'eb': np.zeros(3), 'db': np.zeros(3), 'rx': np.zeros(3), 'ry': np.zeros(3)}
        self.vn = np.zeros(3)
        self.t1s = 0.0
        
        self.current_time = 0.0
        self.L_prev = np.sum(np.log(self.sigma0))
        self.episode_history = []
        
        # 6-face static coverage (time in seconds)
        # Indicies: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
        self.coverage = [0.0] * 6
        
        return self.get_obs(), {}

    def get_gravity_face(self):
        """Returns the index of the face pointing opposite to gravity (Nav +Z)"""
        C_n_b = q2mat(self.qnb)
        # C_n_b[2, :] represents the projection of IMU X, Y, Z onto Nav Z axis.
        # Max absolute value indicates the nearest aligned axis.
        projections = C_n_b[2, :]
        idx_max = np.argmax(np.abs(projections))
        sign = np.sign(projections[idx_max])
        # Map to 0-5
        # idx=0 means sign>0, idx=1 means sign<0
        return int(idx_max * 2 + (0 if sign > 0 else 1))

    def step(self, action):
        action_type, angle_deg, rot_t, static_t = self.action_defs[action]
        duration = rot_t + static_t
        
        C_n_b = q2mat(self.qnb)
        if action_type == 'inner':
            axis = C_n_b[:, 1]
        elif action_type == 'outer':
            axis = np.array([1.0, 0.0, 0.0])
        else: # static
            axis = np.array([0.0, 0.0, 1.0])
            
        ax, ay, az = axis
        n = math.sqrt(ax**2 + ay**2 + az**2)
        if n > 1e-6:
            ax, ay, az = ax/n, ay/n, az/n
        
        paras = np.array([[1, ax, ay, az, angle_deg * glv.deg, rot_t, 0.0, static_t]], dtype=float)
        
        att_seq = attrottt(self.att0, paras, self.ts)
        self.att0 = att_seq[-1, :3] 
        
        if att_seq.shape[0] > 1:
            imu, _ = avp2imu(att_seq, self.pos0)
            imu_clean = imuclbt(imu, self.clbt_t)
            dotwf = imudot(imu_clean, 5.0)

            for k in range(0, len(imu_clean)-self.nn, self.nn):
                k1 = k + self.nn - 1
                wm = imu_clean[k:k1+1, :3]
                vm = imu_clean[k:k1+1, 3:6]
                dwb = np.mean(dotwf[k:k1+1, :3], axis=0)
                phim, dvbm = cnscl(np.hstack((wm, vm)))
                phim = self.clbt['Kg'] @ phim - self.clbt['eb'] * self.nts
                dvbm = self.clbt['Ka'] @ dvbm - self.clbt['db'] * self.nts
                wb, fb = phim/self.nts, dvbm/self.nts
                SS = imulvS(wb, dwb, self.Cba)
                Ft = getFt_36(fb, wb, q2mat(self.qnb), self.wnie, SS)
                Phi = np.eye(self.n) + Ft * self.nts
                self.kf['Pxk'] = Phi @ self.kf['Pxk'] @ Phi.T + self.Qk
                self.t1s += self.nts
                
                if self.t1s > (0.2 - self.ts/2):
                    self.t1s = 0.0
                    ww = np.mean(imu_clean[max(0, k-self.frq2):k+self.frq2+1, :3], axis=0) if k >= self.frq2 else np.zeros(3)
                    
                    if np.linalg.norm(ww)/self.ts < 20*glv.dph:
                        S = self.Hk @ self.kf['Pxk'] @ self.Hk.T + self.kf['Rk']
                        K = self.kf['Pxk'] @ self.Hk.T @ np.linalg.inv(S)
                        I_KH = np.eye(self.n) - K @ self.Hk
                        self.kf['Pxk'] = I_KH @ self.kf['Pxk'] @ I_KH.T + K @ self.kf['Rk'] @ K.T

                fn = qmulv(self.qnb, fb - self.clbt['Ka2']*(fb**2) - SS[:,0:6] @ np.concatenate((self.clbt['rx'], self.clbt['ry'])))
                self.vn = self.vn + (rotv(-self.wnie*self.nts/2, fn) + np.array([0,0,-self.eth.g])) * self.nts
                self.qnb = qupdt2(self.qnb, phim, self.wnie*self.nts)
                
        self.current_time += duration
        self.episode_history.append((int(action), duration))

        # ---------------- Reward Calculation ----------------
        sigma_f = np.sqrt(np.diag(self.kf['Pxk']))[6:36]
        sigma_f = np.maximum(sigma_f, 1e-12)
        L_current = np.sum(np.log(sigma_f))
        
        # 1. Uncertainty reduction (clamped so rotations are not heavily penalized implicitly)
        reduction_gain = max(0, self.L_prev - L_current)
        reward = 10.0 * reduction_gain
        self.L_prev = L_current
        
        # 2. Exploration / Redundancy Bonus for Statics
        if action == 6: # static
            face_idx = self.get_gravity_face()
            # If we haven't spent much time on this face, grant an exploration bonus
            if self.coverage[face_idx] < 120.0:  # budget 120s per face max for bonus
                self.coverage[face_idx] += static_t
                reward += 5.0  # good job collecting static data here!
            else:
                # We spent too much static time here, heavy penalty to force moving
                reward -= 15.0
                
        # 3. Time efficiency penalty
        reward -= 0.1 * duration
        
        red_percent = (sigma_f / self.sigma0) * 100.0
        worst_red = float(np.max(red_percent))
        
        done = False
        info = {'worst_red': worst_red, 'time': self.current_time}
        
        if worst_red <= self.target_reduction * 100.0:
            done = True
            reward += 1000.0
            print(f"Goal Reached! Worst Reduction: {worst_red:.2f}%. Time: {self.current_time}s")
            
        if self.current_time >= self.max_time_s:
            done = True
            
        return self.get_obs(), reward, done, False, info


class ConsoleAndTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episodes = 0
        
    def _on_step(self):
        if "worst_red" in self.locals["infos"][0]:
            if self.locals["dones"][0]:
                self.episodes += 1
                info = self.locals["infos"][0]
                worst_red = info["worst_red"]
                total_time = info["time"]
                reward = self.locals["rewards"][0]
                
                print(f"[Episode {self.episodes:4d}] Time: {total_time:6.1f}s | Worst Reduc: {worst_red:6.2f}% | Final Step Reward: {reward:8.3f}")
                
                self.logger.record("custom/episode_worst_red", worst_red)
                self.logger.record("custom/episode_time", total_time)
        return True


if __name__ == "__main__":
    env = DualAxisCalibrationEnv(max_time_s=1800.0)
    print("Environment Created. Space details:")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    print("Testing 10 random steps...")
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: Action {action:2d} -> Reward: {reward:7.3f}, Worst Reduc: {info['worst_red']:6.2f}%, Time: {info['time']}s")
        if done:
            break
    print("Test passed! Starting training...")

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_tensorboard")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, ent_coef=0.05) # Increased entropy coef for exploration

    print("Training PPO Model... (Console output enabled. View also via python plot_rl_progress.py)")
    model.learn(total_timesteps=1_000, callback=ConsoleAndTensorboardCallback())
    
    model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dual_axis_calib_ppo"))
    print("Done! Model saved.")
