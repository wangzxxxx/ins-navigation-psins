import os
import collections
import numpy as np
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
from .shadow_kf import EnhancedKalmanFilter

class ShadowSimulationManager:
    """
    Manages the closed-loop evaluation between high-frequency KF updates and 
    LLM low-frequency asynchronous evaluation, implementing logic to trigger shadow updates.
    """
    def __init__(self, kf: EnhancedKalmanFilter, window_size: int = 2000, print_debug: bool = True):
        self.kf = kf
        self.print_debug = print_debug
        # Sliding buffer for physical state residuals/innovations
        self.innovation_buffer = collections.deque(maxlen=window_size)
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        self.provider_id = os.getenv("MODEL_PROVIDER_ID", "azure_openai")
        
        # Mapping LLM output confidences to R_llm relative variance multipliers
        # Using a relative ratio of the current variance ensures dimensionally-agnostic updates
        # and prevents absolute "Covariance Collapse" prematurely locking the filter.
        self.confidence_map = {
            "High": 0.5,   # R = 0.5 * P (Strong pull, P drops effectively by ~66%)
            "Medium": 2.0, # R = 2.0 * P (Medium pull, P drops by ~33%)
            "Low": 10.0,   # R = 10.0 * P (Weak pull, P drops by ~9%)
            "Reject": None
        }
        
        self.system_prompt = (
            "角色设定：\n"
            "您是一位顶尖的惯性导航系统（INS）算法专家，精通多位置系统级标定、卡尔曼滤波（Kalman Filter）理论、以及各类传感器误差模型（如零位偏置、标度因数误差、安装失准角等）。\n\n"
            "背景任务：\n"
            "当前，系统正在运行一个标准的 43 维的大型系统级标定误差状态卡尔曼滤波器，执行高频的速度零偏（Zero-velocity）物理更新。然而，某些状态量由于缺乏激励，经常面临不可观测性，导致协方差收敛缓慢。\n\n"
            "系统误差状态向量维度映射（SIMU标定下的无位置43维）：\n"
            "0-2: 姿态误差 (phi_E, phi_N, phi_U)\n"
            "3-5: 速度误差 (dVE, dVN, dVU)\n"
            "6-8: 陀螺零偏 (eb_x, eb_y, eb_z)\n"
            "9-11: 加计零偏 (db_x, db_y, db_z)\n"
            "12-20: 陀螺标度因数及安装角误差 (dKg的9个元素: xx, yx, zx, xy, yy, zy, xz, yz, zz)\n"
            "21-29: 加计标度因数及安装角误差 (dKa的9个元素: xx, yx, zx, xy, yy, zy, xz, yz, zz)\n"
            "30-42: 包括二次项(dKa2)、内臂误差(rx, ry, rz)及其他延迟参数等\n\n"
            "33-42: 内臂误差及其他相关参数\n\n"
            "工作机制：\n"
            "作为“影子观测者（Shadow Observer）”，我们将定期提取当前滤波器的物理残差滑动窗口统计特征以及协方差矩阵（P阵）的对角线方差数值。这是一场持续的系统级标定实验，所以请您记住历史推断，追踪长期的修正效果。\n\n"
            "请您凭借物理专家直觉分析当前的【动作上下文】与【残差收敛迟滞点】，推断出当前系统最有可能产生误差的系统状态向量索引（target_index，只能选取0~42中的一个）及其可能偏离的理论真实数值（predicted_value）。系统会利用您给出的信心指数（confidence）作为观测噪声构建一个非对称的新观测矩阵强制介入物理滤波过程。\n\n"
            "响应格式要求：\n"
            "每次交互，您应当首先输出一段关于您物理机理分析的文字，解释为何某些特征会发散，并论证您推断出的目标参数索引的合理性。分析完毕后，在回答的最后，必须且只能输出一段严格合法的 JSON 对象，包含您的最终调整指令。请务必将该 JSON 对象使用 ```json 和 ``` 包裹起来。\n"
            "JSON 结构须符合: {\"target_index\": int, \"predicted_value\": float, \"confidence\": \"High|Medium|Low|Reject\"}"
        )
        
        # Initialize client directly
        if self.api_key and self.base_url:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "X-Model-Provider-Id": self.provider_id
                }
            )
        else:
            self.client = None

    def _extract_semantic_features(self, current_action: str, past_maneuvers: list = None, dynamic_peaks: dict = None) -> dict:
        """
        Dynamically extracts temporal innovation features inside the window 
        buffer, generating qualitative semantics for LLM consumption.
        """
        innov_array = np.array(self.innovation_buffer) 
        
        # Linear polynomial fit (Degree 1) across the buffer timeline 
        # extracting drift "slopes". Time axis is just normalized index length here.
        time_axis = np.arange(len(innov_array))
        
        slopes = []
        for i in range(innov_array.shape[1]):
            slope = np.polyfit(time_axis, innov_array[:, i], 1)[0]
            slopes.append(float(slope))
        if hasattr(self.kf, 'P'):
            P_mat = self.kf.P
            cov_diag = np.diag(P_mat).tolist()
            state_x = self.kf.X.flatten().tolist() if hasattr(self.kf, 'X') else []
        elif isinstance(self.kf, dict) and 'Pxk' in self.kf:
            P_mat = self.kf['Pxk']
            cov_diag = np.diag(P_mat).tolist()
            state_x = self.kf['xk'].flatten().tolist()
        else:
            P_mat = np.eye(1)
            cov_diag = []
            state_x = []
            
        # Compute Top Correlations (>0.8 threshold)
        high_corr_pairs = []
        if len(cov_diag) > 1:
            try:
                std_devs = np.sqrt(np.abs(cov_diag))
                # Avoid division by zero
                std_devs[std_devs == 0] = 1e-12
                # Calculate Correlation Matrix: corr(i, j) = Cov(i, j) / (std(i) * std(j))
                corr_mat = P_mat / np.outer(std_devs, std_devs)
                
                # Fetch upper triangle values > 0.8
                for i in range(corr_mat.shape[0]):
                    for j in range(i + 1, corr_mat.shape[1]):
                        if abs(corr_mat[i, j]) > 0.8:
                            high_corr_pairs.append({
                                "idx_1": i,
                                "idx_2": j,
                                "correlation": float(corr_mat[i, j])
                            })
            except Exception as e:
                pass
            
        return {
            "maneuver_history": past_maneuvers if past_maneuvers else [],
            "current_maneuver": current_action,
            "dynamic_residual_peaks_during_motion": dynamic_peaks if dynamic_peaks else {},
            "stationary_residual_drift_slopes": slopes,
            "current_estimated_states_x": state_x,
            "cov_diagonal_variances": cov_diag,
            "high_covariance_correlations": high_corr_pairs
        }

    def _call_llm_expert(self, features: dict) -> dict:
        """
        Hits the external expert LLM endpoint feeding the semantic string 
        expecting JSON interpretation of calibration misalignments/scale-factors.
        Returns: Dict representing target index, value, and confidence.
        """
        if not self.client:
            print("Warning: LLM Client is not initialized. Skipping shadow injection.")
            return None
            
        user_prompt = f"当前标定动作与系统特征数据：\n{json.dumps(features, indent=2)}"
        
        if self.print_debug:
            print("\n" + "="*50)
            print("[DEBUG: Multi-turn Memory Active] LLM Update Requested.")
            print(f"User Prompt:\n{user_prompt}")
            print("="*50)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            content = response.choices[0].message.content
            content = content.strip()
            
            if self.print_debug:
                print("\n" + "="*50)
                print("[DEBUG: LLM Response Received]")
                safe_content = content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
                print(f"Content:\n{safe_content}")
                print("="*50 + "\n")
                
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content_to_parse = json_match.group(1).strip()
            else:
                content_to_parse = content
                
            parsed = json.loads(content_to_parse)
            return parsed
        except Exception as e:
            print(f"LLM API Call failed: {e}")
            return None

    def step(self, z_speed: np.ndarray, Phi: np.ndarray, current_action: str = "", trigger_shadow_eval: bool = False):
        """
        Main unified processing loop single-iteration step.
        """
        self.kf.time_update(Phi)
        innovation = self.kf.physical_update(z_speed)
        self.innovation_buffer.append(innovation)
        
        if trigger_shadow_eval and len(self.innovation_buffer) >= self.innovation_buffer.maxlen:
            print(f"Triggering Shadow Expert Logic via LLM for action: {current_action}...")
            features = self._extract_semantic_features(current_action)
            
            decision = self._call_llm_expert(features)
            
            if decision and decision.get("confidence") in self.confidence_map:
                conf = decision["confidence"]
                r_llm = self.confidence_map[conf]
                
                if r_llm is not None:
                    idx = decision.get("target_index", 0)
                    val = decision.get("predicted_value", 0.0)
                    
                    # --- SAFETY CLAMPING TO PREVENT DIVERGENCE ---
                    # Scale factor and physical bound limits to ensure LLM doesn't blow up the EKF
                    if 0 <= idx <= 2:     # phi (rad)
                        val = np.clip(val, -0.01, 0.01)
                    elif 3 <= idx <= 5:   # velocity (m/s)
                        val = np.clip(val, -1.0, 1.0)
                    elif 6 <= idx <= 8:   # gyro bias (rad/s)
                        val = np.clip(val, -1e-4, 1e-4) # ~20 dph max
                    elif 9 <= idx <= 11:  # acc bias (m/s2)
                        val = np.clip(val, -0.05, 0.05) # ~5 mg max
                    elif 12 <= idx <= 29: # dKg, dKa (ppm) -> scale factor & misalignment
                        val = np.clip(val, -0.005, 0.005) # ~5000 ppm max
                    else:
                        val = np.clip(val, -0.05, 0.05)

                    print(f"-> Shadow Injection Authorized! Target [{idx}], Val [{val:.2e}], Conf [{conf}], R_ratio [{r_llm}]")
                    self.kf.shadow_update(val, idx, r_llm)
                else:
                    print("-> Shadow Injection Rejected by LLM Confidence.")
                    
            # Wipe buffer context to require fresh physical residuals before evaluating again
            self.innovation_buffer.clear()
