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
        
        # Mapping LLM output confidences to Covariance Inflation Multipliers
        # Instead of noise constraints, we bloat the P matrix diagonal by these factors
        # High confidence in an error -> Massive inflation to force the KF to re-estimate it
        self.confidence_map = {
            "High": 500.0,   # Multiply P by 500
            "Medium": 50.0,  # Multiply P by 50
            "Low": 5.0,      # Multiply P by 5
            "Reject": None
        }
        
        self.system_prompt = (
            "角色设定：\n"
            "您是一位顶尖的惯性导航系统（INS）算法专家，精通多位置系统级标定、卡尔曼滤波（Kalman Filter）机理、以及“协方差膨胀（Covariance Inflation）”自适应滤波理论。\n\n"
            "背景任务：\n"
            "当前，系统正在运行一个标准的 43 维的大型系统级标定卡尔曼滤波器，执行高频的速度零偏（Zero-velocity）物理更新。然而，某些状态由于长期缺乏旋转激励面临极弱可观测性，导致它们的方差收敛陷入停滞错误参数中（即产生了“协方差坍塌”，滤波器过度自信，不再吸收实际物理观测残差）。\n\n"
            "系统误差状态向量维度映射（SIMU标定下的无位置43维）：\n"
            "0-2: 姿态误差 (phi_E, phi_N, phi_U)\n"
            "3-5: 速度误差 (dVE, dVN, dVU)\n"
            "6-8: 陀螺零偏 (eb_x, eb_y, eb_z)\n"
            "9-11: 加计零偏 (db_x, db_y, db_z)\n"
            "12-20: 陀螺标度因数及安装角误差 (dKg的9个元素: xx, yx, zx, xy, yy, zy, xz, yz, zz)\n"
            "21-29: 加计标度因数及安装角误差 (dKa的9个元素: xx, yx, zx, xy, yy, zy, xz, yz, zz)\n"
            "30-42: 包括二次项(dKa2)、内臂误差(rx, ry, rz)及其他延迟参数等\n\n"
            "33-42: 内臂误差及其他相关参数\n\n"
            "工作机制（新架构）：\n"
            "作为“注意力引导专家（Attention Oracle）”，我们将定期提取当前滤波器的物理残差滑动窗口统计特征以及协方差矩阵（P阵）的对角线方差数值。\n"
            "您**不需要**凭空猜测一个绝对的物理浮点数注入系统（否则会导致幻觉污染），而是需要凭借物理专家直觉分析当前的【动作上下文】与【残差收敛迟滞点】，找准那个“误差极大但方差已坍塌停滞”的系统状态向量索引（target_index）。\n"
            "系统会自动提取您的置信度，并且对该 target_index 的协方差主对角线进行巨大的**乘性方差膨胀**！这样，物理滤波器在下几个观测周期中，就会被强迫把注意力全部分配给该状态，利用最真实的物理方程重新估计它！\n\n"
            "响应格式要求：\n"
            "每次交互，您应当首先输出一段物理机理分析文字，论证您推断出的目标参数索引。分析完毕后，在回答最后，必须且只能输出一段严格合法的 JSON 数组对象。\n"
            "JSON 结构须符合: [{\"target_index\": int, \"confidence\": \"High|Medium|Low|Reject\"}, ...]\n"
            "最多输出 3 个您认为最需要放大注意力的目标状态！"
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
                    
                    print(f"-> Attention Inflation Authorized! Target [{idx}], Conf [{conf}], P Multiplier [{inflation_factor:.1f}x]")
                    self.kf.attention_update(idx, inflation_factor)
                else:
                    print("-> Attention Inflation Rejected by LLM Confidence.")
                    
            # Wipe buffer context to require fresh physical residuals before evaluating again
            self.innovation_buffer.clear()
