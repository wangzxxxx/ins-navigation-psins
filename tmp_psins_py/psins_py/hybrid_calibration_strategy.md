# 双速率大语言模型驱动的混合寻优无源自标定系统
(Dual-Rate LLM-Driven Hybrid INS Self-Calibration System)

这是一个旨在解决高阶组合导航标定中“局部死锁”、“隐藏轴向不可观测”两大痛点的智能标定框架。该方案打破了传统扩展卡尔曼滤波（EKF）纯靠数学协方差硬传导的局限性，**首次将具有全局时空语义感知能力的大语言模型（LLM）作为“虚拟专家观测节点”直接接入到滤波器的深层状态更新方程中**。

---

## 痛点分析与解题思路

在经典的 19 位置法（或动态标定）体系中，陀螺仪和加速度计的刻度系数误差 ($dK_g, dK_a$)、安装误差角、二阶项 ($dK_{a2}$) 甚至是内臂杆效应等近 **43 维** 的超高维误差状态，仅依赖于极少的可观测物理量（如静态速度残差 $v^n - 0$）。
这会导致两个致命的观测瓶颈：
1. **弱观测发散**：非对角线参数如 $dK_{g\_yz}$ 或内杆臂参数受限于机动条件不足，协方差矩阵收敛极其缓慢。
2. **错误洼地与死锁**：一旦某项估计因非线性耦合掉入错误的洼地（即估计值极度偏离真值，但对应协方差已被物理方程压榨至逼近零 $P_{xx} \to 0$），传统的 EKF 会形成“极端自信的错觉”，拒收任何后续的残差校准，这被称为 **滤波死锁 (Filter Deadlock)**。

**混合智能方案 (Hybrid Intelligent Strategy)** 的核心破局点就是引入外部先验：
- 针对“弱观测发散”，采用**大模型深层伪观测注入 (Phantom Observation Injection)**。
- 针对“极端自信导致的死锁”，采用**大模型注意力协方差膨胀 (Attention-Driven Covariance Inflation)**。


---

## 架构：异步双速率闭环 (Dual-Rate Asynchronous Loop)

整个系统由物理引擎层（运行在 PSINS Python 内核中）和虚拟专家层（运行在云端大模型中）构成“双速率”异构架构。

1. **高频内环 (Physical Ring - $100Hz$)**：
   - 运行原生原速的增强版扩展卡尔曼滤波 `EnhancedKalmanFilter`。
   - 不断依靠惯导更新积分 (`qupdt2`, `rotv`) 与真实物理零速反馈 (`physical_update`) 闭环估计误差状态 $X_k$ 和协方差阵 $P_k$。
   - 此环中没有大模型阻塞，保证硬实时性。

2. **低频外环 (Shadow Ring - $0.2Hz \sim 1Hz$)**：
   - 由 `ShadowSimulationManager` (影子管理器) 掌控。
   - 它在后台滑动采集内环物理层在旋转机动阶段的以下语料特征 (Semantic Features)：
     - **新息峰值 (Dynamics Residual Peaks)**：例如机动时的速度跳变极值。
     - **实时内部状态矩阵 (Estimates & Correlates)**：当前状态变量估计值 $x_k$ 以及他们之间在 $P_k$ 中的共变关系。
     - **机动上下文历史 (Maneuver Semantic History)**：将生冷的三轴角速率 $\omega_{b}$ 翻译为 "Continuous rotation around the Y-axis..." 的自然文体。
   - 将上述特征合并打包为 JSON Prompt 发送给 LLM。


---

## 核心机理：二段式混合破局更新 (Two-Stage Hybrid Update)

大模型根据传过去的残差症状和机动画像，输出一份诊断指令（一个包含了所有疑似故障节点的 JSON 清单），每个指令节点包含三大要素：
- `target_name`: 指出是哪个状态病了（例如 `dKa_zx` 横向轴间混叠）。
- `predicted_value`: 大模型凭借全局知识和物理对称性，预发出的真值修正量 $z_{llm}$。
- `deadlock_inflation`: 建议的惩罚膨胀信噪比（例如 100 倍）。
- `confidence`: 对本次判据的自信心。

收到这份包含领域知识的高级诊断后，底层 EKF 会触发独有的 `hybrid_update(val, target_idx, r_llm, inflation, ...)` 机制：

### 第一段：破锁 (Attention-Driven Covariance Inflation)
如果触发了 inflation，直接切开由标准力学传播带来的死锁面纱。
数学上，我们只对被点名的对应状态变量在主对角线上的方差强行扩大：
$$ P_{target, target} = P_{target, target} \times \text{Inflation\_Factor} $$
**效果**：系统被迫对这个维度的历史计算“产生自我怀疑”，增益控制矩阵 Kalman Gain $K$ 的门槛被轰塌，滤波器重置对新生输入的吞吐饥渴感。

### 第二段：引流 (Phantom Observation Injection)
既然破开了死锁，就需要告诉系统正确的走向。此时将模型给出的预判真值 $z_{llm}$ 当作一种**非真实物理发生、但是在数学世界中合法的虚拟观测** 注入。
我们构造一个零散观测矩阵 $H_{llm}$，除了 `target_idx` 位置是 1 以外全为 0：
$$ Z_{llm} = z_{llm} $$
$$ y_{innovation} = Z_{llm} - X[target\_idx] $$
进而使用标准的标准序贯更新：
$$ S_{llm} = H_{llm} P H_{llm}^T + R_{llm} $$
$$ K_{llm} = P H_{llm}^T S_{llm}^{-1} $$
$$ X_{new} = X + K_{llm} \cdot y_{innovation} $$
$$ P_{new} = (I - K_{llm}H_{llm})P $$

**效果**：在 P 阵因上一步剧变而极度敞开接纳的状态下，虚拟残差引导系统强行跃升出“局部极小值洼地”，抵达全局正确的参数谷底。并且整个状态空间的其他相关维度会沿着 $P$ 阵的交叉列联动修正 (Cross-correlation pulling)。


---

## 为什么这种设计合理且极其安全？

1. **单向低频干扰，高频物理闭环接手**：
   LLM 不是一个取代物理的黑盒神经网络，它被设计为一个**拨弦人**。
   通常只会在整个庞大仿真周期中的初始几次迭代时注入大模型预测（这被称为第一遍的激发态 First-Pass Boosting）。被强行牵引出局部死锁点后，随后的数小时乃至数次的标准多位置机动将完全由物理层 (Baseline EKF) 自己通过真实零速反馈压实。
   
2. **拒绝“错误固化”**：
   如果 LLM 因为幻觉带来了一个错误的预估，它在随后的纯物理运动更新中，会表现出剧烈的速度残差激增（不符合物理实际）；此时原有的连续 $R\_speed$ 卡尔曼更新会立刻发挥负反馈阻尼的作用，重新通过矩阵倒数消去了这一影响。

## 结论

这种 "死锁膨胀 (Inflation) + 虚拟观测 (Pseudo-meas)" 双保险混合驱动方案，将纯代数递归模型与概率推理大语言模型相结合。使基于多位置/纯动态机动的高精度标定，在观测自由度不足的极差环境下，打破几何盲区，实现状态方差收敛速度和终态绝对精度的降维打击。
