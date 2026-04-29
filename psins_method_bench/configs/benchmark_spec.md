# Benchmark Spec

## 目标
把原始研究代码整理成**两类**：
1. `baseline/`：对照方法
   - clean baseline
   - noisy baseline
2. `methods/`：所有改动方法
   - SCD
   - Markov / GM
   - adaptive RQ
   - huber robust
   - innovation gating
   - inflation
   - schmidt / shadow
   - staged / hybrid

## 统一要求
- 所有方法统一使用 **36 状态误差模型**
- 统一状态标签定义来自：`configs/state_model_36.py`
- 不再保留 42 / 46 / 48 / 49 / 55 状态作为最终 benchmark 主标准
- 若原方法依赖扩维状态，需先映射 / 裁剪 / 降维到 36 状态主标准后再纳入比较

## 统一评估项
- 标定精度
- 收敛速度
- 后段波动 / 稳定性
- 副作用风险
