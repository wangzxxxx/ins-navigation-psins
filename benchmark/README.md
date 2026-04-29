# psins_method_bench

目的：把 `tmp_psins_py/psins_py` 里混杂的多种标定尝试，拆成**可单独运行、可统一评估、可横向对比**的方法基准。

## 目标
- 每种方法一个精简入口
- 统一输入 / 输出格式
- 统一评估指标：标定精度、收敛速度、稳定性、副作用
- 最终将结果汇总到飞书文档

## 计划中的方法分组
- baseline
- correlation_decay / SCD
- markov noise
- adaptive RQ
- schmidt / shadow
- innovation gating
- inflation
- hybrid / staged

## 目录规划
- `methods/`：方法入口脚本
- `configs/`：统一配置
- `results/`：运行结果
- `summary/`：汇总与可视化
