# psins benchmark workflow

## 目标

针对 `psins_py` 这类包含多种标定尝试的研究工程，执行统一 benchmark：

1. 先识别一个脚本里是否混跑了多个方法
2. 把这些方法提炼成独立方法文件
3. 把公共数据构造抽到公共层
4. 把结果处理和画图抽到独立层
5. 统一对比：
   - 标定精度
   - 收敛速度
   - 后段波动 / 稳定性
   - 副作用风险

## 当前工程约定

工作目录：
- `/root/.openclaw/workspace/psins_method_bench/`

现有分层：
- `methods/`：方法入口
- `configs/`：统一配置
- `results/`：运行结果
- `summary/`：指标提取与画图

## 先做什么

### A. 方法识别
优先判断一个原始脚本中是否混跑：
- clean baseline
- noisy baseline
- 某种增强方法（SCD / Markov / Adaptive RQ / Schmidt / Shadow / Hybrid 等）

### B. 方法提炼
不是只做入口壳子，而是把**每种实验方法本体**提炼出来，至少拆成：
- 公共数据构造层
- 方法运行层
- 结果提取层

### C. 统一跑法
优先用本地 `exec` 跑，不优先用子 agent。
原因：
- 子 agent 有模型 / embedded run timeout 风险
- 纯执行任务更适合本地长超时进程

### D. 统一输出
最少要产出：
- 方法名
- 是否跑通
- 耗时
- tuple / result 结构说明
- 最终精度指标（如 eb/db/Ka2/rx/ry）
- 收敛速度和后段稳定性（如果能提取）

## 结果文档

优先整理成飞书文档，建议先给：
1. 直观对比表
2. 精度对比表
3. 收敛速度 / 稳定性补充表
4. 方法副作用分析
