# Summary Layer

这一层负责把方法层的输出，统一变成：
- 指标 JSON
- 对比表
- 图表

当前组件：
- `extract_metrics.py`：从单个方法返回值里抽核心指标
- `plot_metrics.py`：根据指标画图
- `method_groups.md`：方法分组说明
- `iteration_protocol.md`：实验学习 → 分析 → 结果 → 下一轮实验生成的固定协议
- `round_record_template.md`：每轮实验记录模板
- `current_iteration_state.md`：当前 mainline / branch / 下一步状态卡
- `research_program.md`：主线研究计划、baseline 规范、论文级方法路线图

总控：
- `/root/.openclaw/workspace/psins_method_bench/run_all_methods.py`
  - 负责串联运行、抽指标、存 JSON、画图
