# Unfinished Methods Summary

## 本轮补充结论

这次补跑后，`psins_method_bench` 里剩余未闭环的方法，已经不再只是“没跑到”。现在可以更明确地分成两类：

- **RL 三个方法是明确失败**：因为 `.venv_psins` 缺依赖，不是算法内部错误
- **attention / LLM-tuning / hybrid direct verify 主要是重计算长跑问题**：不是启动即崩，当前更像是预算不够、机器并发过高、缺阶段日志导致难验证

另外，本轮确认到宿主机当时还有别的 PSINS 任务并行在跑（包括 inflation / llm tuning 类），所以 CPU 负载并不干净，会拖慢验证速度。

---

## 一、RL 方法：已拿到精确失败原因

### 1) rl_train
- 入口：`methods/rl/method_rl_train.py`
- 结果：`results/rl_train.json`
- 状态：`dependency_missing`
- 精确原因：
  - `ModuleNotFoundError: No module named 'gymnasium'`

这不是训练逻辑本身的问题，而是运行环境里没有 `gymnasium`。

### 2) rl_evaluate
- 入口：`methods/rl/method_rl_evaluate.py`
- 结果：`results/rl_evaluate.json`
- 状态：`dependency_missing`
- 精确原因：
  - `ModuleNotFoundError: No module named 'stable_baselines3'`

这说明评估脚本连 PPO 模型类都还没法导入。

### 3) rl_progress_plot
- 入口：`methods/rl/method_rl_progress_plot.py`
- 结果：`results/rl_progress_plot.json`
- 状态：`dependency_missing`
- 精确原因：
  - `ModuleNotFoundError: No module named 'tensorboard'`

另外还发现这个脚本默认日志目录写死成 Windows 路径：
- `D:/psins251010/psins251010/psins_py/rl_tensorboard`

即使装完依赖，后面也最好把默认 logdir 改成当前 workspace 下的实际路径。

### RL 侧下一步最直接修法
1. 往 `/root/.openclaw/workspace/.venv_psins` 里补装：
   - `gymnasium`
   - `stable-baselines3`
   - `tensorboard`
2. 然后再跑：
   - `train_calibration_rl.py`
   - 训练完确认 `dual_axis_calib_ppo.zip` 存在
   - 再跑 `evaluate_rl_model.py`
   - 最后再跑 `plot_rl_progress.py`

---

## 二、attention_inflation：仍然是“重方法”，不是立即报错

### 4) attention_inflation
- 入口：`methods/adaptive_robust/method_attention_inflation.py`
- 旧证据：之前汇总里已经标过一次 `killed_timeout`
- 本轮新增观察：
  - 直接用脚本做 20 秒 sanity run：**没有 stdout / stderr，20 秒后超时**
  - 说明它不是一启动就抛 Python 异常
  - 同时机器上还存在别的 inflation 相关长跑任务在并发进行

### 当前更可信的判断
它更像是：
- 43-state + shadow/inflation 结构本身很重
- 再叠加 LLM / shadow manager / 双循环
- 在高并发宿主机上很容易只表现为“长时间沉默 + 吃 CPU”

### 下一步建议
- 不要再把它和别的方法混在一个短 runner 里跑
- 单独长时运行
- 先加阶段日志（初始化 / baseline loop / shadow loop / trigger window）
- 最好先临时 mock 掉 LLM 调用，先分离“算法慢”还是“外部网络慢”

---

## 三、LLM-tuning 四个方法：当前证据更像“长跑未验证”，不是立即崩

### 5) hyperparam_tuner
- 入口：`methods/llm_tuning/method_hyperparam_tuner.py`
- 现有 runner JSON：`results/hyperparam_tuner.json`
- 现象：runner 记成 `returncode=2, runtime_sec=0`
- 但本轮直接 20 秒运行：**没有 stdout/stderr，只是超时**

### 6) llm_assisted_19pos
- 入口：`methods/llm_tuning/method_llm_assisted_19pos.py`
- 现有 runner JSON：`results/llm_assisted_19pos.json`
- 同样：runner 写成 `returncode=2, runtime_sec=0`
- 但直接 20 秒运行：**无输出，超时**

### 7) path_optimizer
- 入口：`methods/llm_tuning/method_path_optimizer.py`
- 现状：当时 `path_optimizer.json` 还没落盘
- 但直接 20 秒运行 `calibration_path_optimizer_llm.py`：**无输出，超时**

### 8) q_tuning
- 入口：`methods/llm_tuning/method_q_tuning.py`
- 现有 runner JSON：`results/q_tuning.json`
- runner 写成 `returncode=2, runtime_sec=0`
- 但直接 20 秒运行：**无输出，超时**

### 这里最重要的判断修正
这些方法之前容易被误判成“runner 一调用就失败”。

但本轮直接跑脚本后，至少可以确认：
- **它们不是明显的 import/syntax 级立即报错**
- 更像是进入了长流程，只是短时间内没有打任何日志
- 现有 `returncode=2/runtime_sec=0` 很可能是某个 wrapper / 路径层面的记录问题，不足以代表方法本体失败

### 下一步建议
- 给 `llm_tuning` 单独写可靠 runner
- 结果 JSON 里必须写：
  - 实际执行命令
  - 绝对脚本路径
  - runtime
  - stdout/stderr tail
  - `llm_available`
  - `fallback_used`
- 先加早期阶段日志，否则很难分辨“正常长跑”还是“卡死”

---

## 四、shadow / hybrid 优先项：结果已存在，但 fresh direct verify 没在预算内跑完

### 9) hybrid_direct_verify（本轮新增验证项）
- 直接脚本：`tmp_psins_py/psins_py/test_system_calibration_19pos_hybrid.py`
- 结果：`results/hybrid_direct_verify.json`
- 状态：`timeout_or_killed`
- 返回码：`137`
- 运行时长：`480s`
- stdout/stderr：都为空

### 这说明什么
- 它不是启动即崩
- 进程检查时看到它一直在吃 CPU，属于**计算中**
- 只是 480 秒预算还不够，尤其当时机器上还有别的 PSINS 长任务并发

### 对“trigger verification”的意义
这次没拿到新的 trigger 打印证据，但可以确认：
- fresh direct rerun 不是 import 问题
- 不是 syntax 问题
- 是**太慢 / 负载太高 / 缺早期日志**，导致在预算内无法完成验证

### 现有可用旧证据
- `results/shadow_hybrid_update.stdout.txt`
- `results/hybrid_shadow_kf.stdout.txt`

这两个已有成功结果仍然是当前最可靠的 shadow/hybrid 成功产物。

另外：
- `results/llm_staged_graduation.stdout.txt` 明确显示：
  - `[LLM] Client initialized: model=mimo-v2-flash`
  - 但每轮 review 都是 `Connection error`
  - 所以 staged 流程确实跑了，但没有拿到真正的 graduation 决策

---

## 五、当前 unfinished 的更准确分类

### A. 明确环境依赖失败
- `rl_train` → 缺 `gymnasium`
- `rl_evaluate` → 缺 `stable_baselines3`
- `rl_progress_plot` → 缺 `tensorboard`

### B. 重计算/长跑，当前预算内未完成验证
- `attention_inflation`
- `hyperparam_tuner`
- `llm_assisted_19pos`
- `path_optimizer`
- `q_tuning`
- `hybrid_direct_verify`

---

## 六、我这轮实际落盘/更新了什么

已更新：
- `psins_method_bench/results/unfinished_methods_summary.json`
- `psins_method_bench/results/unfinished_methods_summary.md`

并新增/确认了这些结果文件：
- `psins_method_bench/results/rl_train.json`
- `psins_method_bench/results/rl_evaluate.json`
- `psins_method_bench/results/rl_progress_plot.json`
- `psins_method_bench/results/hybrid_direct_verify.json`

---

## 七、给主 agent 的最实用结论

如果后续继续 overnight：

### 第一优先级
先补 RL 依赖，不然 RL 三个方法永远是硬失败：
- `gymnasium`
- `stable-baselines3`
- `tensorboard`

### 第二优先级
把 `attention_inflation` 和 `llm_tuning/*` 的 runner 改可靠：
- 绝对路径
- 明确记录命令
- 早期阶段日志
- JSON 产物规范化

### 第三优先级
如果要真做 trigger verification：
- 不要直接拿大脚本全量再跑一遍赌 8 分钟
- 应该先给 hybrid / inflation 脚本加 `flush=True` 的早期打印
- 或做一个 reduced-size verification mode
- 否则很容易只得到“吃满 CPU 然后超时”的结果
