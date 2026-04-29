# PSINS Research Program (Mainline Reset)

_Updated: 2026-03-28_

## 1. 研究目标

目标不是继续堆很多轮“看起来有变化”的实验，而是收敛到：

> 在**固定 noisy dataset** 上，以**标准卡尔曼滤波**为 baseline，逐步融合真正有效的机制，得到一个 **更精确、可解释、可消融、可写论文** 的标定方法。

最终只追求一件事：
- 产出一个比当前最好方法更好的正式方法
- 这个方法必须有明确创新点、严格消融、标准基线、可复现实验链路

---

## 2. 从现在起的硬规范

### 2.1 baseline 规范
所有对比必须包含：
- **标准 KF baseline**
- baseline 数据必须是 **有噪声的同一份数据**
- baseline 的 noise strength、seed、dataset 全部固定

### 2.2 同轮对比规范
同一批实验中：
- 所有方法必须使用 **同一份 noisy dataset**
- 不允许方法 A / B 用不同 noise strength 然后直接比优劣
- 如果要研究 noise sensitivity，那是单独的 **regime robustness study**，不能冒充 mainline result

### 2.3 主线与分支分离
- **mainline**：默认 1x noisy dataset 上的方法优劣比较
- **branch**：ultra-low / other noise regimes 的特殊分支比较
- branch 结果不能直接替代 mainline winner

---

## 3. 历史轮次的筛选结论

基于已经做过的大量轮次，当前应保留为“有真实方法价值”的核心积木主要只有：

1. **Markov 噪声建模**
   - 提供比纯白噪声 baseline 更贴近系统误差动态的状态建模能力
2. **SCD（相关性抑制）**
   - 作为温和、一次性的 cross-covariance control 模块
3. **Round61 主线方法**
   - 当前最成熟的主线融合形态：trust/cov/release 内化框架 + targeted micro-guard + gentle SCD hybrid

除此之外，大量历史轮次更多是：
- 通向这些结论的探索
- 某些局部 regime 的 probe
- 或明确失败后留下的 negative lesson

因此，后续研究不再把所有历史轮次视为“平级候选方法”，而是把它们视为：
- baseline
- ablation steps
- failed lessons
- branch-specific probes

---

## 4. 论文级研究主线应该长什么样

### 4.1 论文主问题
**如何在标准 KF 标定框架中，引入更合理的 coloured-noise state modeling、相关性抑制和置信度驱动的内部反馈机制，从而在固定 noisy dataset 上更精确地恢复误差参数？**

### 4.2 论文主线方法结构（当前版本）
建议把方法主线组织成下面的递进：

1. **Baseline-A：标准 KF on noisy data**
2. **Baseline-B：KF + Markov noise modeling**
3. **Method-C：Markov + SCD**
4. **Method-D：Markov + SCD + Round61 internalized trust/cov/release micro-guard**
5. **Method-E（新方法）**：在 D 的基础上加入一个新的、可解释的、更严格设计的创新模块

也就是说，新方法不能凭空跳出，而应站在：
- standard KF
- Markov
- SCD
- Round61
这条清晰梯子之上。

---

## 5. 新方法设计原则

新方法必须满足：

1. **创新点明确**
   - 不是再调几个数
   - 而是提出一个新的、可命名的机制
2. **机制上可解释**
   - 能说明它为什么可能改善参数可辨识性 / 误差收敛质量
3. **实现上受控**
   - 不能引入大范围不可解释 search
4. **实验上可消融**
   - 能拆出：去掉该模块会怎样
5. **结果上可比较**
   - 统一 baseline、统一 dataset、统一 seed

---

## 6. 允许的下一代创新方向（优先级）

### 优先方向 A：在 Round61 主线上加“信息一致性 / innovation consistency”约束
思路：
- 不是再改超低噪声分支
- 而是让 internalized feedback / SCD 触发更受 **innovation statistics** 约束
- 目标：减少“局部指标改善但 protected metrics 回退”的情况

为什么它值得做：
- 它仍然建立在 KF + Markov + SCD + Round61 上
- 但比单纯调 alpha / Q / R 更像一个方法学创新点
- 也更符合论文表达：不是 heuristic 调参，而是用 innovation-consistency 做控制

### 优先方向 B：Round61 主线上的分层/分组反馈门控
思路：
- 把 selected states 进一步按功能分组
- 对不同组使用不同的 feedback trust gate
- 避免 xx/zz 修复与 yy/Ka_xx / lever 保护项相互牵连

为什么值得做：
- 它直接对应目前实验里最常见的问题：收益和回退耦合
- 它是对 Round61 的结构性升级，而不是 regime patch

### 暂不优先
- 再做 ultra-low SCD gating 主线化
- 再做宽范围 trust-map 形状搜索
- 混入过多新机制导致失去因果可读性

---

## 7. 标准实验链路（之后默认执行）

每次新方法研究按以下顺序：

1. 固定 `baseline noisy dataset D_ref`
2. 跑标准 KF baseline
3. 跑 Markov baseline
4. 跑 Markov + SCD
5. 跑 Round61
6. 跑新方法候选 batch
7. 做严格 ablation
8. 只在 mainline winner 形成后，再做 robustness study

---

## 8. 评价标准

### 必要指标
- 关键误差参数 pct error
- overall mean / median / max
- protected metrics set

### 必要实验
- baseline comparison
- component ablation
- same-noise reproducible rerun
- 若主线胜出，再做 cross-regime robustness

### 不再接受的结论写法
- “某个 branch 在另一个 noise regime 更好，所以它整体更好”
- “这个候选某两项很好，所以可升主线”
- “虽然 baseline 不同，但趋势看起来不错”

---

## 9. 当前推荐的下一步

### 立即执行
做一个新的主线研究批次：

**Round65 / research batch 1**
- 固定：same noisy dataset + same seed + standard KF baseline
- 比较链：
  1. KF baseline
  2. Markov
  3. Markov + SCD
  4. Round61
  5. 新方法 batch（innovation-consistency gated Round61 variants）

### 当前最值得下注的新机制
- **innovation-consistency gated Round61**

一句话定义：
> 用创新序列的一致性/可信度来调度 internalized feedback 与 SCD 的实际作用强度，而不是只靠固定启发式 trust/Q/R 设定。

如果做得好，这会更像一篇文章里的“方法点”。
