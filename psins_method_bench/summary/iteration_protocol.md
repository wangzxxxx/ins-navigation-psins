# PSINS 实验迭代协议

这份协议不是“写给用户看的总结”，而是我自己之后必须遵守的 **实验闭环**。

目标：把每一轮 PSINS 实验都纳入固定流程：

> 现状锚定 → 假设生成 → 候选设计 → 受控 probe → 结果分类 → 学习提炼 → 下一轮实验生成 → formalize / 放弃

---

## -1. 全局硬规则：baseline / dataset / noise 一致性

从现在开始，任何“方法优劣”结论都必须满足：

1. 对比里必须包含 **标准 KF baseline**
2. baseline 必须运行在 **有噪声的数据** 上
3. 同一批方法比较时，必须使用 **同一份 noisy dataset / 同一 noise strength / 同一 seed**
4. 如果改变 noise strength，那是在做 **regime study**，不是在做主线方法优劣比较
5. `mainline winner` 只能从 **同噪声、同数据、同 seed** 的严格对比里产生

这是之后所有 PSINS 轮次的上层实验法约束。

## 0. 轮次开始前：先做状态锚定

每一轮开始前，必须先写清这 5 件事：

1. **当前主线 best 是谁**
   - 默认 1x mainline best
   - ultra-low / 特殊噪声分支 best
2. **当前 round 的目标是什么**
   - 修某个痛点？
   - 换机制？
   - 验证某个结构性猜想？
3. **不允许动什么**
   - 哪些保护项必须守住
   - 哪些机制本轮明确不碰
4. **这轮允许动的旋钮集合**
   - 只允许 1–2 类旋钮
   - 禁止大杂烩
5. **成功标准**
   - 什么叫 clean win
   - 什么叫 partial signal
   - 什么叫直接放弃

如果这 5 件事说不清，不能开新 round。

---

## 1. 新实验必须来自“上一轮结果差分”

新实验不能拍脑袋上。

每个新 round 的候选，必须明确来自以下至少一种来源：

### A. 痛点修补型
- 上一轮某个指标明显好，但 1–2 个保护项回退
- 新实验目标：只修这几个回退点，不动主体收益

### B. 结构换向型
- 当前主线已进入 micro-best 平台期
- 继续同方向微调收益过小
- 新实验目标：换一个机制维度，而不是继续在旧维度抠小数点

### C. 失败反推型
- 某次失败不是纯噪声，而是暴露出一个明确 lesson
- 新实验目标：把 lesson 变成受控反向验证

### D. 分支嫁接型
- 某条支线在局部 regime 有价值
- 新实验目标：验证它能否以“更窄、更受控”的方式嫁接回主线

---

## 2. 每一轮实验必须先定义“主语”

每个 round 开始时，先把自己归类成下面 4 种之一：

1. **mainline refine**
   - 在当前正式最佳主线上做 ultra-small 微调
2. **repair branch**
   - 修复某个已知回退点
3. **new mechanism probe**
   - 换一个机制维度做探索
4. **regime branch**
   - 面向特殊噪声/特殊场景的分支验证

不同类型，评估标准不同。

- `mainline refine`：必须极其严格，优先 no-regression
- `repair branch`：允许局部取舍，但必须明确修谁、代价是什么
- `new mechanism probe`：允许 first batch 没有赢家，但必须带回“机制层学习”
- `regime branch`：禁止冒充 mainline winner

---

## 3. 候选设计规则

### 3.1 批次规模
- 首轮 probe 默认 **3–5 个候选**
- 一轮只测一个主要方向
- 不做“大搜索 + 大混搭”

### 3.2 旋钮范围
- 候选只允许在基线附近做 **窄幅 deterministic 微调**
- 每个候选必须能用一句话描述
- 每个候选必须回答：
  - 它改了什么
  - 为什么改这个
  - 预期修什么
  - 可能伤到什么

### 3.3 禁止事项
- 禁止把多个新机制同时大幅叠加
- 禁止既改 feedback 结构、又改 SCD cadence、又改 noise regime，还想从结果里读因果
- 禁止用“范围太大导致看不懂”的实验浪费轮次

---

## 4. Probe 运行协议

每轮 probe 必须输出：

1. `candidate_json`
   - 每个候选的 patch、动机、预期
2. `probe_summary.json`
   - 每个候选相对基线的 delta
   - selection score
   - penalties
   - note
3. `report.md`
   - 人类可读摘要

同时，每轮必须固定：
- 基线 candidate 名称
- 数据集 / 噪声 regime
- seed
- 保护项列表
- scoring 规则

---

## 5. 结果分类：不是只有“赢/输”两种

每轮结果必须分成以下 4 类之一：

### A. clean win
满足：
- 主目标改善
- protected metrics 无明显回退
- 可 formalize

### B. partial signal
满足：
- 某些指标改善明显
- 但保护项回退，不能直接升格
- 可作为下一轮 repair 的起点

### C. no useful signal
满足：
- 改善很弱，或者方向不稳定
- 不足以支持下一轮继续沿这条线走

### D. negative lesson
满足：
- 清楚证明某种方向不值得再走
- 这是有效学习，不算白跑

**每轮必须明确写出自己属于哪一类。**

---

## 6. 学习提炼：每轮都要产出“可复用结论”

每次实验结束后，不能只报 delta，必须额外提炼三层结论：

### 6.1 数值层
- 哪些指标涨/跌了多少

### 6.2 机制层
- 这说明哪个机制在起作用
- 是 score trust 在主导，还是 covariance schedule 在主导
- 是修了目标项，还是只是误差重分配

### 6.3 流程层
- 这条线是否值得继续
- 下一轮应该更窄、换方向，还是直接停掉

---

## 7. 下一轮实验生成规则

下一轮不应该凭感觉，而应该从当前结果自动落出：

### 如果是 clean win
- formalize
- 再从它出发做 ultra-small refinement

### 如果是 partial signal
- 固定住带来收益的部分
- 只针对回退项做 repair round

### 如果是 no useful signal
- 停掉这条线
- 回到上一稳定主线，换机制方向

### 如果是 negative lesson
- 把 lesson 写进 memory / protocol
- 明确列入“不再重复”的方向

---

## 8. formalize gate

只有满足以下条件，才能从 probe 升为正式 round 方法文件：

1. 有清晰基线
2. 有确定 winner
3. winner 是可解释的
4. 关键保护项没有不可接受回退
5. 有正式 param_errors 结果文件
6. 能用一句话说明“为什么它比上一轮更好”

不满足这些条件，就只保留 probe，不强行升格。

---

## 9. 当前 durable lessons（来自最近 PSINS 主线）

1. **先固化当前 best，再做小范围微调**，不要反复大搜索。
2. **不能把已经在外部 profile 中验证过的强增益，直接按类似幅度分摊回多轮迭代内部**。
3. `mainline best` 和 `special regime best` 必须分开，不允许混写。
4. `new mechanism probe` 的第一轮允许无 winner，但必须带回清晰学习。
5. “结果更好”不能只看单点指标，必须看 protected set 和是否可持续 formalize。

---

## 10. 我之后的默认执行方式

以后做 PSINS 新 round，默认按这个顺序：

1. 先写 round 类型和目标
2. 先列 allowed knobs / protected metrics
3. 先建 candidate_json
4. 再跑 deterministic probe
5. 再做结果分类
6. 再从结果反推下一轮
7. 只有满足 formalize gate 才升正式 round

这份协议是我自己的执行约束，不是装饰文档。
