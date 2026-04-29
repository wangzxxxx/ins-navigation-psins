# INS Navigation & Calibration Research Codebase

惯性导航（Inertial Navigation System, INS）与标定算法研究代码库，包含 PSINS（Python-Based Inertial Navigation System）的核心实现、标定方法对比实验、以及基于 LLM 辅助的标定优化策略。

## 📂 目录结构

```
ins-code/
├── core/                    # PSINS 核心算法实现
│   ├── imu_utils.py        # IMU 数据预处理与仿真
│   ├── kf_utils.py         # 卡尔曼滤波核心实现
│   ├── math_utils.py        # 导航数学工具（姿态、坐标变换）
│   ├── nav_utils.py        # 导航解算与误差建模
│   ├── shadow_kf.py        # Shadow Kalman Filter 实现
│   ├── shadow_manager.py    # 多模型管理框架
│   └── calibration/         # 标定相关算法
│
├── benchmark/              # PSINS 方法基准测试（原 psins_method_bench）
│   ├── methods/           # 各轮实验方法实现（R01-R69）
│   ├── results/            # 实验结果（参数误差、收敛曲线等）
│   ├── scripts/           # 实验脚本与探针工具
│   └── summary/           # 研究总结与迭代记录
│
├── scripts/                # 独立脚本工具
│   ├── alignment/         # 对准算法脚本（18/20/24/42-state）
│   ├── navigation/        # 导航仿真与静态测试
│   ├── visualization/     # 结果可视化（SVG/PDF 生成）
│   └── utils/            # 辅助工具脚本
│
├── results/               # 独立实验结果（12-state、Markov 系列等）
│
└── docs/                 # 文档与说明（可选）
```

## 🔬 核心研究方向

### 1. 标定方法对比与优化
- **状态维度**：18-state → 20-state → 24-state → 42-state → 55-state
- **关键方法**：Markov 链建模、SCD（Strongly Connected Component Decomposition）、Shadow KF、自适应标定
- **实验轮次**：R01–R69，覆盖噪声鲁棒性、可观测性分析、参数收敛性

### 2. 双轴旋转对准策略
- 静态对准、SCD 辅助对准、Hybrid 方法
- 噪声水平：`0.03×` → `3.0×` 范围测试
- 重复对准精度：航向误差 < 20 arcsec（1σ）

### 3. LLM 辅助标定优化
- 利用大语言模型生成标定策略
- 可观测性引导的参数搜索
- 迭代反馈与细化框架

## 🚀 快速开始

### 依赖安装
```bash
pip install numpy scipy matplotlib seaborn
# 如需 RL 相关实验
pip install torch tensorboard
```

### 运行基准测试
```bash
cd benchmark/
python run_all_methods.py --noise 0.08 --mc 50
```

### 运行对准仿真
```bash
cd scripts/alignment/
python generate_ch4_alignment_convergence_fig_2026-03-31.py
```

## 📊 主要结果

| 方法 | 状态维度 | 噪声水平 | 航向误差 (1σ) | 备注 |
|--------|----------|----------|----------------|------|
| 18-state (G1) | 18 | 0.08× | ~60 arcsec | 基线方法 |
| 20-state (G2) | 20 | 0.08× | ~45 arcsec | + 18-state SCD |
| 24-state (G3+G4) | 24 | 0.08× | ~30 arcsec | 双轴完整建模 |
| 42-state Markov | 42 | 0.08× | ~22 arcsec | + Markov 链建模 |
| 42-state SCD | 42 | 0.08× | **~17 arcsec** | 纯 SCD 优化 |

> 详细结果见 `benchmark/results/` 与 `benchmark/summary/`

## 📚 相关论文与参考

1. **SkVM (2026)**：Skill Virtual Machine for Cross-Model Skill Execution  
   arXiv:2604.03088v2 | https://arxiv.org/abs/2604.03088

2. **PSINS (Python)**：Python-Based Inertial Navigation System  
   核心实现见 `core/` 目录

## 📝 提交信息

- **提交邮箱**：wrichu0126@gmail.com
- **维护者**：wrichu0126
- **最后更新**：2026-04-29

## 👥 贡献者

| 贡献者 | GitHub 账号 | 邮箱 | 角色 |
|----------|-------------|------|------|
| wrichu0126 | wrichu0126 | wrichu0126@gmail.com | 项目维护者、算法设计 |
| wangzxxxx | wangzxxxx | — | 代码托管、仓库管理 |

## 📄 License

未经书面许可，不得用于商业用途。学术研究请注明出处。
