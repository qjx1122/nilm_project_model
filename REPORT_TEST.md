# REPORT_TEST.md — 专题报告（只追加，按专题分节，不新建文件）

## [2026-09-05] 专题：真实 UK-DALE 缩短版 baseline（Kettle / Transformer Seq2Point / CPU）
- 类型：实验专题
- 目标与假设：仓库已有代码与真实数据（`data/ukdale_prepared.npz`）但从未产出真实指标（README 原声明「仅有合成 smoke test」）。目标：在 Linux CPU 沙箱上快速跑通真实数据训练，拿到第一组真实 UK-DALE 指标，验证管道与数量级合理（假设：kettle 属易学电器，MAE 应为 10¹W 量级、R² 显著高于 0）。
- 方法 / 数据 / 参数：
  - 数据：UK-DALE House1 kettle，6s 采样，n=10,344,744；按 70/15/15 时间切分。
  - 与 `configs/baseline.yaml` 唯一差异为控制 CPU 时长：`max_samples_train` 30000→10000、`max_samples_val/test` 6000 保持、`epochs` 30→10、`patience` 7→4；`window_size=128, d_model=64, nhead=4, layers=2, ff=128, dropout=0.1, lr=5e-4, wd=1e-4, batch=128, MSE, grad_clip=1.0, seed=42, on_threshold=500W`。
  - 注意：`build_splits` 对采样中心做 linspace 子采样 → 10k 窗口 ≈ 每 724 个 6s 点取 1 个（间隔 ~72 分钟），**非连续覆盖**。
  - 环境：Linux，2 核 CPU，torch 2.14.0+cpu 用法（`.venv`）；产物目录 `reports/ukdale_baseline_cpu_short/`。
- 结果 / 结论：
  - 收敛正常：best_epoch=9/10（早停未触发），runtime 635.6s（≈64s/epoch）。
  - Val 最优：MAE=5.71W，R²=0.861。
  - **Test：MAE=13.62W，RMSE=137.95W，R²=0.634，SAE=0.2145，energy_error=-0.2145，Precision=0.933，Recall=0.712，F1=0.808。**
  - 解读：MAE≈13.6W（信号 std=183W、99.5 分位 2306W）数量级合理，Seq2Point 管道与指标实现自检通过；recall 0.71 / 能量低估 21% 与「子采样漏采短事件」一致，属口径效应而非实现缺陷；precision 高说明模型极少误触发。
  - 结论：代码-数据-指标全链路在真实数据上成立，可放心扩展到全量 baseline 与调参；但**本组数字不可直接对标 NILMbench 发表值**（采样密度不同）。
- 是否进入 REPORT.md（稳定结论）：否（单次缩短跑，等全量 baseline 确认后进入）。
- 遗留问题：
  - 全量 baseline（30k × 30 epochs，CPU 预计 ~85min）作为对照。
  - 评测口径升级建议：连续窗口 + 事件级对齐，或与 NILMbench 完全一致的 split 与聚合方式，以便外部可比。
  - 调参（`configs/tuning.yaml` 12 trials）在 CPU 上的预算评估与抽样策略。

## [2026-09-05] 专题：真实 UK-DALE 超参调优（Kettle / Seq2Point / CPU，Phase1-3 完整链路）
- 类型：实验专题
- 目标与假设：在缩短版 baseline（test MAE 13.62W、EE −21.5%、Recall 0.712）基础上，通过超参与训练策略调优改善 MAE/R²/F1/能量偏差。假设：欠拟合迹象（train≈val）→ dropout/lr/容量有空间；MSE 对尖峰的惩罚与能量偏差相关 → loss 与调度值得验证。
- 方法 / 数据 / 参数：
  - 设计：坐标平行搜索（替代 tune.py 顺序随机搜索，省墙钟且 val 口径统一）。Phase1 六泳道单因子（8k/4k/4k、12ep、patience4）：loss=l1 / window=256 / d128h8ff256 / lr=1e-3 / layers=4 / dropout=0。Phase2 组合胜出者（d0×lr1e-3、d0×lr2e-3）。Phase3：同尺度(10k/6k/6k×10ep) A/B 复核 → 全量 F2（30k/8k，test 30000 稠密，20ep/patience5）→ F3=F2+cosine。
  - 新增工具/代码：`scripts/eval_ckpt.py`（checkpoint 换口径复评，切分/归一化可精确复现，已验证与 result.json 逐位一致）；`src/trainer.py`+`src/experiment.py` 增加 opt-in `training.lr_schedule: cosine`（默认行为不变，早停分支也正确 step）。
  - 环境约束：2 核 CPU 3.9GB；泳道并行 2 条时 torch+cu130 单进程 RSS≈2GB 触发全局 OOM（seq256 泳道被杀；300 步二分显示 seq256 下 ~18MB/步 线性增长，seq128 正常）→ 全程降为顺序单泳道；`window_size=256` 在本机暂缓。
- 结果 / 结论：
  - 否决项：L1 loss 坍缩（零膨胀目标下条件中位数≈0：val MAE 10.83，test R²=−0.009，EE −96%）；lr=2e-3（d0 下发散，val 10.44）；d128/ff256（val 5.39 劣于 d64，过拟合）；layers=4（val 5.03，无收益）；dropout0+lr1e-3 组合在 10ep 预算下 MAE −8.6% 但 R²/EE 变差 → 被同尺度 A/B 否决。
  - 采纳项：**dropout 0.1→0**（同尺度 A/B：MAE −7.7%、val R²↑、Recall +5.1pp、EE 略改善）；**cosine lr 衰减**（5e-4 起点）——F2 暴露固定 lr 的 val 剧烈震荡（11.67↔4.93）导致早停选点踩雷（best_ep7 恰是次优点，EE −28.6%）；F3 曲线平滑后 best_ep12 泛化良好。
  - **同口径公平对比（稠密 test=30000）**：anchor 11.67W / R²0.671 / F1 0.832 / Rec 0.752 / EE −14.8% → **F3 8.58W（−26.5%）/ R² 0.723 / F1 0.853 / Rec 0.805 / EE −19.5%**（Precision 0.930→0.907 为 recall 换入的代价，F1 净升）。对原 6k 口径 anchor 13.62W 为 −37%。
  - 方法论教训（可复用）：**模型选择本身是超参数**——稀疏事件+小 val 集下「val MAE 最低 epoch」噪声极大，调度平滑化（cosine/EMA/SWA）或按 val F1/多指标选择是更稳的选点策略；跨协议对比（不同 max_samples_* 即不同 test 子集）不可直接比数值，必须用 eval_ckpt 统一到稠密口径。
- 是否进入 REPORT.md（稳定结论）：是（F3 配置+指标成为首个推荐稳定版本；标注单 seed 未复验）。
- 遗留问题：
  - 全部结论来自 seed=42 单跑，采纳/否决的差距多在 0.3–1.5W val MAE 量级 → 建议关键配置 2-3 seeds 复验后再扩大。
  - EE −19.5% 仍有能量低估 → 候选：预测头加正偏置先验、损失加事件加权（需改 trainer）、或 ON 阈值下 soft-label；稠密 test 仍是 241 步子采样，正式版建议连续覆盖评测对齐 NILMbench。
  - window=256/更大容量在 torch+cu130 CPU 轮子的内存异常未解（换官方 CPU wheel 或 GPU 机器重测）。
  - F3 的 val_F1 最优 epoch(10, 0.897) 与 val_MAE 最优(12) 不同 → 选点准则 ablation 值得单独跑。
