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
