# NILM_AC 会话纪要（只追加，不新建文件）

> 协议见 `BOOTSTRAP.md`。每 session 一条，倒序或正序均可，当前采用**正序追加**。

## [2026-09-05] 会话纪要
- 目标：按 BOOTSTRAP v2.0 执行开局/收尾仪式；经用户确认，本 session 任务定为「真实 UK-DALE 数据的缩短版 baseline（CPU）」并将结果目录整体入库。
- 完成项：
  - 开局：git 核对（`arena/01a06f16-nilm-project-model` @ origin `d0529cf`）；创建缺失的 `STATUS.md`；通读 `src/`、`scripts/`、configs 与 README。
  - 环境：Linux 沙箱无 conda/GPU，建 `.venv`，PyPI 装 torch 2.14.0+cu130 + numpy/pandas/sklearn/matplotlib/pyyaml/tqdm/h5py/pytest；`pytest tests/` 通过。
  - 数据核验：`data/ukdale_prepared.npz` = aggregate+target（float32，n=10,344,744，≈718 天@6s），kettle ≥500W 占比 0.63%，与 `src/data.py::load_simple_npz` 约定一致；2-epoch 冒烟通过。
  - 实验：缩短版 baseline（train 10k 窗口 × 10 epochs，其余口径同 `configs/baseline.yaml`）→ test MAE 13.62W / R² 0.634 / SAE 0.215 / F1 0.808（P 0.933 / R 0.712）；runtime 635.6s（CPU 2 核）。
  - 落盘：`STATUS.md` 终态、本纪要、`REPORT_TEST.md` 首个实验专题、README 条件触发更新、新增 `.gitignore`（挡住 .venv/__pycache__）。
- 关键决策：
  - `download.pytorch.org` 被沙箱 TLS 拦截 → 走 PyPI 默认源；nvidia-* 依赖不可删（torch import 需要）。
  - 缩短版口径明确记录「linspace 子采样、非连续覆盖」，与 NILMbench 连续窗口对比时须注明差异（详见踩坑）。
  - `REPORT.md` 暂不写入：缩短版单跑不算「重大实验结论稳定」，待全量 baseline 确认后进入。
  - 结果目录全部入库（含 best.pt 286KB，用户选择）；不再新增 >10MB 产物。
- 未决问题：
  - 全量 baseline（30k × 30 epochs，CPU ~85min）是否本分支继续跑？
  - `src/__pycache__/*.pyc` 系上一 commit 误入库，建议 `git rm -r --cached` 清掉（待确认）。
  - `run_real.ps1` 未加仓库内数据默认路径（沙箱无法测试 PowerShell，留待确认）。
- 相关文件/分支：分支 `arena/01a06f16-nilm-project-model`（自 `d0529cf`）；产物 `reports/ukdale_baseline_cpu_short/`；台账 `STATUS.md`、`REPORT_TEST.md`、`README.md`（§2/§3/§6/§7/§8）。
