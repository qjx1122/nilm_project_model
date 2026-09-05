# STATUS.md

## 当前目标
- [x] 已完成（2026-09-05 本 session）：缩短版真实数据 baseline —— UK-DALE kettle，train 10000 窗口 × 10 epochs，CPU，结果目录 `reports/ukdale_baseline_cpu_short/` 已整体入库。
- [ ] 下一任务（待用户确认优先级）：全量 baseline（`configs/baseline.yaml`，30k 窗口 × 30 epochs，CPU 预计 ~85 分钟）→ 若指标稳定，写入 `REPORT.md` 作为首个「真实数据稳定结论」。

## 已完成
- [x] 开局仪式：git 核对（分支 `arena/01a06f16-nilm-project-model`，与 origin 同步）、按模板创建 STATUS.md
- [x] 环境恢复（Linux 沙箱）：`.venv` + PyPI 安装 torch 2.14.0/依赖，`pytest tests/` 通过
- [x] 数据核验：`data/ukdale_prepared.npz`（aggregate/target，float32，n=10,344,744 ≈ 718 天@6s；kettle ≥500W 占比 0.63%，99.5 分位 2306W，噪声底 ≈1W）与 `load_simple_npz` 格式匹配
- [x] 真实数据 2-epoch 冒烟（管道验证，产物在 /tmp，未入库）
- [x] **缩短版真实数据 baseline**：best_epoch=9，test MAE=13.62W / RMSE=137.95W / R²=0.634 / SAE=0.215 / energy_error=-0.215 / P=0.933 / R=0.712 / F1=0.808（val 最优 MAE=5.71W）；runtime 635.6s
- [x] 专题报告追加至 `REPORT_TEST.md`；会话纪要落盘 `session/NILM_AC_session_complete.md`；README 按条件触发更新；新增 `.gitignore`

## 进行中
- （无——本 session 任务闭环，等待用户定下一优先级）

## 下一步（TODO）
1. 全量 baseline（30k 窗口 × 30 epochs，CPU ~85min）：`OMP_NUM_THREADS=2 ./.venv/bin/python scripts/train.py --config configs/baseline.yaml --data-path data/ukdale_prepared.npz --out reports/ukdale_baseline_cpu_full`
2. 指标显著优于/劣于缩短版时，分析采样密度差异（linspace 子采样 vs 连续窗口）；稳定后把结论写进 `REPORT.md`
3. 清理仓库遗留：`src/__pycache__/*.pyc`（上一 commit 误入库的 Windows 编译缓存）建议 `git rm -r --cached` 移除，`.gitignore` 已就位
4. （可选）`scripts/tune.py` 真实数据小规模调参试点（CPU 预算允许时）；`run_real.ps1` 可加默认路径 `data/ukdale_prepared.npz`（本环境无法测 PowerShell，留给用户或下次确认）
5. （可选）Test 集当前是 718 天中后 15% 一段的 linspace 子采样，建议正式版补一份「连续覆盖 + 按事件对齐」的评测口径，对齐 NILMbench 可比性

## 决策记录 / 踩坑
- [2026-09-05] 沙箱为 Linux + 2 核 CPU + 3GB 内存：README 的 `conda`/`.ps1` 流程不适用，改用 `python3 -m venv .venv` + `pip`（`.venv` 已加 `.gitignore`，避免 `git add -A` 误收 3.5GB 依赖）。
- [2026-09-05] `download.pytorch.org` 在本沙箱 TLS 被拦截 → 改 PyPI 默认源装 torch 2.14.0+cu130；**其 import 依赖 nvidia-* 动态库，不可卸载精简**（删了会 ImportError，需按 pin 版本逐个装回）。
- [2026-09-05] python 非 tty 时 stdout 块缓冲，`tee` 看不到实时 epoch 日志；用 `best.pt` 的 mtime 当进度心跳可观察训练推进。
- [2026-09-05] 缩短版实验口径说明：`max_samples_train=10000` 是从 70% 时间段（≈7.2M 中心点）linspace 均匀子采样，窗口间隔 ~724 点（≈72 分钟），非连续覆盖；kettle 事件短，子采样导致部分 ON 事件漏采 → energy_error=-21%、recall 0.71 与此吻合。对比 NILMbench 连续窗口口径时需注意此差异。
- [2026-09-05] 本 session 分支被 Arena 平台固定在 `arena/01a06f16-nilm-project-model`，按 BOOTSTRAP「平台固定 session 分支时可直接在当前分支进行」执行。
- [2026-09-05] 结果入库策略经用户确认：**结果目录全部提交（含 best.pt，286KB 量级）**；大文件（npz 83MB）此前已入库，不再新增大产物。

## 关键文件路径
- 数据：`data/ukdale_prepared.npz`（aggregate+target，kettle，6s）
- 本次实验产物：`reports/ukdale_baseline_cpu_short/`（config.yaml / train.log / history.json / result.json / best.pt，已入库）
- 全量配置（下一步用）：`configs/baseline.yaml`；调参搜索：`configs/tuning.yaml`
- 训练入口：`scripts/train.py`；查看结果：`scripts/evaluate.py --run-dir reports/ukdale_baseline_cpu_short`
- 模块：`src/experiment.py`（train_experiment）、`src/trainer.py`（fit+早停，按 val MAE 选 best）、`src/data.py`（linspace 子采样在 build_splits）、`src/metrics.py`
- 台账：`session/NILM_AC_session_complete.md`（纪要）、`REPORT_TEST.md`（专题）、`README.md`（环境与命令，本 session 已更新）
