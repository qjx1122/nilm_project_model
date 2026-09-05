# STATUS.md

## 当前目标
- 在 Arena Linux 沙箱（CPU）中恢复开发环境，用仓库内真实 UK-DALE 数据 `data/ukdale_prepared.npz` 跑通 baseline 训练（Kettle / Seq2Point），得到第一组「真实数据」指标。

## 已完成
- [x] 开局仪式：git 状态核对（分支 `arena/01a06f16-nilm-project-model`，与 origin `d0529cf` 同步，工作区干净）
- [x] 通读代码：`src/`（data / model / trainer / metrics / experiment）与 `scripts/`（train / run_smoke / evaluate / tune / inspect_h5）
- [x] 核对数据：`data/ukdale_prepared.npz` 含 `aggregate`、`target`（float32，长度 10,344,744 ≈ UK-DALE 6s 采样约 2 年），与 `src/data.py::load_simple_npz` 期望格式一致
- [x] 按 BOOTSTRAP 模板创建本文件（STATUS.md 此前不存在）

## 进行中
- Python 依赖安装：沙箱禁用系统 pip（PEP 668），已建 `.venv`（不入库）；`download.pytorch.org` 被 TLS 拦截，改走 PyPI 默认源装 torch（含 CUDA 轮子，纯 CPU 用）

## 下一步（TODO）
1. 等 `.venv` 依赖装完，跑 `tests/test_model.py`（pytest）验证环境
2. 用真实数据跑 baseline：`.venv/bin/python scripts/train.py --config configs/baseline.yaml --data-path data/ukdale_prepared.npz --out reports/baseline_cpu`（CPU 环境，baseline 配置已限 30k 训练窗口，预计数十分钟量级，需后台跑）
3. 结果落盘：更新本文件 + `session/NILM_AC_session_complete.md`；若指标口径/结论稳定，按收尾仪式条件触发更新 `REPORT.md`
4. （可选，待与用户确认）`scripts/tune.py` 超参搜索同样在 CPU 上小规模试跑

## 决策记录 / 踩坑
- [2026-09-05] 沙箱为 Linux + CPU、无 conda：README 的 `conda` / `.ps1` 流程不适用，改用 `python3 -m venv .venv` + `pip`（`.venv` 已被快照排除，不污染仓库）。
- [2026-09-05] `pip --index-url https://download.pytorch.org/whl/cpu` 在本沙箱 TLS 握手失败（连接被重置）；改用 PyPI 默认源安装，torch 会带 CUDA 依赖包，体积大但 CPU 可正常运行。
- [2026-09-05] 本 session 分支被 Arena 平台固定在 `arena/01a06f16-nilm-project-model`，按 BOOTSTRAP「平台固定 session 分支时可直接在当前分支进行」执行，不另建 feature 分支。

## 关键文件路径
- 数据：`data/ukdale_prepared.npz`
- 训练入口：`scripts/train.py`（`--config configs/baseline.yaml --data-path ... --out ...`）
- 冒烟测试：`scripts/run_smoke.py`（合成数据，不代表 UK-DALE 结果）
- 上次冒烟产物：`reports/smoke/result.json`（合成数据，MAE≈61W，仅代码路径验证）
- 模块：`src/experiment.py`（train_experiment）、`src/trainer.py`（fit + early stop）、`src/metrics.py`（MAE/RMSE/R²/SAE/EAE/P/R/F1）
