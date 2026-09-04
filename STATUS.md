# STATUS.md

## 当前目标

- **本会话任务（已完成）**：跑完整 baseline——`configs/baseline.yaml` + `ukdale_prepared.npz`，30 epoch（patience=7，ep15 早停，best\_epoch=8），cuda 59.1s。**首个真实科学结果**：Test F1=0.792 / MAE=13.09 / R²=0.59

- **下一会话待定**：据 README §8 用 val 指标调参 / 多种子稳定性验证 / 跨建筑验证（择一）

## 已完成

- [x] 仓库初始化与原始代码上传（commit `ff2a340`）

- [x] 添加 Agent 会话与任务协议文档 BOOTSTRAP.md（commit `7a91470`）

- [x] 开局仪式执行：拉取 git 现状、确认远端可访问、创建 STATUS.md 续接骨架

- [x] **Smoke test 验证**：conda `test_gpu` 环境下 `python scripts/run_smoke.py` 全链路跑通（cuda, 4.74s），产物落 `reports/smoke/`；详见 `REPORT_TEST.md` 同期专题

- [x] **UK-DALE 数据下载与验证**：用户手动下载 `ukdale.h5`（3.19GB）到 `D:\Work\testPython\datasets\`；`inspect_h5.py` 确认合法 NILMTK HDF5，含 building1-5

- [x] **预处理脚本**：写 `scripts/prepare_ukdale.py`（纯 h5py+pandas+metadata pickle，无需 nilmtk），定位 mains=meter1/kettle=meter10，6s 重采样对齐，输出 `ukdale_prepared.npz`（82.8MB，10.3M 对齐点）

- [x] **真实数据链路验证**：`load_simple_npz` + `train_experiment`（2 epoch 小样本）端到端跑通（cuda, 1.6s），详见 `REPORT_TEST.md` 预处理专题

- [x] **补依赖**：`requirements.txt` 加 `tables>=3.9`；`run_real.ps1` 改用 `conda activate test_gpu` + 示例路径更新

- [x] **Baseline 真实训练**：`configs/baseline.yaml` + `ukdale_prepared.npz`，15 epoch 早停（best\_epoch=8），cuda 59.1s。Test: MAE=13.09/RMSE=145.64/R²=0.592/**F1=0.792**(P=0.952/R=0.678)；详见 `REPORT_TEST.md` baseline 专题

## 进行中

- 无（本会话任务收尾中）

## 下一步（TODO）

1. 据 README §8 用 val 指标调参：候选方向 (a) 增大 max\_samples\_train（30k→全量）(b) 调 dropout（0.10→0.15/0.20）(c) lr 调度（ReduceLROnPlateau）(d) 降 on\_threshold（500W→200W）提 Recall
2. 多种子稳定性验证：seed=42 单次结果，跑 seed=0/1/123 重训看 F1 方差
3. 跨建筑验证：用 building2/3/5 数据做泛化测试（需扩展 prepare\_ukdale.py 支持 building 参数——已支持 `--building`）
4. 据稳定性/调参结果决定是否沉淀进 `REPORT.md`

## 决策记录 / 踩坑

- gh CLI 未安装，使用 `git ls-remote origin` 验证远端访问（HTTPS 凭据可用）

- 工作目录在 worktree 分支 `nilm-project-model-ritual-4zSHFv`，远端 main 为 `7a91470`

- `checkpoints/`、`logs/`、`data/` 目录尚未创建，待实际训练/数据下载时建立

- **\[2026-09-04]** 用户指定使用本地 conda `test_gpu` 环境运行 smoke test，而非 README §3 默认的 `transformer_nilm`；baseline 训练沿用 `test_gpu`（已确认）

- **\[2026-09-04]** Smoke test 两条 PyTorch UserWarning（nested\_tensor / flash attention 未编译）非致命，不影响结果

- **\[2026-09-04]** `run_smoke.py` 用相对路径 `configs/baseline.yaml`，必须在项目根目录执行（已验证）

- **\[2026-09-04]** UK-DALE 下载阻塞：huggingface.co 系统层超时（GFW）、hf-mirror.com 返回 429 限流。用户决定**手动下载**，存 `D:\Work\testPython\datasets`

- **\[2026-09-04]** conda run 不传递父 shell 的 `HF_ENDPOINT` 环境变量给 Python 子进程

- **\[2026-09-04]** **关键数据流发现**：项目 `src/data.py` 不直接读 `ukdale.h5`，只支持合成/NILMbench npz/简单 npz；`scripts/train.py` 用 `load_simple_npz` 读 NPZ。项目原本缺 `ukdale.h5 → npz` 预处理脚本——已由 `scripts/prepare_ukdale.py` 补齐

- **\[2026-09-04]** 数据存 `D:\Work\testPython\datasets\ukdale.h5`；同时存在 `ukdale.h5.tgz`（2.84GB）作备份；README 默认 `D:\datasets` 与实际不符

- **\[2026-09-04]** nilmtk 装不上（不在清华镜像、GitHub 被 SSL 墙）；改用纯 h5py+pandas+metadata pickle 方案，无 nilmtk 依赖

- **\[2026-09-04]** **meter 定位**：building1 metadata pickle（`building1.attrs['metadata']`）含 `elec_meters`（site\_meter 标志）和 `appliances`。mains=meter1（site, EcoManagerWholeHouseTx），kettle=meter10（meter10 还被 food processor/sandwich maker 共享，nilmtk 标准取首个 type=kettle，存在已知噪声）

- **\[2026-09-04]** pandas.read\_hdf 读 pytables table 需装 `tables`——已在 `test_gpu` 装 `tables 3.11.1`，`requirements.txt` 已补 `tables>=3.9`

- **\[2026-09-04]** **补依赖完成**：`run_real.ps1` 已改 `conda activate test_gpu` + 示例 NPZ 路径更新

- **\[2026-09-04]** **PS 5.1 编码发现**：`run_real.ps1` UTF-8 无 BOM，PS 5.1 用 GBK 解码导致中文 Write-Host 乱码 + Parser 报字符串终止符错误；PS 7 正常。脚本逻辑正确

- **\[2026-09-04]** h5py `ds[:]` 读 pytables `meterN/table` 报 `can't open directory`，改用 `pd.read_hdf` 成功

- **\[2026-09-04]** 对齐策略：`resample('6s').mean()` + `concat(inner)` + `ffill(5).dropna()`；11.3M→10.3M 对齐点

- **\[2026-09-04]** **baseline 训练**：15 epoch 早停（patience=7，best\_epoch=8），cuda 59.1s。Test F1=0.792/P=0.952/R=0.678。val/test F1 gap 大（0.93→0.79）——test 段更难或分布漂移；Precision>Recall 模型偏保守漏报

- **\[2026-09-04]** baseline 用 `conda run -n test_gpu python scripts/train.py`（非 run\_real.ps1，因 conda activate 在非交互 shell 可能失败）；run\_real.ps1 在交互式 PS 可用

## 关键文件路径

- 协议：`BOOTSTRAP.md`

- 续接：`STATUS.md`、`session/NILM_AC_session_complete.md`、`REPORT_TEST.md`、`REPORT.md`

- 入口：`README.md`、`requirements.txt`

- 脚本：`run_baseline.ps1`、`run_tuning.ps1`、`run_real.ps1`、`scripts/run_smoke.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/inspect_h5.py`、`scripts/prepare_ukdale.py`

- 配置：`configs/baseline.yaml`、`configs/tuning.yaml`

- 模块：`src/data.py`、`src/model.py`、`src/metrics.py`、`src/trainer.py`、`src/experiment.py`

- 数据：`D:\Work\testPython\datasets\ukdale.h5`（3.19GB，NILMTK HDF5，5 buildings）

- 预处理：`D:\Work\testPython\datasets\ukdale_prepared.npz`（82.8MB，10.3M 对齐点，aggregate+target）

- 预处理脚本：`scripts/prepare_ukdale.py`

- Smoke 产物：`reports/smoke/{best.pt, history.json, result.json}`

- Baseline 产物：`reports/baseline/{best.pt(280KB, 未提交), history.json, result.json}`

- 验证产物：`reports/verify_real/{best.pt, history.json, result.json}`（未提交，仅磁盘留证）

