# STATUS.md

## 当前目标

- **本会话任务（已完成）**：补依赖——`requirements.txt` 加 `tables>=3.9`（pandas.read\_hdf 读 pytables table 必需，已在 `test_gpu` 装 3.11.1）；`run_real.ps1` 改用 `conda activate test_gpu`（替代 README 默认 `transformer_nilm`）+ 示例路径更新为实际 `D:\Work\testPython\datasets\ukdale_prepared.npz`

- **下一会话待定**：跑完整 baseline（`run_real.ps1`，30 epoch，30k 样本，按 README §8 据 val 调参）出真实 Kettle 指标

## 已完成

- [x] 仓库初始化与原始代码上传（commit `ff2a340`）

- [x] 添加 Agent 会话与任务协议文档 BOOTSTRAP.md（commit `7a91470`）

- [x] 开局仪式执行：拉取 git 现状、确认远端可访问、创建 STATUS.md 续接骨架

- [x] **Smoke test 验证**：conda `test_gpu` 环境下 `python scripts/run_smoke.py` 全链路跑通（cuda, 4.74s），产物落 `reports/smoke/`；详见 `REPORT_TEST.md` 同期专题

- [x] **UK-DALE 数据下载与验证**：用户手动下载 `ukdale.h5`（3.19GB）到 `D:\Work\testPython\datasets\`；`inspect_h5.py` 确认合法 NILMTK HDF5，含 building1-5

- [x] **预处理脚本**：写 `scripts/prepare_ukdale.py`（纯 h5py+pandas+metadata pickle，无需 nilmtk），定位 mains=meter1/kettle=meter10，6s 重采样对齐，输出 `ukdale_prepared.npz`（82.8MB，10.3M 对齐点）

- [x] **真实数据链路验证**：`load_simple_npz` + `train_experiment`（2 epoch 小样本）端到端跑通（cuda, 1.6s），详见 `REPORT_TEST.md` 预处理专题

- [x] **补依赖**：`requirements.txt` 加 `tables>=3.9`；`run_real.ps1` 改用 `conda activate test_gpu` + 示例路径更新

## 进行中

- 无（本会话任务收尾中）

## 下一步（TODO）

1. 跑完整 baseline：`$env:UKDALE_PREPARED_NPZ="D:\Work\testPython\datasets\ukdale_prepared.npz"; .\run_real.ps1`（已改用 `test_gpu` env，30 epoch，30k 样本，按 README §8 据 val 调参）
2. 真实 Kettle 指标产出后，视稳定程度决定是否沉淀进 `REPORT.md`

## 决策记录 / 踩坑

- gh CLI 未安装，使用 `git ls-remote origin` 验证远端访问（HTTPS 凭据可用）

- 工作目录在 worktree 分支 `nilm-project-model-ritual-4zSHFv`，远端 main 为 `7a91470`

- `checkpoints/`、`logs/`、`data/` 目录尚未创建，待实际训练/数据下载时建立

- **\[2026-09-04]** 用户指定使用本地 conda `test_gpu` 环境运行 smoke test，而非 README §3 默认的 `transformer_nilm`；后续真实训练是否沿用此环境待用户确认

- **\[2026-09-04]** Smoke test 两条 PyTorch UserWarning（nested\_tensor / flash attention 未编译）非致命，不影响结果；真实训练若要消除可考虑编译 flash-attn，但非必需

- **\[2026-09-04]** `run_smoke.py` 用相对路径 `configs/baseline.yaml`，必须在项目根目录执行（已验证）

- **\[2026-09-04]** UK-DALE 下载阻塞：huggingface.co 系统层超时（GFW）、hf-mirror.com 返回 429 限流。用户决定**手动下载**，存 `D:\Work\testPython\datasets`（sandbox 可写区内，无需禁 sandbox 写入；但 README 默认 `D:\datasets`，后续训练脚本若硬编码该路径需调整）

- **\[2026-09-04]** conda run 不传递父 shell 的 `HF_ENDPOINT` 环境变量给 Python 子进程；huggingface\_hub 的 endpoint 在 import 时读取，os.environ 后置设置无效

- **\[2026-09-04]** **关键数据流发现**：项目 `src/data.py` 不直接读 `ukdale.h5`，只支持合成信号 / NILMbench npz / 简单 npz；`scripts/train.py` 用 `load_simple_npz` 读 NPZ（须含 `aggregate`+`target` 数组）；`run_real.ps1` 期望用户先备好 `ukdale_prepared.npz` 并通过 `$env:UKDALE_PREPARED_NPZ` 传入。项目原本缺 `ukdale.h5 → npz` 预处理脚本——已由 `scripts/prepare_ukdale.py` 补齐

- **\[2026-09-04]** 数据存 `D:\Work\testPython\datasets\ukdale.h5`（用户选定，sandbox 可写区内）；同时存在 `ukdale.h5.tgz`（2.84GB 原始压缩包）作备份；README 默认 `D:\datasets` 与实际不符，需在跑实验时调整路径或更新 README

- **\[2026-09-04]** nilmtk 装不上（不在清华镜像、GitHub 被 SSL 墙）；改用纯 h5py+pandas+metadata pickle 方案，无新 nilmtk 依赖

- **\[2026-09-04]** **meter 定位**：building1 metadata pickle（`building1.attrs['metadata']`）含 `elec_meters`（site\_meter 标志）和 `appliances`（type+meters 映射）。mains=meter1（site, EcoManagerWholeHouseTx），kettle=meter10。注：meter10 还被 food processor / toasted sandwich maker 共享（共享插座），nilmtk 标准取首个 type=kettle，存在已知噪声

- **\[2026-09-04]** pandas.read\_hdf 读 pytables table 需装 `tables`（pytables）——已在 `test_gpu` 装 `tables 3.11.1`（依赖 blosc2/numexpr 等）；`requirements.txt` 应补 `tables` 依赖

- **\[2026-09-04]** **补依赖完成**：`requirements.txt` 已加 `tables>=3.9`；`run_real.ps1` 已改 `conda activate test_gpu`（替代 `transformer_nilm`）+ 示例 NPZ 路径更新为 `D:\Work\testPython\datasets\ukdale_prepared.npz`

- **\[2026-09-04]** **PS 5.1 编码发现**：`run_real.ps1` 是 UTF-8 无 BOM 文件，PowerShell 5.1 用 GBK 解码导致中文 Write-Host 串乱码 + Parser 报"字符串终止符"错误；PS 7 能正确处理 UTF-8 无 BOM。脚本逻辑正确，用户若在 PS 5.1 跑遇乱码可考虑：(a) 用 PS 7；(b) 重存为 UTF-8 BOM；(c) Write-Host 改英文

- **\[2026-09-04]** h5py `ds[:]` 读 pytables `meterN/table` 报 `can't open directory`，改用 `pd.read_hdf`（pytables 后端）成功

- **\[2026-09-04]** 对齐策略：mains+target 各自 `resample('6s').mean()` 后 `pd.concat(join='inner')` + `ffill(limit=5).dropna()`；inner-join 11.3M→10.3M 对齐点（去掉无重叠段 + 短 gap）

- **\[2026-09-04]** 真实数据小样本验证指标差（Test MAE=38.2, R²=0.11, F1=0）属预期——仅 2 epoch + 4000 样本，目的是验证链路非科学指标；正式 baseline 需 30 epoch + 30k 样本

- **\[2026-09-04]** STATUS.md 被 markdown 渲染器规范化（条目间空行 + 转义反斜杠）导致 Edit 字符串失配，改用 Write 整体重写维护

## 关键文件路径

- 协议：`BOOTSTRAP.md`

- 续接：`STATUS.md`、`session/NILM_AC_session_complete.md`、`REPORT_TEST.md`、`REPORT.md`

- 入口：`README.md`、`requirements.txt`

- 脚本：`run_baseline.ps1`、`run_tuning.ps1`、`run_real.ps1`、`scripts/run_smoke.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/inspect_h5.py`、`scripts/prepare_ukdale.py`

- 配置：`configs/baseline.yaml`、`configs/tuning.yaml`

- 模块：`src/data.py`、`src/model.py`、`src/metrics.py`、`src/trainer.py`、`src/experiment.py`

- 数据：`D:\Work\testPython\datasets\ukdale.h5`（3.19GB，用户手动下载，NILMTK HDF5，5 buildings）

- 预处理：`D:\Work\testPython\datasets\ukdale_prepared.npz`（82.8MB，10.3M 对齐点，aggregate+target）

- 预处理脚本：`scripts/prepare_ukdale.py`

- Smoke 产物：`reports/smoke/{best.pt, history.json, result.json}`

- 验证产物：`reports/verify_real/{best.pt, history.json, result.json}`（未提交，仅磁盘留证）

