# STATUS.md

## 当前目标

- **本会话任务（已完成）**：下载 UK-DALE 数据集并验证。`ukdale.h5`（3.19GB，5 buildings）已就位 `D:\Work\testPython\datasets\ukdale.h5`

- **下一会话待定**：需先写 `ukdale.h5 → ukdale_prepared.npz` 预处理脚本（项目缺此环节），再跑 baseline

## 已完成

- [x] 仓库初始化与原始代码上传（commit `ff2a340`）

- [x] 添加 Agent 会话与任务协议文档 BOOTSTRAP.md（commit `7a91470`）

- [x] 开局仪式执行：拉取 git 现状、确认远端可访问、创建 STATUS.md 续接骨架

- [x] **Smoke test 验证**：conda `test_gpu` 环境下 `python scripts/run_smoke.py` 全链路跑通（cuda, 4.74s），产物落 `reports/smoke/`；详见 `REPORT_TEST.md` 同期专题

- [x] **UK-DALE 数据下载与验证**：用户手动下载 `ukdale.h5`（3.19GB）到 `D:\Work\testPython\datasets\`；`inspect_h5.py` 确认合法 NILMTK HDF5，含 building1-5

## 进行中

- 无（本会话任务收尾中）

## 下一步（TODO）

1. **写预处理脚本** `scripts/prepare_ukdale.py`：从 `D:\Work\testPython\datasets\ukdale.h5` 提取 House 1 的 aggregate(mains) + kettle(appliance) 功率序列，按 README §9 时序划分（train 70% / val 15% / test 15%）保存为 `ukdale_prepared.npz`（含 `aggregate`、`target` 两数组，符合 `src/data.py::load_simple_npz` 期望）
2. 预处理脚本依赖 nilmtk（未装于 `test_gpu`）——需 `pip install nilmtk` 或改用 h5py 直接读（需探查 building1 内 elec/meter/key 结构）
3. 跑 `run_real.ps1`（设 `UKDALE_PREPARED_NPZ` 指向 npz，用 `conda activate test_gpu` 替代 README 默认 `transformer_nilm`）
4. 真实 Kettle 指标产出后，视稳定程度决定是否沉淀进 `REPORT.md`

## 决策记录 / 踩坑

- gh CLI 未安装，使用 `git ls-remote origin` 验证远端访问（HTTPS 凭据可用）

- 工作目录在 worktree 分支 `nilm-project-model-ritual-4zSHFv`，远端 main 为 `7a91470`

- `checkpoints/`、`logs/`、`data/` 目录尚未创建，待实际训练/数据下载时建立

- **\[2026-09-04]** 用户指定使用本地 conda `test_gpu` 环境运行 smoke test，而非 README §3 默认的 `transformer_nilm`；后续真实训练是否沿用此环境待用户确认

- **\[2026-09-04]** Smoke test 两条 PyTorch UserWarning（nested\_tensor / flash attention 未编译）非致命，不影响结果；真实训练若要消除可考虑编译 flash-attn，但非必需

- **\[2026-09-04]** `run_smoke.py` 用相对路径 `configs/baseline.yaml`，必须在项目根目录执行（已验证）

- **\[2026-09-04]** UK-DALE 下载阻塞：huggingface.co 系统层超时（GFW）、hf-mirror.com 返回 429 限流。用户决定**手动下载**，存 `D:\Work\testPython\datasets`（sandbox 可写区内，无需禁 sandbox 写入；但 README 默认 `D:\datasets`，后续训练脚本若硬编码该路径需调整）

- **\[2026-09-04]** conda run 不传递父 shell 的 `HF_ENDPOINT` 环境变量给 Python 子进程；huggingface\_hub 的 endpoint 在 import 时读取，os.environ 后置设置无效

- **\[2026-09-04]** **关键数据流发现**：项目 `src/data.py` 不直接读 `ukdale.h5`，只支持合成信号 / NILMbench npz / 简单 npz；`scripts/train.py` 用 `load_simple_npz` 读 NPZ（须含 `aggregate`+`target` 数组）；`run_real.ps1` 期望用户先备好 `ukdale_prepared.npz` 并通过 `$env:UKDALE_PREPARED_NPZ` 传入。**项目缺少** **`ukdale.h5 → npz`** **预处理脚本**，这是跑真实实验的前置阻塞点

- **\[2026-09-04]** 数据存 `D:\Work\testPython\datasets\ukdale.h5`（用户选定，sandbox 可写区内）；同时存在 `ukdale.h5.tgz`（2.84GB 原始压缩包）作备份；README 默认 `D:\datasets` 与实际不符，需在跑实验时调整路径或更新 README

## 关键文件路径

- 协议：`BOOTSTRAP.md`

- 续接：`STATUS.md`、`session/NILM_AC_session_complete.md`、`REPORT_TEST.md`、`REPORT.md`

- 入口：`README.md`、`requirements.txt`

- 脚本：`run_baseline.ps1`、`run_tuning.ps1`、`run_real.ps1`、`scripts/run_smoke.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/inspect_h5.py`

- 配置：`configs/baseline.yaml`、`configs/tuning.yaml`

- 模块：`src/data.py`、`src/model.py`、`src/metrics.py`、`src/trainer.py`、`src/experiment.py`

- 数据：`D:\Work\testPython\datasets\ukdale.h5`（3.19GB，用户手动下载，NILMTK HDF5，5 buildings）

- Smoke 产物：`reports/smoke/{best.pt, history.json, result.json}`

