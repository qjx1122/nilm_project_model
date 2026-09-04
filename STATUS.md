# STATUS.md
## 当前目标
- **本会话任务（已完成）**：Smoke test 验证——conda `test_gpu` 环境运行 `scripts/run_smoke.py`，验证 NILM Transformer 代码链路（数据→Dataset→Transformer→Train→Val→Test→Metrics→Checkpoint）。结果：链路通过，详见 `REPORT_TEST.md`
- **下一会话待定**：需用户在 baseline / 真实 UK-DALE / 调优 中选择

## 已完成
- [x] 仓库初始化与原始代码上传（commit `ff2a340`）
- [x] 添加 Agent 会话与任务协议文档 BOOTSTRAP.md（commit `7a91470`）
- [x] 开局仪式执行：拉取 git 现状、确认远端可访问、创建 STATUS.md 续接骨架
- [x] **Smoke test 验证**：conda `test_gpu` 环境下 `python scripts/run_smoke.py` 全链路跑通（cuda, 4.74s），产物落 `reports/smoke/`；详见 `REPORT_TEST.md` 同期专题

## 进行中
- 无（本会话任务收尾中）

## 下一步（TODO）
1. 激活 conda env `test_gpu`，按 `requirements.txt` 检查并补装缺失依赖
2. 核对 `scripts/run_smoke.py` 入口与合成数据配置
3. 运行 `python scripts/run_smoke.py` 验证代码链路，收集指标与输出路径
4. 视结果决定是否进入 baseline / tuning / 真实 UK-DALE 实验

## 决策记录 / 踩坑
- gh CLI 未安装，使用 `git ls-remote origin` 验证远端访问（HTTPS 凭据可用）
- 工作目录在 worktree 分支 `nilm-project-model-ritual-4zSHFv`，远端 main 为 `7a91470`
- `checkpoints/`、`logs/`、`data/` 目录尚未创建，待实际训练/数据下载时建立
- 当前无 UK-DALE 原始/处理数据，真实实验指标需先下载数据后用 `run_real.ps1` 跑出
- **[2026-09-04]** 用户指定使用本地 conda `test_gpu` 环境运行 smoke test，而非 README §3 默认的 `transformer_nilm`；后续真实训练是否沿用此环境待用户确认
- **[2026-09-04]** Smoke test 两条 PyTorch UserWarning（nested_tensor / flash attention 未编译）非致命，不影响结果；真实训练若要消除可考虑编译 flash-attn，但非必需
- **[2026-09-04]** `run_smoke.py` 用相对路径 `configs/baseline.yaml`，必须在项目根目录执行（已验证）

## 关键文件路径
- 协议：`BOOTSTRAP.md`
- 续接：`STATUS.md`、`session/NILM_AC_session_complete.md`、`REPORT_TEST.md`、`REPORT.md`
- 入口：`README.md`、`requirements.txt`
- 脚本：`run_baseline.ps1`、`run_tuning.ps1`、`run_real.ps1`、`scripts/run_smoke.py`
- 配置：`configs/baseline.yaml`、`configs/tuning.yaml`
- 模块：`src/data.py`、`src/model.py`、`src/metrics.py`、`src/trainer.py`、`src/experiment.py`
