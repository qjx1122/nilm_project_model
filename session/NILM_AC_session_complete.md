# NILM_AC 会话纪要（只追加，每 session 一条）

## [2026-09-03] 会话纪要
- 目标：按 BOOTSTRAP v2.0 完成开局仪式（环境恢复/依赖安装/smoke 验证）；用户指派任务「下载 UK-DALE 并跑真实 baseline」。
- 完成项：
  1. 开局仪式：STATUS.md 首建；venv（`/home/user/venv`，torch 2.14.0 CPU）+ requirements 全装；`run_smoke.py` 通过；gh 认证 OK
  2. 数据通路侦察：沙箱 egress 白名单 ≈ {github.com, api.github.com, codeload.github.com, PyPI}，官方 UK-DALE/HF/Kaggle/Zenodo/S3 不可达；系统排查 20+ GitHub 仓库后锁定 `Ken89MathCompSci/UKDALE-NILM`（真 UK-DALE house1 6s 切片，aggregate 均值 418.9W 与论文 417.9W 交叉验证一致）
  3. 工具链：新增 `scripts/prepare_ukdale_subset.py`（CSV→6s 网格→NPZ+审计 JSON）、`configs/ukdale_fridge.yaml`、`configs/ukdale_washing_machine.yaml`；.gitignore + `__pycache__` 移出跟踪
  4. 真实 baseline 两发（CPU）：fridge test MAE 30.63W/R² 0.297/SAE 15.6%；washing_machine test MAE 14.59W/R² 0.863/F1 0.913；`evaluate.py` 复核通过
  5. 文档：REPORT_TEST.md（2 专题）、REPORT.md（路线/KPI 口径/稳定结论首建）、README.md（受限网络数据路径、Linux venv 命令、新脚本）
- 关键决策：
  - 子集无 kettle → 改用 fridge+washing_machine 并在所有报告中显式声明，不冒充全量/电器结果
  - fridge 的 P/R/F1 阈值从 500W 改 30W（500W 会把冰箱压缩机全判 OFF）
  - 超参完全沿用 baseline.yaml（仅电器+阈值两处差异），保证两电器可横向比较；epoch 触顶/早停信息如实记录
- 未决问题：
  - kettle 与全量 18 个月需官方数据（用户本机下载后走原 `inspect_h5.py`/nilmtk 路线）
  - epochs=30 是否偏短需调参验证；跨家庭（house2 数据已定位）需双 NPZ 入口
  - 点级 P/R/F1 vs 事件级口径是否切换
- 相关文件/分支：分支 `arena/01a0671f-nilm-project-model`（session 固定分支，按协议直接在其上提交）；commits：`1296c3a`（工具链）、`c250062`（fridge 报告）、收尾 commit（washing 结果+纪要+STATUS）；数据：`data/ukdale_house1_*.npz`（gitignored）、源克隆 `/home/user/work/UKDALE-NILM`（重建命令：`git clone --depth 1 https://github.com/Ken89MathCompSci/UKDALE-NILM`）
