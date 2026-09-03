# STATUS.md

## 当前目标
- 任务「真实 UK-DALE baseline」：在可达数据源上跑通真实数据的训练+评估并沉淀报告（用户确认：下载 UK-DALE 并跑真实 baseline）

## 已完成
- [x] **真实数据 baseline 全链路**：UK-DALE house1 子集（fridge + washing_machine）训练→验证→测试→评估，结果已沉淀 REPORT_TEST.md（2026-09-03 两节）
- [x] 代码框架：Transformer Encoder Seq2Point（src/：data / model / metrics / trainer / experiment）
- [x] 脚本链：inspect_h5 / train / evaluate / tune / run_smoke
- [x] 配置：configs/baseline.yaml、configs/tuning.yaml
- [x] CPU smoke test 通过（seed=42，链路：数据→Dataset→模型→训练→验证→测试→指标→checkpoint）
- [x] 本 Linux 沙箱环境恢复：venv 于 `/home/user/venv`，依赖全装（torch 2.14.0，无 CUDA）
- [x] 数据源侦察：沙箱出口白名单≈{github.com, api.github.com, codeload.github.com, PyPI}；jack-kelly/UKERC/HF/Kaggle/S3 全部不可达 → 全官方 UK-DALE 下载在本环境不可行
- [x] 选定替代数据：GitHub 仓库 Ken89MathCompSci/UKDALE-NILM `APR-new-House1-dataset/`（真 UK-DALE house1 6s，2014-09-08→10-21 共 633,600 行零缺口；aggregate+fridge/washing_machine/dishwasher/microwave，**无 kettle**；house1 aggregate 均值 418.9W 与论文 417.9W 交叉验证一致）
- [x] `scripts/prepare_ukdale_subset.py`：CSV→严格 6s 网格→最长连续段→NPZ+报告（commit 1296c3a）
- [x] 专项配置 `configs/ukdale_fridge.yaml`、`configs/ukdale_washing_machine.yaml`（fridge ON 阈值 30W，洗衣机 500W）

## 进行中
- （任务「真实 UK-DALE baseline」已收尾；下一个可立项任务见 TODO）

## 本任务结果快照（2026-09-03，UK-DALE house1 子集 44 天 @6s）
- fridge：test MAE 30.63W / RMSE 50.39W / R² 0.297 / SAE 15.6%（+15.6% 低估）/ P 0.634 R 0.960 F1 0.764 @30W；30 epoch 触顶
- washing_machine：test MAE 14.59W / RMSE 92.51W / R² 0.863 / SAE 2.29% / P 0.909 R 0.917 F1 0.913 @500W；epoch 16 早停（best 9）
- 产物：`reports/ukdale_fridge/`、`reports/ukdale_washing_machine/`（result.json/history.json/best.pt），训练日志 `reports/ukdale_baseline_train.log`

## 下一步（TODO）
1. 调参（fridge 优先，30 epoch 触顶 + 15.6% 低估）：`scripts/tune.py --config configs/tuning.yaml --data-path data/ukdale_house1_fridge.npz`（只看 Validation；建议先动 epochs/lr/max_samples，再动容量）
2. kettle / 全量 18 个月 / 官方 `ukdale.h5`：需用户本机下载（本沙箱网络不可达）；拿到后 `scripts/inspect_h5.py` → nilmtk 转换器或自写抽取 → `data/` 下 NPZ，即可复跑同一配置对比
3. 跨家庭第二阶段：`UKDALE-NILM/APR-new-House2-dataset/`（house2 2013-06-15→07-28 同为 6s 网格）可做 train house1 / test house2 —— 需给 `train.py` 加双 NPZ 入口（当前只有单序列 70/15/15 切分）
4. 启动/关闭沿的大误差处理：事件沿加权损失或分段建模（RMSE/MAE=6.3:1，见 REPORT_TEST washing 专题）
5. 沙箱数据文件位置：`/home/user/work/UKDALE-NILM`（浅克隆，未入 git；沙箱重建后需重新 clone，命令见 session 纪要）

## 决策记录 / 踩坑
- [2026-09-03] 本环境无 conda，且 `download.pytorch.org` 不通 → 改用 `python3 -m venv /home/user/venv` + PyPI 默认源装 torch（CPU-only 也可用；venv 放仓库外避免污染 git / 不受快照排除目录影响）
- [2026-09-03] smoke 重跑会改动已跟踪的 `reports/smoke/*` 与 `src/__pycache__/*`；均为确定性产物，验证后 `git checkout --` 还原，保持树干净；`__pycache__` 已移出跟踪并加 .gitignore
- [2026-09-03] 已知隐患：run_*.ps1 为 Windows 专用，Linux 下直接调 `python scripts/xxx.py`
- [2026-09-03] **沙箱网络**：egress 白名单近似 {github.com/api/codeload, PyPI}；`raw.githubusercontent.com`、`objects.githubusercontent.com`（release 资产/LFS）、HF、Kaggle、Zenodo、S3/GCS、jack-kelly.com 全部 TLS 阻断。**结论：官方全量 UK-DALE 在本沙箱不可达**；可用通道=浅克隆 ≤100MB/文件的 GitHub 仓库，或 `api.github.com` contents(≤1MB)/blobs(≤100MB) 直接取文件
- [2026-09-03] **数据源筛选过程**（均验证后排除）：AAJGithub/Real-Time-Recs-UKDale（npy 是 `<U19` 字符串状态标签非功率）、hrts51/AI-Smart-Energy-Monitoring（合成数据，正态分布温度）、NunoAlberto/seq2point（是 REDD）、Ken89MathCompSci/TRFNILM（extraction_report 自述 redd.h5）、nilmtk 仓库自带 h5（合成）、PyPI nilm 系包（无数据）→ 唯真数据可得 = Ken89MathCompSci/UKDALE-NILM 的 UK-DALE 切片
- [2026-09-03] **kettle 不可得 → 本 baseline 换用 fridge（压缩机循环、on_frac 48.8%、1148 事件）+ washing_machine（事件型、354 事件）两电器**；fridge 的 P/R/F1 阈值改 30W（500W 会全判 OFF）；报告如实标注"子集+换电器"，不冒充全量 kettle 结果
- [2026-09-03] 数据完整性交叉验证：house1 aggregate 均值 418.92W vs UK-DALE 论文 417.9W ✓；6s 网格零缺口/无重复/单调 ✓

## 关键文件路径
- 协议：`BOOTSTRAP.md`
- 训练入口：`scripts/train.py`；调参：`scripts/tune.py`；冒烟：`scripts/run_smoke.py`；数据准备：`scripts/prepare_ukdale_subset.py`
- 核心代码：`src/`；配置：`configs/baseline.yaml`、`configs/ukdale_fridge.yaml`、`configs/ukdale_washing_machine.yaml`
- 结果：`reports/ukdale_fridge/result.json`、`reports/ukdale_washing_machine/result.json`、`reports/ukdale_baseline_train.log`
- 本地数据（不入 git）：`data/ukdale_house1_*.npz`（已 gitignore）；源 CSV 克隆：`/home/user/work/UKDALE-NILM`
- Python 环境（沙箱内、仓库外）：`/home/user/venv`（`/home/user/venv/bin/python`）

## 会话收尾状态（2026-09-03）
- [x] STATUS.md / session 纪要 / REPORT_TEST.md / REPORT.md / README.md 全部落盘
- [x] 本地 commit ×3（1296c3a 工具链 → c250062 报告 → 收尾 commit）
- [ ] **push 未完成**：会话后期 GH_TOKEN 过期（gh api 401 Bad credentials，git push 同拒）。下个 session 第一件事：`gh auth status` 验证 → 若仍失效请用户在 Arena 重连 GitHub → `git push origin arena/01a0671f-nilm-project-model`
