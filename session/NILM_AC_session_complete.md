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

## [2026-09-05] 会话纪要（续：调参专题）
- 目标：基于第一组真实指标（MAE 13.62W / EE −21.5% / Rec 0.712）继续调优。
- 完成项：13 次真实数据训练（总约 55 分钟 CPU）：Phase1 六泳道（l1 / win256 / d128 / lr1e-3 / layers4 / drop0）→ Phase2 组合（d0×1e-3 / d0×2e-3）→ 同尺度 A/B（10k/6k/6k）→ F2 全量（30k，稠密 test30000）→ F3=F2+cosine（胜出）。新增 `scripts/eval_ckpt.py`、opt-in `training.lr_schedule: cosine`（trainer 早停分支亦正确 step）。建立 `REPORT.md`：推荐稳定版本 `tune_final_f3_cosine`，稠密 test MAE 8.58W / R² 0.723 / F1 0.853 / EE −19.5%（对 anchor 同口径 −26.5%）。REPORT_TEST.md 追加调参专题全文。
- 关键决策：组合结论须回锚点口径复核（Phase2 冠亚军在 10k A/B 下被否）；cosine 是本轮最大增益（平滑 val 震荡）；并行泳道降为顺序（torch cu130+seq256 OOM）；L1 loss 禁用（零膨胀坍缩）。
- 未决问题：单 seed（建议 43/44 复验）；模型选择准则（val MAE vs F1）ablation；EE −19.5% 的能量损失补偿；seq256/更大容量在 GPU 机器复测；稠密 test 仍是 241 步子采样，未对齐 NILMbench 连续口径。
- 相关文件/分支：`arena/01a06f16-nilm-project-model`；产物 `reports/tune_*`、`reports/ukdale_baseline_cpu_short/dense_test_eval.json`；文档 `REPORT.md`、`REPORT_TEST.md`、`README.md`、`STATUS.md`。
- ⚠️ 推送阻塞：session 后半程 GH_TOKEN 失效，最后 6 个 commit 仅存本地（HEAD=727ffb4，远端=3fa28c7）；重连 GitHub 后 push 即可，无数据丢失风险。

## [2026-09-05] 会话纪要（续：补 push 与事故恢复）
- 目标：把上轮 token 过期未推送的调参 commit 补推到远端。
- 完成项：push 时发现沙箱已重建——.git 为全新 clone（HEAD=d0529cf），本 session 全部 commit 对象丢失（含已推过的 3fa28c7 引用），仅剩工作区文件快照；远端 arena 分支停在 3fa28c7。恢复路径：`add -A`+tmp commit → fetch → `reset --soft` 远端 tip → 重放为单 commit `1faa468` → push 成功（`3fa28c7..1faa468`，远端=本地已核验）。
- 关键决策：6 个语义化 commit 合并为 1 个重放 commit（细粒度历史不可恢复，快照树=终态，信息无损）；重放前用 ast 校验 trainer/experiment 代码一致性通过。
- 未决问题：**`.venv` 丢失**（快照排除目录），下次实验 session 需按 README §3.1 重装依赖（注意：PyPI 默认源装 torch+cu130 后不可删 nvidia 依赖）；原调参中间 commit 信息以本纪要/REPORT_TEST 为准。
- 相关文件/分支：`arena/01a06f16-nilm-project-model` 远端 tip=`1faa468`。

## [2026-09-05] 会话纪要（P/R 双 0.9 攻坚 · F4–F13 全泳道）
- 目标：稠密 test30000 上 P>0.9 且 R>0.9（500W 判定）；MAE 0.2W 经用户确认不可达→重设为尽量压低（承诺 ≤6W，P/R 优化不得劣化 MAE）。
- 完成项：F4(ew3/ew8)/F5/F6/F9 四路证伪（λ 加权与静态 boost 全死路）；F7 确立 stochastic+wavg 配方；F8 证明容量收益（被 timeout 截停，checkpoint 快照抢救）；F10(=F11 确定性复现) 全长完赛成**新稳定版本**：MAE 6.68W（−22% vs F3）/R² .770/EE −10.0%/P .938/R .797/F1 .862；median4 集成（f8,f10,f12,f13）val 选点 t*=115 → test .907/.880、test 侧 t=95 处 .906/.906（诊断）。新能力入库：`stochastic_epochs/event_frac`、`weight_avg`、`select_on:f1|minprf`、`roll_jitter`、`init_ckpt`（全 opt-in）。噪声地板定量：FP 76% 为子电表丢数（剔噪 P→.976）、FN ~29% 聚合无痕迹、FN 跨模型重合 98%——**双 0.9 在 500W 点级协议被标签噪声锁定**。REPORT.md §3/§4 定稿；REPORT_TEST.md 追加攻坚专题全表。
- 关键决策：F11 身份更正为「F10 全长对照」（sed 失配事故因祸得福，验证管道确定性）；F12 jitter 定性双刃、不进稳定版；post-hoc TTA/跨族集成/λ 系列判死勿再试；产物 canonical 收敛到 reports/tune_f10_biglong/。
- 未决问题：seed44 复验；val 加密（30k）用于决策层选点；jitter 全长单变量对照；500W 双 0.9 若为硬验收需与用户重议口径（决策阈值 95-115W 或事件级 P/R），依据已备好。
- 相关文件/分支：`arena/01a06f16-nilm-project-model`；产物 `reports/tune_f4_* … tune_f13_seed43`；诊断件 `/tmp/*_preds.npz`、`/tmp/val_gate.py`；文档 `REPORT.md` §3.5-3.8/§4、`REPORT_TEST.md` 末节、`STATUS.md`。
- 事故记录：沙箱全程未重建但 turn 间长 sleep 轮询易被杀（以后台进程+result.json 为准）；d128 泳道期间并行第二进程触发全局 OOM 误杀 F10（教训入 REPORT.md §5）；GitHub 令牌短时效，push 需择机重试。
