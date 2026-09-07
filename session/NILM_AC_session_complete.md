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

## [2026-09-05] 会话纪要（GPU 自动优先小改造）
- 目标：代码自动检查 GPU——有则优先使用，无则按原有逻辑。
- 完成项：新增 `src/device.py`（`cuda_ready/resolve_device`：有卡→cuda，config 的 cpu/auto 被覆盖；`cuda:i` 合法索引保留、越界回落首卡；无卡→auto=cpu、显式名照旧返回）。接线 `src/experiment.py`（select_device 委托）、`scripts/eval_ckpt.py`、`scripts/threshold_scan.py`（脱离硬编码 cpu，map_location 同步走卡）；决策打印 `[device]` 入 train.log。新增 tests/test_device.py（4 例，monkeypatch 双分支）全绿（合计 5 passed）；合成端到端冒烟确认无 GPU 机器行为与数值路径不变。README §3 补一句设备选择说明。
- 关键决策：GPU 存在时**覆盖** config 的 `device: cpu`（用户指令「优先使用 GPU」优先于配置）；无 GPU 时不做任何静默降级——显式 cuda 照旧交给下游报错，保持原语义。
- 未决问题：真实 GPU 机器上的显存适配（d128/bs64 seq128 显存需求小，预计 <1GB；未见真实 CUDA 环境实测，本沙箱无卡）。
- 相关文件：`src/device.py`、`tests/test_device.py`、`README.md`、`STATUS.md` 决策记录。

## 2026-09-07 续7（沙箱重建第7次·恢复+CSV 依据列）
- 症状同前六次：`.venv`/`.git` 重置、工作区文件幸存、远端 tip `059a008` 完好。恢复=`git fetch origin <branch>`→`git reset FETCH_HEAD`→重建 venv。注：pytorch 官方 whl/cpu 源 TLS 被断开，改走 pypi 默认源装 torch 2.14.0+cu130 成功（本沙箱无 GPU，resolve_device 自动回落 CPU，行为符合设计）。
- 需求追加：CSV 增加 `rationale` 列——每轮调参参数选择的依据（针对父轮指标的诊断与可证伪假设），26 行全覆盖无缺失；生成器 LANES 注册表升级为 6 元组，依据与台账同源，后续重跑不丢。
- 交付：`reports/tuning_rounds.csv`（26×56，指标数值与 55 列版逐项一致），commit `见下`，已推送。

## 2026-09-07 续8（A1 实验执行：四元约束点估计达标）
- 沙箱第 8 次重建（.venv+.git 灭失）→ 标准恢复（fetch→reset FETCH_HEAD→venv 重建；新坑：新 venv 依赖装齐要含 sklearn/tqdm，requirements.txt 有全表，照单安装）。
- 跑通 `scripts/a1_val30k.py`（4 成员×val/test 30k 稠密推理 ≈25min，缓存 `.a1_preds.npz` 幂等）：**median4 raw 在 val30k max-min 选点 t*=95 → test P=R=F1=.9060、SAE=.189**——四元约束运营点口径点估计全过；val8k 时代 0.026 选点缺口被精确收复，"val 分辨率=瓶颈"诊断闭环。
- B1（sup60/m2of3 时序滤波）**判死**：seq2point 点标签下真事件即单点尖峰，滤波同杀 TP；后处理须配事件级聚合口径（教训入册）。
- 统计诚实性：N_ON=266（修正前案 665 的反推错误），1σ=1.84pp，bootstrap min(P,R) CI=[.868,.940] → 按预注册记"达标待复验"，未动 REPORT §4。转正=B3（seed44/45 补员 median6，≈4h CPU）或 C1（剔噪口径）。
- 台账：REPORT_TEST 新增 A1 专题、STATUS 决策行、commit ff10209 已推。阻塞：无；待用户裁决 B3 是否开跑。

## 2026-09-07 续9（教学文档重构：TUNING_GUIDE.md）
- 需求：以"无算法理论基础的软件工程师"视角重梳调参过程并整理落盘文档。产出 `TUNING_GUIDE.md`（13 节：任务/指标/配置字典/方法论五铁律/四阶段战史/SOP/命令/坑单/术语表/文档地图）；BOOTSTRAP 文档表登记（整篇重写式维护，不追加流水，守住防文件爆炸纪律）；README 加阅读顺序入口+文件树补登（threshold_scan/build_tuning_csv/a1_val30k/tuning_rounds.csv）。
- 数字对账：全部对照 tuning_rounds.csv 与 a1_val30k.json 复核；修正旧稿三处不精确（"99.37% 为 0W"→"<500W 占比"；drop0 收益口径按 8k 泳道内最佳+10k A/B −7.7% 重述；攻坚尸检 FN/FP 用实测 54/14，弃 665 反推旧数）。
- 期间沙箱第 9 次重建（venv+.git 灭失）→ 标准恢复；`.a1_preds.npz`(764K) 入 .gitignore 排除（缓存可再生）。
- 状态：文档任务闭环；B3（seed44/45 补员转正）仍待用户拍板。

## 2026-09-07 续10（CSV 版式：每轮前置对照轮+空行分块）
- 需求：tuning_rounds.csv 每轮输出前先行输出所对比轮（父轮）整行，再当前轮，块尾空行。实现于 `scripts/build_tuning_csv.py` 写出段（数据不变、纯排版）；26 块校验（1 个 baseline 单行块）+ 数字与 result.json 回归 OK；父行在多个子块中重复为预期设计。
- 期间沙箱第 10 次重建（.venv+git 再灭失，本轮编辑落在工作区幸存）→ 标准恢复（fetch/reset + numpy/pyyaml 重装即可重跑生成器）。
- commit+push 见下；无遗留。

## 2026-09-07 续11（三口径公平性代码审计）
- 用户问题：对比调优时是否必须固定 max_samples_train/val/test。代码结论：test=考卷（linspace(N) 换 N 即换题，必须同值或 eval_ckpt 复评）；val=评审答卷（影响 best-epoch/top-k 留权重，单变量对比必须冻结）；train=学习材料（合法独立因子，但改它=数据规模轴，且隐性联动 event_boost 替换量 n×frac 与每 epoch 步数 n/batch；归一化统计与抽样无关）。
- 落地：`build_tuning_csv.py` Δ 守卫由"仅 n_test 相等"升级为三口径（未记录侧记 ? 不阻塞）；CSV 新增 `protocol_diff` 列（57 列），F2 行 Δ 转标注；规则全文写入 TUNING_GUIDE §4 铁律 1。历史 26 轮复审：单因子轮全部合规（三值同），Phase 间对比由 eval_ckpt 统一，无被掩盖的结论反转（F2 属声明式规模轴，数字仍在 note/rationale）。

## 2026-09-07 续12（表格体检与两处修复）
- 按用户复核请求对 CSV 做全面体检（结构/8 指标×26 轮对账 result.json/Δ 数学/父行重复一致/守卫行为），发现两处真问题并修：①昨日升级的 Δ 守卫误以"实际 n_*"判公平→F5/F6 静态 boost 行（44935 实际长度）被误伤，改回 config 名义值判定后 Δ 恢复（+11.9%/+8.5%）；②anchor 行 p_max_samples_test 显示 6000（baseline 名义）与复评实际 30000 不符，已修。
- 规则最终版：test/val 名义采样参数必须同值（跨口径走 eval_ckpt），train 可动但即数据规模轴；实际集长可被处置（boost）合法改变、不参与公平性判定。
- 指南 §4 铁律 1、REPORT_TEST 验证专题、STATUS 各落笔；commit 见下。

## 2026-09-07 续13（CSV 版式回退单行制 + report_dir 列）
- 需求：去掉父轮对照行/空行（对比块作废，恢复一行一轮），并加 report_dir 列指向 reports/ 下产物文件夹。实现于 build_tuning_csv 写出段+cols+load 循环；anchor 行目录=ukdale_baseline_cpu_short（dense_test_eval.json 所在），win256/集成行按实际留空。
- 校验：文件 27 行（表头+26）、58 列、report_dir 全部真实存在（除两处设计性留空）、Δ 与指标数字零回归。commit 见下。
