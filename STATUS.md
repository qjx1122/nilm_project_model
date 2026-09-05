# STATUS.md

## 当前目标
- [进行中] **达标攻坚（用户指令）**：稠密 test30000 上 P>0.9 且 R>0.9（硬约束，500W 判定）；MAE 0.2W 口径经确认「物理不可达需重设」→ 目标改为尽量压低（≤6W）。泳道：F6(boost-only) → F7(stochastic boost+权重平均) → **F8(容量×4+d128/4L+stochastic+wavg，跑中，ETA≈10:3x)**。
- 攻坚已获证据（2026-09-05 本轮，全部稠密 test30000）：
  - F6 boost-only：MAE 9.31 / P .925 / R .744（ep6 早停）→ 静态事件拼接死路（过拟合，+17 新漏检）。
  - F4_ew8（λ8 无 boost）：MAE 13.75 / P .916 / R .774 → 大 λ 死路确认。
  - F7 stochastic+wavg：**F1 .892（全域最高，F3 为 .853）**/ 协议点 MAE 8.98 / P **.951** / R .733；前沿 t=150 → **P .931 / R .857**，t=80 → P .881 / R .887——离双 0.9 仅差前沿整体外扩 ~3%。缺陷：MAE 早停在 ep6 截断（激进期未展开），wavg 均值[6,9,10,11]。
  - 硬骨头分析：F3/F6 FN **98% 重合**（51 个点模型共漏）；F7 把 FN 从"无响应 10W"抬到 118–388W 临界带（可救），且 FP 从 22 压到 10（P 上限大幅抬高）。→ 攻坚方向正确：**让模型在事件上输出更满**而非改阈值（阈值扫描=单调校准，已证单点是单调族最优，无增益）。
  - 双模型 max 集成 @500W：R .812 / P .904（F3+F6+F7），增益有限；预测域乘性放大伤 MAE，死路。
- [x] 已完成（2026-09-05 调参专题）：坐标平行搜索 → `tune_final_f3_cosine`：MAE 8.58W / P 0.907 / R 0.804（**P 已达标，R 差 0.10**）。FN 解剖：88.5% 漏检点的聚合 bump≥300W（可学），瓶颈=训练事件覆盖仅~10% → 引入事件分层采样。
- [x] 已完成（2026-09-05 上一任务）：缩短版真实数据 baseline —— test MAE=13.62W / R²=0.634 / F1=0.808，结果已入库。


## 已完成
- [x] 开局仪式：git 核对（分支 `arena/01a06f16-nilm-project-model`，与 origin 同步）、按模板创建 STATUS.md
- [x] 环境恢复（Linux 沙箱）：`.venv` + PyPI 安装 torch 2.14.0/依赖，`pytest tests/` 通过
- [x] 数据核验：`data/ukdale_prepared.npz`（aggregate/target，float32，n=10,344,744 ≈ 718 天@6s；kettle ≥500W 占比 0.63%，99.5 分位 2306W，噪声底 ≈1W）与 `load_simple_npz` 格式匹配
- [x] 真实数据 2-epoch 冒烟（管道验证，产物在 /tmp，未入库）
- [x] **缩短版真实数据 baseline**：best_epoch=9，test MAE=13.62W / RMSE=137.95W / R²=0.634 / SAE=0.215 / energy_error=-0.215 / P=0.933 / R=0.712 / F1=0.808（val 最优 MAE=5.71W）；runtime 635.6s
- [x] 专题报告追加至 `REPORT_TEST.md`；会话纪要落盘 `session/NILM_AC_session_complete.md`；README 按条件触发更新；新增 `.gitignore`
- [x] **调参专题全流程**（13 次真实数据训练，累计约 55 分钟 GPU 时当）：Phase1 六泳道 → Phase2 组合 → 同尺度 A/B → F2 全量 → F3+cosine 胜出；新增 `scripts/eval_ckpt.py` 与 opt-in `lr_schedule: cosine`；`REPORT.md` 建立（算法路线/KPI 口径/稳定结论/推荐版本）；README 同步；`__pycache__` 出库

## 进行中
- **F8(容量 d128/4L+stochastic+wavg4)已证实为最强配方**：稠密 test MAE **7.37W**（新王，前最佳 F3 8.58）/ R² .725 / 协议点 P .952 R .744；前沿 t=55 **max-min(P,R)=.895**（P.895/R.895，TP238/266）——距双 0.9 仅差 ~2 TP 的样本量！被 timeout(4800s) 截死于 ep23（ep22 val MAE 3.57 仍在改善、wavg 未及执行）。快照权重存 reports/tune_f8_big/ep22_val3.57.pt。
- F9（λ2×minprf 选点）**死路确认**：MAE 12.38 / P .919 / R .767（best_ep5 早早截停）——λ 系列全灭（2/3/8 三档均劣于 λ1）。
- **F10 事故**：11:37 我并行跑的 TTA 扫描(≈1GB)+F10(d128,RSS1.48GB) 触发全局 OOM，F10 被内核击杀（只活到 ep5）。教训：**与 d128 泳道并行期间禁止任何第二进程**。F10(无jitter对照)暂不重跑——F11 结果决定是否需要。
- TTA 复核（修正归一化后，roll=0 与保存预测 corr=1.000）：F8 的 28 个 FN 中 13/28 在 ±2 roll 下被同一模型检出（-3:13 / -2:12 / -1:8 / +1:3 / +2:5 / +3:6）——对齐敏感性真实存在；但推理侧 TTA mean±1 前沿仅 .896（原始 .895）、MAE 反升至 7.69 → **post-hoc TTA 死路**，正确做法=训练时 roll-jitter 增强（已实现 `data.roll_jitter`，pytest+冒烟过；顺带加了 `training.init_ckpt` 热启动，冒烟显示 F8 权重 warm-start 首 epoch val MAE 1.65）。
- **F11 身份更正（sed 事故，因祸得福）**：config 里 roll_jitter 实际未写入（F10 模板无该行，sed 空转）→ F11=**F10 全长对照的确定性复现**（ep4/5 与被杀 F10 逐位一致，顺带验证全管道可复现）。让其跑完（ETA≈14:0x）作为「无 jitter 满额版」定稿对照，产物将回填 tune_f10_biglong。
- **F12 已排队**（轮询 PHASE11_DONE 自动开跑）：init_ckpt=F11 终权重热启动 + roll_jitter=2 续训 12ep（lr3e-4, wavg4, timeout 5400+兜底 eval），ETA≈15:0x。若 F11 已满额达标，F12 作为增益消融仍跑。
- 判定计划：F11 完赛后 threshold_scan 全网格 → 若 max-min(P,R)>0.9 则定 F11 为新稳定版（REPORT.md §4 换锚），并复验 500W 协议点+MAE≤6W 进度；若仍 .89x，则从 F8/F11 权重 warm-start（init_ckpt）+更多 epoch 做 F12；同强模型间 max 集成最后再试（勿混 d64 弱模型——已证稀释）。

## 下一步（TODO）
1. **多 seed 复验**：F3/drop0 结论差距多在 0.3–1.5W val MAE 量级、单 seed=42；对 `tune_final_f3_cosine` 用 seed 43/44 复验后再扩大结论
2. 选点准则 ablation：val MAE 最优 vs val F1 最优 vs 多目标（F2 教训的延伸；F3 中 ep10 F1=0.897 vs ep12 MAE=4.12 不同最优点）
3. 能量低估遗留：EE −19.5% → 试预测头正偏置先验 / 事件加权损失 / soft-label ON 判定
4. `window_size=256` 在大内存或 GPU 机器重测（本沙箱 torch cu130+seq256 内存线性增长至 OOM，见踩坑）
5. 评测口径升级：连续覆盖 + 按事件对齐（对齐 NILMbench 可比性）；`run_real.ps1` 可补默认路径 `data/ukdale_prepared.npz`（沙箱无法测 PowerShell）

## 决策记录 / 踩坑
- [2026-09-05 调参] **组合不必然=单因子之和**：drop0+lr1e-3 在 Phase2 小口径（8k train, 4k val）最优，但在与 anchor 同尺度 A/B（10k/6k/6k, 10ep）下 R²/EE 反而劣于 drop0-only，故最终配置回到 baseline lr=5e-4。**换更大训练预算时须重跑同尺度 A/B 复核**，小口径排序不能直接外推。
- [2026-09-05 调参] **cosine 调度是本轮最大增益来源**：F2（固定 lr）val MAE 剧烈震荡（4.9↔11.7），早停在 ep7 选点纯看运气；F3 加 cosine 后后半场稳定收敛于 4.1–5.6，稠密 test MAE 10.59→8.58（−19%）。给 trainer/experiment 加了 opt-in `lr_schedule: cosine`（默认不变，早停分支也 step）。
- [2026-09-05 调参] **模型选择本身是高影响元超参**：F2 的 best_ep7 checkpoint 在稠密 test 上 EE −28.6%，而 val 曲线 ep12 明显更优——val MAE 单调最小选点在噪声下失效。候选：val F1 / 多目标选点（TODO 2）。
- [2026-09-05 调参] **L1 loss 否决**：`tune_l1loss` val MAE 10.83、test R²=-0.009、energy_error=-96%——零膨胀稀疏目标下 L1 使模型坍缩到条件中位数(≈0)。调优保持 MSE。（history 在 `reports/tune_l1loss/`）
- [2026-09-05 调参] **window_size=256 泳道暂缓（本机内存异常）**：seq256+d64 泳道被全局 OOM 杀死；300 步内存二分显示 ~18MB/步 线性增长（seq128 锚点同环境跑 10 epochs 无恙）。torch 2.14.0+cu130 单 import 即 505MB，seq256 单 lane 峰值 2.18GB（MALLOC_ARENA_MAX=2 无效）。aliyun/pku/nju/bfsu 镜像均 TLS 拦截，无法换 CPU 轮子。结论：本环境只跑 seq128；256 窗口留到有 GPU/大内存机器。
- [2026-09-05 调参] **并行度=1**：两泳道并行时内存余量 <800MB 导致吞吐骤降（epoch 从 ~64s 恶化到 156-390s），改为顺序单泳道（OMP=2 吃满双核）。
- [2026-09-05 调参] `pkill -f <pattern>` 在 bash 工具里会匹配到自身命令行（含同样字面量）导致自杀，×2 次踩中；清进程一律先 `ps` 定位 PID 再精确 kill。
- [2026-09-05] 沙箱为 Linux + 2 核 CPU + 3GB 内存：README 的 `conda`/`.ps1` 流程不适用，改用 `python3 -m venv .venv` + `pip`（`.venv` 已加 `.gitignore`，避免 `git add -A` 误收 3.5GB 依赖）。
- [2026-09-05] `download.pytorch.org` 在本沙箱 TLS 被拦截 → 改 PyPI 默认源装 torch 2.14.0+cu130；**其 import 依赖 nvidia-* 动态库，不可卸载精简**（删了会 ImportError，需按 pin 版本逐个装回）。
- [2026-09-05] python 非 tty 时 stdout 块缓冲，`tee` 看不到实时 epoch 日志；用 `best.pt` 的 mtime 当进度心跳可观察训练推进。
- [2026-09-05] 缩短版实验口径说明：`max_samples_train=10000` 是从 70% 时间段（≈7.2M 中心点）linspace 均匀子采样，窗口间隔 ~724 点（≈72 分钟），非连续覆盖；kettle 事件短，子采样导致部分 ON 事件漏采 → energy_error=-21%、recall 0.71 与此吻合。对比 NILMbench 连续窗口口径时需注意此差异。
- [2026-09-05] 本 session 分支被 Arena 平台固定在 `arena/01a06f16-nilm-project-model`，按 BOOTSTRAP「平台固定 session 分支时可直接在当前分支进行」执行。
- [2026-09-05] 结果入库策略经用户确认：**结果目录全部提交（含 best.pt，286KB 量级）**；大文件（npz 83MB）此前已入库，不再新增大产物。

## 关键文件路径
- 数据：`data/ukdale_prepared.npz`（aggregate+target，kettle，6s）
- **推荐稳定版本**：`reports/tune_final_f3_cosine/`（config/result/history/best.pt）；结论见 `REPORT.md` §4
- 调参过程产物：`reports/tune_{l1loss,lr1e3,layers4,drop0,d128,c_d0lr1e3,c_d0lr2e3}/`、A/B：`reports/tune_final_ab10k{,_drop0only}/`、全量对照：`reports/tune_final_full/`；anchor 稠密复评：`reports/ukdale_baseline_cpu_short/dense_test_eval.json`
- 复评工具：`scripts/eval_ckpt.py`（checkpoint 换稠密口径复评，跨 run 公平对比）
- 本次实验产物：`reports/ukdale_baseline_cpu_short/`（config.yaml / train.log / history.json / result.json / best.pt，已入库）
- 全量配置（下一步用）：`configs/baseline.yaml`；调参搜索：`configs/tuning.yaml`
- 训练入口：`scripts/train.py`；查看结果：`scripts/evaluate.py --run-dir reports/ukdale_baseline_cpu_short`
- 模块：`src/experiment.py`（train_experiment）、`src/trainer.py`（fit+早停，按 val MAE 选 best）、`src/data.py`（linspace 子采样在 build_splits）、`src/metrics.py`
- 台账：`session/NILM_AC_session_complete.md`（纪要）、`REPORT_TEST.md`（专题）、`README.md`（环境与命令，本 session 已更新）
