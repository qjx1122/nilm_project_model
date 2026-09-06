# REPORT_TEST.md — 专题报告（只追加，按专题分节，不新建文件）

## [2026-09-05] 专题：真实 UK-DALE 缩短版 baseline（Kettle / Transformer Seq2Point / CPU）
- 类型：实验专题
- 目标与假设：仓库已有代码与真实数据（`data/ukdale_prepared.npz`）但从未产出真实指标（README 原声明「仅有合成 smoke test」）。目标：在 Linux CPU 沙箱上快速跑通真实数据训练，拿到第一组真实 UK-DALE 指标，验证管道与数量级合理（假设：kettle 属易学电器，MAE 应为 10¹W 量级、R² 显著高于 0）。
- 方法 / 数据 / 参数：
  - 数据：UK-DALE House1 kettle，6s 采样，n=10,344,744；按 70/15/15 时间切分。
  - 与 `configs/baseline.yaml` 唯一差异为控制 CPU 时长：`max_samples_train` 30000→10000、`max_samples_val/test` 6000 保持、`epochs` 30→10、`patience` 7→4；`window_size=128, d_model=64, nhead=4, layers=2, ff=128, dropout=0.1, lr=5e-4, wd=1e-4, batch=128, MSE, grad_clip=1.0, seed=42, on_threshold=500W`。
  - 注意：`build_splits` 对采样中心做 linspace 子采样 → 10k 窗口 ≈ 每 724 个 6s 点取 1 个（间隔 ~72 分钟），**非连续覆盖**。
  - 环境：Linux，2 核 CPU，torch 2.14.0+cpu 用法（`.venv`）；产物目录 `reports/ukdale_baseline_cpu_short/`。
- 结果 / 结论：
  - 收敛正常：best_epoch=9/10（早停未触发），runtime 635.6s（≈64s/epoch）。
  - Val 最优：MAE=5.71W，R²=0.861。
  - **Test：MAE=13.62W，RMSE=137.95W，R²=0.634，SAE=0.2145，energy_error=-0.2145，Precision=0.933，Recall=0.712，F1=0.808。**
  - 解读：MAE≈13.6W（信号 std=183W、99.5 分位 2306W）数量级合理，Seq2Point 管道与指标实现自检通过；recall 0.71 / 能量低估 21% 与「子采样漏采短事件」一致，属口径效应而非实现缺陷；precision 高说明模型极少误触发。
  - 结论：代码-数据-指标全链路在真实数据上成立，可放心扩展到全量 baseline 与调参；但**本组数字不可直接对标 NILMbench 发表值**（采样密度不同）。
- 是否进入 REPORT.md（稳定结论）：否（单次缩短跑，等全量 baseline 确认后进入）。
- 遗留问题：
  - 全量 baseline（30k × 30 epochs，CPU 预计 ~85min）作为对照。
  - 评测口径升级建议：连续窗口 + 事件级对齐，或与 NILMbench 完全一致的 split 与聚合方式，以便外部可比。
  - 调参（`configs/tuning.yaml` 12 trials）在 CPU 上的预算评估与抽样策略。

## [2026-09-05] 专题：真实 UK-DALE 超参调优（Kettle / Seq2Point / CPU，Phase1-3 完整链路）
- 类型：实验专题
- 目标与假设：在缩短版 baseline（test MAE 13.62W、EE −21.5%、Recall 0.712）基础上，通过超参与训练策略调优改善 MAE/R²/F1/能量偏差。假设：欠拟合迹象（train≈val）→ dropout/lr/容量有空间；MSE 对尖峰的惩罚与能量偏差相关 → loss 与调度值得验证。
- 方法 / 数据 / 参数：
  - 设计：坐标平行搜索（替代 tune.py 顺序随机搜索，省墙钟且 val 口径统一）。Phase1 六泳道单因子（8k/4k/4k、12ep、patience4）：loss=l1 / window=256 / d128h8ff256 / lr=1e-3 / layers=4 / dropout=0。Phase2 组合胜出者（d0×lr1e-3、d0×lr2e-3）。Phase3：同尺度(10k/6k/6k×10ep) A/B 复核 → 全量 F2（30k/8k，test 30000 稠密，20ep/patience5）→ F3=F2+cosine。
  - 新增工具/代码：`scripts/eval_ckpt.py`（checkpoint 换口径复评，切分/归一化可精确复现，已验证与 result.json 逐位一致）；`src/trainer.py`+`src/experiment.py` 增加 opt-in `training.lr_schedule: cosine`（默认行为不变，早停分支也正确 step）。
  - 环境约束：2 核 CPU 3.9GB；泳道并行 2 条时 torch+cu130 单进程 RSS≈2GB 触发全局 OOM（seq256 泳道被杀；300 步二分显示 seq256 下 ~18MB/步 线性增长，seq128 正常）→ 全程降为顺序单泳道；`window_size=256` 在本机暂缓。
- 结果 / 结论：
  - 否决项：L1 loss 坍缩（零膨胀目标下条件中位数≈0：val MAE 10.83，test R²=−0.009，EE −96%）；lr=2e-3（d0 下发散，val 10.44）；d128/ff256（val 5.39 劣于 d64，过拟合）；layers=4（val 5.03，无收益）；dropout0+lr1e-3 组合在 10ep 预算下 MAE −8.6% 但 R²/EE 变差 → 被同尺度 A/B 否决。
  - 采纳项：**dropout 0.1→0**（同尺度 A/B：MAE −7.7%、val R²↑、Recall +5.1pp、EE 略改善）；**cosine lr 衰减**（5e-4 起点）——F2 暴露固定 lr 的 val 剧烈震荡（11.67↔4.93）导致早停选点踩雷（best_ep7 恰是次优点，EE −28.6%）；F3 曲线平滑后 best_ep12 泛化良好。
  - **同口径公平对比（稠密 test=30000）**：anchor 11.67W / R²0.671 / F1 0.832 / Rec 0.752 / EE −14.8% → **F3 8.58W（−26.5%）/ R² 0.723 / F1 0.853 / Rec 0.805 / EE −19.5%**（Precision 0.930→0.907 为 recall 换入的代价，F1 净升）。对原 6k 口径 anchor 13.62W 为 −37%。
  - 方法论教训（可复用）：**模型选择本身是超参数**——稀疏事件+小 val 集下「val MAE 最低 epoch」噪声极大，调度平滑化（cosine/EMA/SWA）或按 val F1/多指标选择是更稳的选点策略；跨协议对比（不同 max_samples_* 即不同 test 子集）不可直接比数值，必须用 eval_ckpt 统一到稠密口径。
- 是否进入 REPORT.md（稳定结论）：是（F3 配置+指标成为首个推荐稳定版本；标注单 seed 未复验）。
- 遗留问题：
  - 全部结论来自 seed=42 单跑，采纳/否决的差距多在 0.3–1.5W val MAE 量级 → 建议关键配置 2-3 seeds 复验后再扩大。
  - EE −19.5% 仍有能量低估 → 候选：预测头加正偏置先验、损失加事件加权（需改 trainer）、或 ON 阈值下 soft-label；稠密 test 仍是 241 步子采样，正式版建议连续覆盖评测对齐 NILMbench。
  - window=256/更大容量在 torch+cu130 CPU 轮子的内存异常未解（换官方 CPU wheel 或 GPU 机器重测）。
  - F3 的 val_F1 最优 epoch(10, 0.897) 与 val_MAE 最优(12) 不同 → 选点准则 ablation 值得单独跑。

## [2026-09-05] 专题：P/R 双 0.9 攻坚（事件覆盖·损失加权·平均化·容量·集成·噪声地板）
- 类型：实验专题（目标导向攻坚，含 3 项新代码能力）
- 目标与假设：用户指令「Recall 与 Precision 均 >0.9，MAE 尽量压低（0.2W 经确认物理不可达→重设 ≤6W 承诺）」。起点=F3（MAE 8.58 / P .907 / R .805）。假设链：H1 阈值校准可达标 → 死（单调族中单阈值即最优，F3 无解）；H2 事件覆盖密度不足 → F6 静态拼接死路（FN 跨模型 98% 重合证伪）；H3 损失λ加权 → λ∈{2,3,8}×选点{mae,f1,minprf} 五连全灭（保守化：FP↓FN↑）；H4 训练事件池每 epoch 重采样+checkpoint 平均 → F7 前沿 F1 .892 成立；H5 容量×预算匹配 → F10 命中（MAE 6.68）；H6 对齐敏感性（13/28 FN 在 ±2 点滚动下被同模型检出）→ F12 jitter 增强=召回暴涨但伤 MAE/EE，双刃。
- 方法 / 口径：全部结论在「val 8k 选点 + 稠密 test n=30000 @500W 复评」协议下产生；threshold_scan/`--save-preds` 仅诊断。新代码（opt-in，默认关闭不影响既有口径）：`data.event_boost.stochastic_epochs/event_frac`（每 epoch 重采样，n 不膨胀）、`training.weight_avg:k`（top-k 权重平均）、`training.select_on:f1|minprf`、`data.roll_jitter`（输入窗 ±r 移位增强）、`training.init_ckpt`（热启动）。
- 结果总表（稠密 test30000）：
  | 泳道 | 配方 | MAE | P@500 | R@500 | F1 | 判定 |
  |---|---|---|---|---|---|---|
  | F4_ew3/ew8 | λ3/λ8 | 9.6x/13.75 | .91-.92 | .71-.77 | ≤.84 | ❌ |
  | F5 | λ3×boost×f1 | 9.61 | .913 | .711 | .799 | ❌ 三因混杂 |
  | F6 | boost 静态 | 9.31 | .925 | .744 | .825 | ❌ |
  | F7 | stochastic+wavg4 | 8.98 | .951 | .733 | .828（前沿 .892@150） | ✅ 配方部分 |
  | F9 | F7+λ2×minprf | 12.38 | .919 | .767 | .836 | ❌ |
  | F8 | +容量 d128/4L(23ep截停) | 7.37 | .952 | .744 | .835 | ✅ 容量 |
  | **F10(=F11 复现)** | **F8 配方全长 30ep+wavg** | **6.68** | **.938** | **.797** | **.862** | 🏆 新稳定版 |
  | F12 | F10 热启动+jitter | 7.63 | .954 | .782 | .860 | ⚠️ 双刃 |
  | F13 | F10 配方 seed43 | 7.24 | .948 | .748 | .836 | seed 敏感 ±0.5W |
- 前沿与集成：F10 max-min(P,R)=.895@t=100（P.901/R.895）；对 7 ckpt 穷举 avg/max/min/AND/median3/median4（60+ 组合）全部饱和 .89-.90——**median4(f8,f10,f12,f13)@t=95 = .906/.906 越过双线，但 val 选点协议给出 t*=115 → .907/.880**（选点差 0.026 即全部缺口，val 仅 ~80 个 ON 点的分辨率问题）。
- 噪声地板（为什么差的就是物理极限）：FP 侧 25 个中 **19 个（76%）为子电表丢数**（聚合 bump≥1500W、子电表连续 OFF 中位 2381 点≈4h）——剔噪 P .906→**.976**；FN 侧 25 个中 ~29% 聚合无痕迹（bump<300W，不可学）；跨模型/跨配方 FN 重合 98%。结论：**500W 判定点 P/R 双 0.9 在本数据集点级口径下被标签噪声锁定，非调参可破**；可达路径为（i）决策阈值运营点（95-115W，代价=违反双端同阈值协议），（ii）事件级口径（NILMbench 风格，F10 事件级 P.913/R.821），（iii）数据修复（对齐/补子电表，工程侧）。
- 方法论教训（可复用）：①**组合收益必须同口径全长复验**（F8 截停 23ep 的 ckpt 与全长 F10 差 0.7W MAE/2pp R）；②**timeout 预算按 pace 上限×1.5 设**（F6/F8/F10 三次被截/杀，其中 F10 死于并行扫描触发的全局 OOM——d128 泳道期间禁止第二进程）；③ val 极稀疏时按 minprf/F1 选点方差大于收益，MAE+wavg 仍是稳健解；④ sed 改 yaml 会静默失配——配置生成必须 python+yaml.safe_dump 并 diff 校验。
- 是否进入 REPORT.md：是。F10 成为新稳定版本（§4），稳定结论 §3.5-3.8 追加；median4 运营点作为可选检测配置记录。
- 遗留问题：① 多 seed 复验仅部分完成（F13=seed43：回归结论稳，精确运营点不稳），seed44 未跑；② val 集太小（8k 仅 ~80 ON）→ 决策层选点建议 val 加密到 30k 同稠密口径；③ jitter 增强的「干净单变量」全长对照未跑（F12 是热启动 12ep 版）；④ 事件级/数据修复路径未展开；⑤ 500W 双 0.9 若为硬性验收，需与用户重议口径（决策阈值或事件级），已备好全部定量依据。
