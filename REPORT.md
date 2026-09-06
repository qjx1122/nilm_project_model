# REPORT.md — 稳定结论（算法路线 / 稳定实验结论 / KPI 口径 / 推荐稳定版本）

> 仅收录已验证的稳定结论；专题过程与未定论数据见 `REPORT_TEST.md`。最近更新：2026-09-05。

## 1. 算法路线

- 任务：UK-DALE 单电器（kettle）非侵入式负荷分解，Aggregate → 中心点功率回归（Seq2Point）。
- 模型：Transformer Encoder（Linear 投影 + 正弦位置编码 + norm-first encoder + center-point 回归头）。
- 数据：仓库内置 `data/ukdale_prepared.npz`（6s 采样，n=10,344,744 ≈ 718 天，House1 aggregate+kettle；float32 无缺失；kettle ≥500W 占比 0.63%，符合 UK-DALE 特性）。
- 切分：按时间顺序 70/15/15，窗口不跨切分边界；归一化统计仅取 train 段。

## 2. KPI / 验收口径

- 指标：MAE、RMSE、R²、SAE、energy_error（单位 W，反归一化后计算）；开关态 P/R/F1 以 `on_threshold_watts=500` 判定。
- 模型选择：**只允许用 Validation 指标**（当前=val MAE）；Test 仅在锁定配置后跑一次。
- 跨 run 公平对比必须同 test 口径；不同 `max_samples_test` 即不同子集，数值不可直接互比，可用 `scripts/eval_ckpt.py --max-samples-test` 统一加密到稠密口径（n=30000）复评。
- 合成数据（`run_smoke.py`/`--synthetic`）指标一律不得作为科学结果引用。

## 3. 稳定实验结论

1. **真实数据管道成立**（原缩短版 baseline，10k×10ep）：test MAE 13.62W，代码-数据-指标全链路验证通过（`reports/ukdale_baseline_cpu_short/`）。
2. **dropout 0.1→0 是真实收益**（同尺度 10k/6k/6k×10ep 公平 A/B）：MAE 13.62→12.57（−7.7%），val R² 与 Recall 同向改善；小数据欠拟合区间内正则无益。
3. **L1 loss 在零膨胀稀疏目标上坍缩，禁用**：模型收敛到条件中位数（≈0），test R²=−0.009、能量偏差 −96%。
4. **lr 与预算需匹配**：lr=1e-3 短预算（10ep）下 MAE 略降但 R²/能量偏差恶化；lr=2e-3+dropout0 不稳定（val MAE 10.44）。固定 lr 收尾会造成 val 剧烈震荡、早停点踩雷；**cosine 衰减可显著平滑**（val MAE 震荡区间从 ±140% 收窄）。
5. **模型选择本身是高影响「元超参」**：val 噪声大时按 val MAE 单调最小选点会放大运气成分（F2 best_ep7 次优 checkpoint → 稠密 test EE −28.6%；同曲线 ep12 实际更优）。改进已落地：opt-in `training.select_on: f1|minprf` 与 `training.weight_avg: k`（top-k 权重平均覆盖 best.pt）——F10 以「val MAE + wavg4」吃到平滑红利。
6. **容量必须配合覆盖与预算**：d128/h8/4L/ff256 在 8-20ep 小预算下无益（Phase1 d128 val 5.39 劣于 d64），但在 30ep+stochastic 事件重采样+cosine+wavg 下成为最大增益源：MAE 8.58→6.68（−22%）、R² .723→.770、EE −19.5%→−10.0%。
7. **事件增广家族判定**：静态事件拼接（F6，MAE 9.31/R .744）、λ 事件加权损失（F1b/F4_ew3/ew8/F5/F9，全数劣于 λ1）、预测侧阈值校准（单调族中单阈值已是最优）均为死路；**每 epoch 重采样事件池（45%）+ 权重平均**是该维度唯一有效配方。
8. **500W 判定点的 P/R 上限由标签噪声锁定**：稠密 test 硬骨头 FN 跨模型 98% 重合（含 ~25% 总表无痕迹的不可学点）；FP 中 68% 为子电表丢数噪声（聚合 bump≥1500W、子电表 OFF 中位 4h）——剔噪后 t=95 工作点 P .906→.976。任何单模型/组合在协议点只能达 P .93-.95 / R .78-.81。

## 4. 推荐稳定版本

- **`tune_f10_biglong`（2026-09-05，取代 tune_final_f3_cosine）**：d128/h8/4L/ff256 + dropout 0 + cosine lr(4e-4, T_max=30) + weight_avg 4 + `event_boost{stochastic_epochs, event_frac .45, max_extra 15k}`，bs64，30k train / 8k val / 30k 稠密 test，CPU（~2h）。
  - 稠密 test(n=30000)：**MAE 6.68W，RMSE 104.4W，R² 0.770，SAE 0.100，P 0.938 / R 0.797 / F1 0.862，energy_error −10.0%**；较前稳定版 MAE −22%、R² +.047、EE 减半。
  - 复现：`OMP_NUM_THREADS=3 ./.venv/bin/python scripts/train.py --config reports/tune_f10_biglong/config.yaml --data-path data/ukdale_prepared.npz --out <目录>`（产物 canonical 已在 reports/tune_f10_biglong/，历史目录 tune_f11_big_jit/ 为同配置的确定性复现）。
  - 检测运营点（可选）：与 f8/f12/f13 三 ckpt 取 **median4 功率集成**，决策阈值按 val 8k 选点=115W → 稠密 test **P .907 / R .880**；test 侧最优 t=95 处 .906/.906（仅诊断，选点违规不作验收）。协议点 500W：.950/.778。
  - 边界：主结论单 seed(42)，seed43 复跑（tune_f13_seed43）MAE 7.24/前沿 .865——容量收益稳、精调幅度对 seed 敏感 ±0.5W；待多 seed 复核（遗留项见 `REPORT_TEST.md`）。
- `tune_final_f3_cosine`（前稳定版，保留备查）：MAE 8.58W / R² .723 / F1 .853。

## 5. 重大工程教训

- 2 核/3.9GB CPU 沙箱上 **torch cu130 轮子 + seq256 会线性涨内存直至 OOM**（300 步实测 ~18MB/步），seq128 无恙；CPU 环境勿并行硬扛多泳道，先测单泳道 RSS 峰值再定并发度。
