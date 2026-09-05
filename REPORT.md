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
5. **模型选择本身是高影响「元超参」**：val 噪声大时按 val MAE 单调最小选点会放大运气成分（F2 best_ep7 次优 checkpoint → 稠密 test EE −28.6%；同曲线 ep12 实际更优）。候选改进：val F1 或多目标选点。

## 4. 推荐稳定版本

- **`tune_final_f3_cosine`（2026-09-05）**：anchor + dropout 0 + cosine lr(5e-4, T_max=20)，30k train / 8k val / 30k 稠密 test，CPU。
  - 稠密 test(n=30000)：**MAE 8.58W，RMSE 114.6W，R² 0.723，SAE 0.195，P 0.907 / R 0.805 / F1 0.853，energy_error −19.5%**；较 anchor 同口径 MAE −26.5%，为目前最好结果。
  - 复现：`OMP_NUM_THREADS=2 ./.venv/bin/python scripts/train.py --config reports/tune_final_f3_cosine/config.yaml --data-path data/ukdale_prepared.npz --out reports/tune_final_f3_cosine`（Windows/conda 用等价命令）。
  - 边界：单 seed、子采样口径下结论；置信度中等，待多 seed 与连续窗口口径复核（遗留项见 `REPORT_TEST.md`）。

## 5. 重大工程教训

- 2 核/3.9GB CPU 沙箱上 **torch cu130 轮子 + seq256 会线性涨内存直至 OOM**（300 步实测 ~18MB/步），seq128 无恙；CPU 环境勿并行硬扛多泳道，先测单泳道 RSS 峰值再定并发度。
