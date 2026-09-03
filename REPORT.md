# REPORT.md — 稳定结论（算法路线 / 稳定实验结论 / KPI 口径 / 推荐稳定版本）

> 仅沉淀「可复现且不随实验轮次漂移」的结论；单轮数字进 REPORT_TEST.md。

## 1. 算法路线（已验证可行）
- Transformer Encoder **Seq2Point**（Input Projection → Sinusoidal PE → pre-norm Encoder(d_model64/nhead4/L2/ff128/drop0.1) → 中心点回归头）
- 输入 = 128 点（12.8 min @6s）家庭总负荷窗口；输出 = 中心时刻目标电器功率
- 归一化：仅用 Train 段拟合 mean/std，线性缩放后 MSE 损失；推理后反归一化为瓦特
- 数据接入：`{aggregate, target}` 一维对齐序列 NPZ；`scripts/prepare_ukdale_subset.py` 负责从「timestamp,aggregate,<appliance>...」CSV 生成（严格 6s 网格、缺失 ffill≤10 行、取最长连续段并输出审计 JSON）

## 2. KPI 口径（本仓库现行定义）
- MAE / RMSE / R²：瓦特域、点级（每样本=中心时刻）
- SAE = |Σy−Σŷ| / Σy；Energy Error 同式带符号
- Precision / Recall / F1：**点级**功率阈值分类（y≥thr 视为 ON），非事件级；fridge 用 thr=30W，功率型电器用 thr=500W
- 调参只看 Validation；Test 只在锁定配置后跑一次（baseline.yaml 的 70/15/15 时间前切分）

## 3. 稳定实验结论
- 合成信号 smoke（`scripts/run_smoke.py`）只能验证链路正确性，不可作为效果结论（其 MAE≈60W、R²≈0.91 均为人造信号产物）
- 真实 UK-DALE house1 44 天子集上，未调参 baseline 即可获得 val≈test 的稳定性（时间相邻切分下分布漂移小）；细节见 REPORT_TEST.md 2026-09-03 专题
- 环境经验：download.pytorch.org / raw.githubusercontent.com / 对象存储域名在受限沙箱不可达时，`github.com + api.github.com + codeload.github.com + PyPI` 是可用传输通道；模型权重/数据文件 ≤100MB 可从仓库直接浅克隆获取

## 4. 推荐稳定版本
- 当前推荐：commit `1296c3a` 之后的 main 线（新增 prepare_ukdale_subset.py + ukdale_*.yaml 专项配置 + .gitignore 卫生）；`configs/baseline.yaml` 参数对真实数据偏短（epochs 触顶），进入调参阶段时优先动 epochs/lr 而非容量
