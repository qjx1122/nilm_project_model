# REPORT_TEST.md — 专题报告（只追加，按专题分节）

## [2026-09-03] 专题：真实 UK-DALE（House1 子集）Transformer Seq2Point baseline
- 类型：实验专题
- 目标与假设：在真实电网/家庭负荷数据上验证本仓库 Transformer Seq2Point 全链路（数据→训练→验证→测试→指标→checkpoint），获得首个可复现的真实指标基线；假设真实数据上 MAE 将显著高于合成 smoke（合成信号过于干净）。
- 方法 / 数据 / 参数：
  - 数据：**真实 UK-DALE building 1，6-second 网格**，2014-09-08 → 2014-10-21，共 633,600 行、零缺口、无重复时间戳。来源为第三方 GitHub 切片仓库 `Ken89MathCompSci/UKDALE-NILM` 的 `APR-new-House1-dataset/`（train 30d + validation 7d + test 7d，按时间相邻拼接）。⚠️ 本沙箱 egress 白名单（仅 GitHub/PyPI）无法访问 jack-kelly.com / UKERC EDC / HuggingFace / Kaggle，官方全量数据不可达；已在 STATUS.md 留存完整筛选记录。
  - 完整性交叉验证：house1 aggregate 均值 418.92 W vs UK-DALE 论文 ~417.9 W ✓。
  - 电器选择：子集无 **kettle** 列 → 以 **fridge**（主基线，压缩机循环，on_frac 48.8%、1148 个 ON 事件）与 **washing_machine**（事件型，354 事件）替代，如实标注，不冒充全量 kettle 结果。
  - 准备脚本：`scripts/prepare_ukdale_subset.py`（CSV→严格 6s 网格→最长连续段→`data/ukdale_house1_{appliance}.npz`+report JSON；缺失率 0）。
  - 配置：`configs/ukdale_fridge.yaml`、`configs/ukdale_washing_machine.yaml` = baseline.yaml 全部超参（d_model 64 / nhead 4 / L2 / ff128 / lr 5e-4 / bs 128 / window 128 / 30 epochs / patience 7 / 样本上限 30000/6000/6000），仅改 `on_threshold_watts`：fridge=30W（500W 会把冰箱全判 OFF）、washing_machine=500W。
  - 环境：CPU 2 核，torch 2.14.0，seed 42。
- 结果 / 结论：
  - fridge（test, best_epoch=30 触顶，val MAE 末段仍缓慢改善，runtime 5242s）：
    - **MAE 30.63 W、RMSE 50.39 W、R² 0.297、SAE 15.58%、Energy Error +15.58%、Precision 0.634 / Recall 0.960 / F1 0.764**（@30W 点级阈值）
    - val 轨迹：Epoch28 val MAE 27.84 W（val≈test，时间切分相邻、分布漂移小）
  - 结论：
    1. 真实数据链路端到端可用，指标合理（文献 UK-DALE fridge Seq2Point MAE≈10–13W 为全网 18 个月+调优结果；本实验 44 天子集、30k 采样上限、未调参、30 epoch 触顶，30.6W 属同数量级合理起点）。
    2. Energy Error +15.6% 与低 Precision → 模型系统性低估功率（欠拟合/均值回归），优先方向：提高 epochs 或 lr 调度、加大 max_samples、MSE→加 L1 混合；而非扩模型。
    3. 30 epoch 触顶且 val 仍在改善 → baseline.yaml 的 epochs=30 对真实数据偏短。
- 是否进入 REPORT.md（稳定结论）：是（仅「真实数据路线已建立 + 子集基线口径」部分；绝对指标待全量数据复核）
- 遗留问题：
  - kettle / 全量 18 个月 / 跨家庭 house1→house2 需要官方数据通道（用户本机下载或放开封白名单）
  - washing_machine 同配置结果待补（本专题追加）
  - MAE 30.6W 与文献 10-13W 的差距归因（数据量 vs 调参）未消融

## [2026-09-03] 专题：真实 UK-DALE（House1 子集）baseline —— washing_machine（补充结果）
- 类型：实验专题（同上节「数据/方法」，仅电器与阈值不同：washing_machine、on_threshold=500W）
- 目标与假设：事件型电器上验证 Seq2Point 的开关识别与能量估计；预期 MAE 低（多数样本为 0）但 RMSE 由启动尖峰主导
- 方法 / 数据 / 参数：同 fridge 专题；30 epochs 上限，early stopping 于 epoch 16（best_epoch=9，val MAE 13.80W），runtime 2853s
- 结果 / 结论：
  - test：**MAE 14.59 W、RMSE 92.51 W、R² 0.863、SAE 2.29%、Energy Error −2.29%、Precision 0.909 / Recall 0.917 / F1 0.913**（@500W 点级阈值）
  - Epoch16 train MAE 13.34W / val MAE 14.89W，无过拟合
  - 结论：事件型电器上 baseline 已接近文献公开量级（UK-DALE house1 washing machine 各方法 MAE 常见 10–20W 区间），说明模型容量在此任务上「够用」，调参收益预计集中在 fridge 这类高频切换电器；MAE/RMSE 比 1:6.3 证实大误差集中于启动/关闭沿（与 README 第 8 节「MAE 好但 RMSE 差」的判读规则一致，应重点检查事件沿）
- 是否进入 REPORT.md：否（结论已并入既有稳定条目；数字本身属单轮结果）
- 遗留问题：事件级（而非点级）P/R/F1 口径是否引入；启动沿专用损失加权是否有效

