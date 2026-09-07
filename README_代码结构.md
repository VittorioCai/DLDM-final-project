# Shortcut Learning 代码工作区

当前状态：代码已经实现并运行，不再是结构草案。Python 环境使用 `/opt/anaconda3/bin/python`，本机 MPS 可用。

## 代码结构

```text
05_代码工作区/
├── configs/
│   ├── base.yaml       # 已锁定的正式参数
│   ├── smoke.yaml      # 1 epoch 小数据管线检查
│   └── pilot.yaml      # 本地 10 epoch 实验
├── src/shortcut_learning/
│   ├── data.py         # 平衡划分、颜色、嵌套反例、DFR 子集
│   ├── models.py       # LeNet 与 MNIST ResNet-18
│   ├── training.py     # ERM、GroupDRO、DFR
│   ├── metrics.py      # 十色反事实指标
│   ├── experiment.py   # 单次 run 与产物保存
│   ├── batch.py        # 依赖排序的实验矩阵
│   └── analysis.py     # 汇总、N80 与 mean±SD 图
├── tests/              # 自动测试
├── train.py            # 运行一个配置
├── evaluate.py         # 从 checkpoint 重新评估
├── run_matrix.py       # 批量运行并跳过 completed runs
├── analyze.py          # 汇总和作图
├── results/            # 最终 230-run 矩阵的汇总结果（入库共享）
├── outputs_smoke/      # 真实 MNIST smoke 产物（不入库）
├── outputs_pilot/      # 本地 MPS pilot 产物（不入库）
└── outputs_final/      # AutoDL 最终矩阵原始输出（1.7G，不入库）
```

## 已实现的实验约束

- 每个训练集固定 50,000 张、每类 5,000 张。
- N 个冲突样本替换 aligned 样本，不增加总量。
- 相同 seed 下预算嵌套。
- 每个预算保持全局颜色边际不变。
- 十种颜色使用近似等亮度 YIQ palette。
- GroupDRO 训练组为 `digit × aligned/conflict`。
- DFR 使用 N 个 conflict + N 个 digit-matched aligned 样本。
- 主评估为十色反事实测试，不依赖 Grad-CAM。

## 常用命令

在本文件所在目录执行：

```bash
/opt/anaconda3/bin/python -m pytest -q

/opt/anaconda3/bin/python train.py \
  --config configs/base.yaml \
  --method erm --model lenet --budget 500 --seed 0 --device mps

/opt/anaconda3/bin/python run_matrix.py \
  --config configs/base.yaml \
  --models lenet \
  --methods erm,groupdro,dfr \
  --budgets 0,50,100,250,500,1000,2500,5000 \
  --seeds 0,1,2,3,4 \
  --device mps

/opt/anaconda3/bin/python analyze.py \
  --output-root outputs_pilot \
  --destination ../06_结果记录/pilot_analysis
```

`run_matrix.py` 发现已有 `status=completed` 的 `metrics.json` 时会跳过该 run；崩溃不会自动重试。

## 单次输出

```text
outputs/{model}/{method}/N{budget}/seed{seed}/
├── config_resolved.yaml
├── train_log.csv
├── metrics.json
├── checkpoint.pt       # 按 storage policy 保留
└── environment.txt
```

## 最终 CUDA 执行环境

最终 230-run 矩阵在 AutoDL 上运行，硬件为 NVIDIA GeForce RTX 4090，CUDA 版本为 12.4。原始输出在本地 `outputs_final/`（不入库），汇总结果的入库副本在 `results/`。早期云端工作流的遗留文件已移出仓库，仅保留在本地。

## 当前实测

- LeNet：ERM 40、GroupDRO 40、DFR 35，全部 completed。
- ResNet-18：ERM 40、GroupDRO 40、DFR 35，全部 completed。
- Canonical results：AutoDL / RTX 4090 / CUDA 12.4，共 230 runs；汇总见 `results/final_analysis/`（仓库内）与 `../06_结果记录/final_analysis/`（本地）。
