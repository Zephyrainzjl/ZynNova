# PolyPrediction + PolyGen：储能高分子新增模块

本增量实现以 Li 等人在 *Nature Materials* 发表的 “Enhanced energy storage in
high-entropy ferroelectric polymers” 为物理主线，同时面向不限于
P(VDF-TrFE-CTFE) 的公开数据、自建数据和多工况数据。

> 论文：https://doi.org/10.1038/s41563-025-02211-z

## 新增结构

```text
src/zynnova/ml/
├── polymer_utils.py
├── prediction/PolyPrediction/
│   ├── config.py          # 储能属性/工况规范
│   ├── data.py            # 公共或自建数据、家族隔离划分、缺失标签
│   ├── model.py           # PSMILES Transformer + 重复单元 GNN + 工况融合
│   ├── normalizer.py      # 带物理域变换的掩码标准化
│   ├── physics.py         # 构型熵与储能一致性损失
│   ├── trainer.py         # 异方差多任务训练
│   ├── calibration.py     # split-conformal 校准
│   ├── predictor.py       # MC-dropout 推理与置信区间
│   └── screening.py       # 约束满足概率与快速筛选
└── generation/PolyGen/
    ├── config.py
    ├── data.py
    ├── model.py           # 条件掩码离散流 Transformer
    ├── trainer.py         # 连续时间掩码流目标 + classifier-free dropout
    ├── sampler.py         # CFG 采样、预测器复筛、物理/多样性排序
    └── validation.py      # 化学价态解析与聚合端口检查
```

模型分别注册为 `prediction/poly_prediction` 与 `generation/poly_gen`。

## 论文物理如何进入模型

论文给出的核心不是“化学键种类越多越好”，而是不同键型的摩尔分数必须足够均衡：

\[
\frac{\Delta S_{\mathrm{conf}}}{R}=-\sum_i c_i\ln c_i.
\]

补充材料以 \(1.5R\) 作为高熵参考线；五种等摩尔键型的上限为
\(\ln 5=1.609\)。因此代码会显式统计键型比例（按论文忽略常见的 C–C
单键），同时报告键型数、\(\Delta S_{\mathrm{conf}}/R\) 和相对 \(1.5R\)
的裕量。`PolyGen` 可以设置硬熵阈值，但不会把“至少五种键”错误地当作充分条件。

预测训练还加入以下软约束：

- \(P_m\ge P_r\)；
- 有测试电场时，预测击穿场不应低于测试电场；
- 由 \(U_d=\int_{P_r}^{P_m}E\,dP\) 得到
  \(U_d\le E_{\max}(P_m-P_r)\)。这里采用的单位恰好满足
  `MV m^-1 × C m^-2 = J cm^-3`；
- PSMILES 直接计算的构型熵与构型熵预测头保持一致。

论文中质子辐照引入 C=C、C=O、C–O 和 O–H，改变 3/1 螺旋与 all-trans
构象能差，并通过随机场、介电背景、链间距和结晶度共同影响畴尺寸、
极化、漏电与击穿。相应变量没有被压成一个不可解释的“剂量特征”，而是可分别作为
结构属性、预测目标或实验工况输入。

## 默认属性和工况

所有目标均支持缺失标签，且可以在配置里裁剪或扩展。

| 类别 | 规范字段 | 物理作用 |
|---|---|---|
| 基础 | `glass_transition_temperature_K` | 链段运动与加工窗口 |
| 基础 | `bandgap_eV` | 漏电和电子击穿代理量 |
| 介电 | `dielectric_constant` | 可极化性、退极化场 |
| 介电 | `dielectric_loss_tangent` | 交流损耗 |
| 绝缘 | `breakdown_strength_MV_m` | 可用电场上限 |
| 绝缘 | `leakage_current_density_A_m2` | 导电损耗 |
| 极化 | `maximum_polarization_C_m2` | \(P_m\) |
| 极化 | `remanent_polarization_C_m2` | \(P_r\) 与滞回损耗 |
| 储能 | `recoverable_energy_density_J_cm3` | \(U_d\) |
| 储能 | `efficiency` | \(U_d/(U_d+U_{\mathrm{loss}})\) |
| 高熵 | `configurational_entropy_R` | \(\Delta S_{\mathrm{conf}}/R\) |
| 形态 | `crystallinity_fraction` | 晶相比例 |
| 形态 | `interchain_spacing_A` | 链间相互作用与偶极密度 |
| 构象 | `helix_trans_energy_delta_eV` | 3/1 螺旋–all-trans 能差 |

默认实验工况为 `temperature_K`、`log10_frequency_Hz`、
`electric_field_MV_m`、`proton_dose_Mrad`、`film_thickness_um`、
`crystallinity_fraction`、`random_field_strength_MV_m` 与
`background_dielectric_constant`。介电常数、损耗、极化、漏电和储能密度
必须与工况同表保存，不能把不同温度/频率/电场的数值当成同一个无条件标签。

## PolyPrediction

`PolyPredictionNetwork` 同时使用三条信息通路：

1. 数据集拟合词表的 PSMILES Transformer，学习长程化学上下文；
2. 不依赖 PyG 的重复单元消息传递图网络，学习原子、键、聚合端口和共轭信息；
3. 带显式缺失掩码的实验工况编码器。

三路表示经过门控融合后，每个任务输出均值与对数方差。训练损失按任务分别聚合，
避免样本多的任务吞没稀疏但关键的击穿/储能任务。推理同时合并异方差噪声与
MC-dropout 模型不确定性；可再用独立校准集拟合 `ConformalCalibrator`。

### 自建 CSV

```python
from zynnova.ml.prediction.PolyPrediction import (
    PolyPredictionConfig,
    PropertySpec,
    train_poly_prediction,
)

config = PolyPredictionConfig()
config.model.property_specs = (
    PropertySpec("dielectric_constant", "1", "log", lower_bound=0.0),
    PropertySpec(
        "efficiency",
        "1",
        "logit",
        lower_bound=0.0,
        upper_bound=1.0,
    ),
    PropertySpec(
        "recoverable_energy_density_J_cm3",
        "J cm^-3",
        "log1p",
        lower_bound=0.0,
    ),
)
config.data.dataset = "polymer_table"
config.data.dataset_kwargs = {
    "path": "/data/polymer_energy.csv",
    "psmiles_column": "psmiles",
    "id_column": "sample_id",
    "target_columns": {
        "dielectric_constant": "epsilon_r",
        "efficiency": "eta_fraction",
        "recoverable_energy_density_J_cm3": "Ud_J_cm3",
    },
    "condition_columns": {
        "temperature_K": "temperature_K",
        "log10_frequency_Hz": "log10_frequency_Hz",
        "electric_field_MV_m": "field_MV_m",
        "proton_dose_Mrad": "dose_Mrad",
    },
    "split_column": "split",
}
result = train_poly_prediction(config)
```

对于自建表，效率必须保存为 0–1 而非百分数；频率建议预先取
`log10(frequency_Hz)`；同一聚合物的不同工况可以占多行。若未提供
`split`，代码按规范化 PSMILES 家族分组，不把同一家族随机散入训练集和测试集。

### 公共数据

`dataset="transpolymer"` 可直接使用现有 TransPolymer 适配器：

```python
config.data.dataset = "transpolymer"
config.data.dataset_kwargs = {
    "file_name_contains": "dielectric",
    "target_columns": {"dielectric_constant": "dielectric_constant"},
    "condition_columns": {"log10_frequency_Hz": "log10_frequency_Hz"},
}
```

建议的数据分层如下：

- PI1M 或 TransPolymer：结构语言预训练；
- Polymer Genome/公开 DFT 表：带隙、介电与热性质；
- 频率依赖介电公开表：\(\epsilon_r(f)\) 和损耗；
- 论文 Source Data：P(VDF-TrFE-CTFE) 在剂量、温度、频率和电场上的精细曲线；
- 自建实验/DFT/MD：目标配方、加工与形态数据。

论文 Source Data 是非常有价值的工况增强数据，但体系数太少，不能单独证明模型对
全化学空间有泛化能力。合并数据时应保留 `provenance`、单位、测试方法和不确定度，
并先确认各数据集许可。

### 快速概率筛选

```python
from zynnova.ml.prediction.PolyPrediction import (
    PropertyConstraint,
    load_poly_predictor,
    predict_polymers,
    screen_predictions,
)

predictor = load_poly_predictor("/runs/poly-prediction/checkpoints/best.pt")
predictions = predict_polymers(
    predictor,
    psmiles_candidates,
    conditions={
        "temperature_K": 298.15,
        "electric_field_MV_m": 600.0,
        "log10_frequency_Hz": 3.0,
    },
)
ranked = screen_predictions(
    predictions,
    [
        PropertyConstraint("bandgap_eV", lower=4.0),
        PropertyConstraint("breakdown_strength_MV_m", lower=600.0),
        PropertyConstraint("recoverable_energy_density_J_cm3", lower=30.0),
        PropertyConstraint("efficiency", lower=0.80),
    ],
)
```

排序依据为各约束的联合满足概率，而不是只按预测均值排序，因此会自动降低
“均值很好但不确定性很大”的候选优先级。

## PolyGen

`PolymerMaskedFlow` 是属性条件化的掩码离散流：

- 在连续时间 \(t\in(0,1)\) 上把真实 PSMILES 随机流向吸收态 `[MASK]`；
- Transformer 学习反向条件速度/去噪分布；
- 训练时随机丢弃全部条件，采样时使用 classifier-free guidance；
- 单独学习长度分布，随后并行、置信度驱动地逐步解掩码；
- 无条件属性头用于约束跟随评分，避免把输入属性直接复制到评分头。

与只做单属性自回归字符串生成相比，这一实现原生支持多属性、多工况、并行离散流、
独立预测器复筛和不确定性惩罚。它仍需在用户数据上做消融和外部基准，架构先进本身
不等于未经验证的指标优势。

```python
from zynnova.ml.generation.PolyGen import (
    PolyGenConfig,
    PolyGenSamplingConfig,
    generate_polymers,
    load_poly_generator,
    train_poly_gen,
)
from zynnova.ml.prediction.PolyPrediction import (
    PropertyConstraint,
    load_poly_predictor,
)

train_result = train_poly_gen(PolyGenConfig())
generator = load_poly_generator(train_result.best_checkpoint)
predictor = load_poly_predictor("/runs/poly-prediction/checkpoints/best.pt")

generated = generate_polymers(
    generator,
    {
        "bandgap_eV": 5.0,
        "dielectric_constant": 20.0,
        "recoverable_energy_density_J_cm3": 35.0,
        "efficiency": 0.85,
        "configurational_entropy_R": 1.55,
    },
    process_conditions={
        "temperature_K": 298.15,
        "electric_field_MV_m": 600.0,
    },
    config=PolyGenSamplingConfig(
        num_candidates=40,
        guidance_scale=3.5,
        minimum_configurational_entropy_R=1.5,
    ),
    predictor=predictor,
    constraints=[
        PropertyConstraint("breakdown_strength_MV_m", lower=600.0),
        PropertyConstraint("efficiency", lower=0.80),
    ],
)
```

每个保留候选必须通过：

1. RDKit 化学解析和价态检查；
2. 恰好两个聚合端口（可配置）；
3. 可选的构型熵硬阈值；
4. 独立 `PolyPrediction` 的多目标概率复筛；
5. 不确定性惩罚与集合多样性选择。

输出包含具体 PSMILES、完整 `PolymerRecord`、预测性质、不确定性、键构型熵、
端口数和理论检查结果，可继续送入已有的构象、DFT、MD 或相场流程。

## 推荐验证顺序

1. **数据审计**：统一单位和条件，删除重复/冲突记录，按高分子家族和来源隔离测试集。
2. **预测基准**：报告每目标 MAE/RMSE、校准误差、90% 区间覆盖率和约束召回率；
   与单序列、单图和无物理损失版本做消融。
3. **生成基准**：报告化学有效率、双端口率、唯一性、新颖性、条件命中率和家族外距离。
4. **计算验证**：带隙/偶极/构象能差用 DFT；形态、链间距和击穿代理用 MD/相场。
5. **实验闭环**：优先合成高可行概率且模型不确定性低的候选，再把失败样本回流训练。

生成结果是可检验的材料假设，不是合成可行性或实际储能性能的保证。特别是辐照剂量
并不唯一决定新键比例、相分离、结晶度和缺陷分布；若没有相应表征标签，模型不能从
一个剂量数字中恢复这些隐变量。

