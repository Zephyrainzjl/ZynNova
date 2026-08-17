# Polymer Physical Discovery

这个子包把 ZynNova 已有的 JouleWeave、PolyPrism、PolyLoom、数据注册表和
动力学工作流连接成一个可审计的高分子物理性质发现闭环。项目中 MLFF 的真实路径是
`src/zynnova/ml/mlff`，因此模块路径为：

```python
from zynnova.ml.mlff.poly_discovery import PolymerDiscoveryCampaign
```

导入后也会注册为 `discovery/polymer_physics`，可通过 ZynNova 的模型注册表创建。
本子包内部把势能、MD、NEB 和生成相关重依赖延迟到具体操作；纯统计实现只依赖
NumPy。不过当前项目的 `zynnova.ml` 父包会主动导入现有 MLFF，因此常规安装和导入
仍建议使用 `pip install -e ".[ml-all]"`，与项目原有依赖契约保持一致。

## 设计边界

本模块不会把预测相关性包装成“物理定律”，也不会让经典神经网络势越权预测能带。
不同问题使用不同的可信 oracle：

| 问题 | 首选证据 | 本模块的角色 |
|---|---|---|
| 能量、力、应力、构象差、势垒、MD | DFT 标注后微调的 JouleWeave；委员会不确定性 | 训练、加载、ASE 计算、MD、NEB、主动选帧 |
| HOMO、LUMO、带隙、电子亲和能 | 同一方法的 DFT/HSE06 或实验；可加经验证的电子预测器 | 配对基团替换、跨环境稳定性、介导和反事实检验 |
| 储能密度和效率 | 条件一致的 P–E 回线，配合介电、击穿和多尺度模拟 | 正确积分、稀疏规律、机制链和下一轮实验选择 |
| 高熵效应 | 组成/键型熵、势垒分布、随机场、剩余极化和过程变量 | 检验非单调关系，不把“熵越高越好”写成先验结论 |
| 压电和交联效应 | 构象扫描/NEB、相能差、匹配结晶度的实验 | 检验局域构象异质性和能量景观变平的介导路径 |
| 新结构 | PolyLoom 化学有效性 + PolyPrism 独立预测 + 机制适用域 | 机制约束重排、拒绝外推、送入主动验证 |

## 为什么默认使用 JouleWeave

JouleWeave 已经是本项目的原生 O(3) 等变势，具备能量/力/应力接口、短程 ZBL
保护、色散、可选全局 QEq、电荷和偶极头，并已有 ASE、MD、NEB、微调和冻结骨干
训练链路。相比另接一个模型，这能保留项目统一的工作空间、数据字段和检查点契约。
对于处在所选 MACE-OFF 版本训练域内的有机体系，可以把这个短程模型作为独立基线；
`PolymerMechanismSimulator` 也接受任意 ASE calculator。反应、带电、强场、长程
极化或特殊元素体系必须重新验证适用域，不能因模型“更大”而跳过标注。

提供三个 JouleWeave 高分子预设：

- `condensed`：默认。色散开启，适用于致密无定形/晶态聚合物的结构、力学和构象采样。
- `electrostatic`：开启 QEq，可结合统一电荷分区和偶极标签，用于偶极与介电机制。
- `reactive`：更强调短程保护，适用于交联、断键和辐照后局部反应；必须包含反应路径数据。

`PotentialCommittee` 使用多个独立检查点的能量/力方差并结合结构多样性选择新 DFT
帧。这是模型外推告警，而不是误差的严格概率保证。

## 1. 训练并使用高分子神经网络势

输入轨迹应是 ASE 可读文件，每帧至少包含总能和原子力；周期体系建议同时提供应力。
训练/验证/测试必须按高分子家族、轨迹或构象簇划分，不能随机打散相邻 MD 帧造成泄漏。

```python
from zynnova.ml.mlff.poly_discovery import (
    PolymerMechanismSimulator,
    PolymerPotentialConfig,
    load_polymer_potential,
    train_polymer_potential,
)

potential_config = PolymerPotentialConfig(
    preset="condensed",
    dataset="trajectory",
    dataset_root="data",
    dataset_kwargs={
        "path": "data/polymer_train.extxyz",
        "energy_key": "energy",
        "forces_key": "forces",
        "stress_key": "stress",
    },
    fine_tune_checkpoint="checkpoints/jouleweave-foundation.pt",
    freeze_backbone_epochs=10,
    force_centric=True,
    epochs=200,
    run_name="polymer-condensed-v1",
)
training = train_polymer_potential(potential_config)

loaded = load_polymer_potential(training.best_checkpoint)
simulator = PolymerMechanismSimulator(loaded)
relaxed = simulator.relax(periodic_polymer_cell)
profile = simulator.energy_profile(torsion_scan_frames)
neb = simulator.neb(initial_crosslink_state, final_crosslink_state)
```

电荷/偶极监督必须记录同一种分区定义，例如 DDEC6，不能混合 Bader、Hirshfeld 和
DDEC 标签。`electrostatic` 预设不会凭空产生可信偶极：只有在标签、边界条件和独立
验证齐全时，`dipole_probe` 才能用于体相静态介电估计。

## 2. 从数据主动发现物理机理

“主动发现”不是对特征重要性排序，而是以下闭环：

1. 将结构特征、过程条件、介变量、目标、误差、环境和干预信息分开记录。
2. 在多个数据集/高分子家族/计算保真度内做稀疏稳定选择和符号候选搜索。
3. 检查效应符号是否跨环境一致，并给出分层 bootstrap 区间。
4. 对基团做匹配骨架比较，对文献路径做介导检验。
5. 每条假设同时输出反证实验；无干预时只写“关联”。
6. 用模型不确定性、机制信息增益、新颖性、多样性和成本选择下一批 DFT/NEB/实验。
7. 新结果回写后重新发现；只有跨家族验证或干预支持才提升证据等级。

```python
from zynnova.ml.mlff.poly_discovery import (
    MechanismDiscoveryConfig,
    Observation,
    PolymerDiscoveryCampaign,
    save_discovery_report,
)

observations = [
    Observation(
        sample_id=row["id"],
        features={
            "bond_configurational_entropy_R": row["bond_entropy_R"],
            "group_carbonyl": row["carbonyl_fraction"],
            "crystallinity_fraction": row["crystallinity"],
        },
        mediators={
            "barrier_standard_deviation_eV": row["barrier_std_eV"],
            "remanent_polarization_C_m2": row["Pr_C_m2"],
        },
        conditions={
            "electric_field_MV_m": row["field_MV_m"],
            "temperature_K": row["temperature_K"],
        },
        targets={
            "recoverable_energy_density_J_cm3": row["Ud_J_cm3"],
        },
        uncertainty={"recoverable_energy_density_J_cm3": row["Ud_std"]},
        environment=row["polymer_family"],
        fidelity=row["fidelity"],
        intervention={"proton_dose_Mrad": row["dose"]}
        if row["irradiated"]
        else {},
        provenance={
            "doi": row["doi"],
            "psmiles": row["psmiles"],
            # 只有受控/随机干预设计才设置此标记。
            "controlled_intervention": row["controlled_intervention"],
        },
    )
    for row in rows
]

campaign = PolymerDiscoveryCampaign(
    observations=observations,
    discovery_config=MechanismDiscoveryConfig(
        bootstrap_repeats=512,
        environment_invariance_threshold=0.67,
    ),
)
report = campaign.discover(
    "recoverable_energy_density_J_cm3",
    control_names=(
        "crystallinity_fraction",
        "electric_field_MV_m",
        "temperature_K",
    ),
)
save_discovery_report(report, "results/storage_mechanism.json")
save_discovery_report(report, "results/storage_mechanism.md")
```

`FeatureEffect` 给出稀疏系数、bootstrap 区间、选择频率、符号一致性和环境一致性。
`MediationResult` 检验“结构 → 势垒/极化等介变量 → 性能”，`MatchedPairEffect`
比较控制已测混杂变量后的基团替换，`DiscoveredLaw` 输出小型可审计表达式及留环境
验证 R²。所有符号关系都在标准化变量中，独立验证前不能称为量纲严格的物理定律。

### 2.1 神经—符号物理学习

`physics_learning` 在原有稳定选择之上增加五层物理约束：

1. 用七个 SI 基本量纲和 Buckingham Π 群限制候选搜索，并把单位未知明确标为
   `unchecked`，而不是猜测单位。
2. 用稀疏 RBF-KAN 作为可微非线性 oracle；通过 PyTorch 精确自动微分的混合
   Hessian 建立特征相互作用图，再把弱相互作用子图拆开搜索。若没有 PyTorch，
   自动使用可审计的二次 Hessian 回退。
3. 并列接入五类符号后端：原生 NumPy 搜索、2026 年 PSE/PSRN 的 GPU 并行枚举、
   PySR 的进化 Pareto 搜索、PhySO 的量纲引导强化学习，以及 2025 年 PhyE2E 的
   导数分治、Transformer、MCTS 和遗传编程精修。外部后端全部延迟导入，任何失败
   都写入 `backend_status`。
4. 优先留出完整高分子家族/数据环境；按验证 R²、表达式复杂度、量纲一致性和跨环境
   一致性排序，并另外保留精度—复杂度 Pareto 前沿。
5. 对介电弛豫、结晶、扩散或极化时间序列提供积分弱形式稀疏动力学发现，避免直接
   对噪声轨迹求导；也可显式切换到 PySINDy。

轻量路径仍只依赖 NumPy。KAN 安装：

```bash
pip install -e ".[physics-discovery]"
```

PySR、PhySO、PySINDy 和 SymPy 全部安装：

```bash
pip install -e ".[physics-symbolic]"
```

```python
from zynnova.ml.mlff.poly_discovery import (
    ELECTRIC_FIELD,
    PRESSURE,
    PhysicsLearningConfig,
    PolymerDiscoveryCampaign,
    VariableSpec,
)

campaign = PolymerDiscoveryCampaign(
    observations=observations,
    physics_config=PhysicsLearningConfig(
        enabled=True,
        oracle_backend="kan",
        symbolic_backends=("native", "pse", "pysr", "physo"),
        workspace_root="runs/polymer-physics",
        monotonic_constraints={"breakdown_strength_MV_m": 1},
        pse_symbol_layers=2,  # 显存有限时先用 2；高显存可用默认 3
        pse_time_limit_seconds=600,
    ),
)
report = campaign.discover(
    "recoverable_energy_density_J_cm3",
    include_advanced_physics=True,
    variable_specs={
        "electric_field_MV_m": VariableSpec(
            "electric_field_MV_m", "MV/m", ELECTRIC_FIELD
        ),
        "recoverable_energy_density_J_cm3": VariableSpec(
            "recoverable_energy_density_J_cm3", "J/cm^3", PRESSURE
        ),
    },
)

physics = report.physics_learning
print(physics.best_equation.expression)
print(physics.interaction_decomposition.edges[:5])
print(physics.diagnostics["pareto_equation_ids"])
```

PSE 是截至 2026 年已正式发表且有 MIT 官方实现的并行枚举后端。它复用表达式公共
子树，并在 GPU 上同时评估大量候选；对 RTX 级显卡应从 `pse_symbol_layers=2`、
较少的 `pse_inputs` 和 `SemiSub`/`SemiDiv` 运算符开始，再依据显存提高规模。
默认关闭依赖工作目录的 DR mask；若已按官方工具生成与层数、输入槽和算符完全匹配
的 mask，可设置 `pse_use_dr_mask=True` 和 `pse_dr_mask_dir=...` 以节省显存。
PSE 搜索阶段本身不施加 SI 量纲，所以本适配器对其 Pareto 表逐式做事后量纲检查；
量纲未知时明确标记为 `unchecked`；量纲已知且不一致的候选默认由
`reject_unit_inconsistent=True` 排除。

PhyE2E 使用作者发布的代码和权重，不复制进 ZynNova。下载并核对官方 release 后，
在配置中增加：

```python
physics_config = PhysicsLearningConfig(
    enabled=True,
    symbolic_backends=("native", "phye2e"),
    phye2e_repository="/external/PhysicsRegression",
    phye2e_checkpoint="/external/checkpoints/model.pt",
    phye2e_device="cuda",
)
```

官方 PhyE2E 预训练规模远高于普通工作站，因此它是显式启用的高算力后端，不是
安装包的默认步骤。模块只适配其公开 API；检查点许可、来源、哈希和设备要求应随
实验记录保存。

时间序列规律可独立调用：

```python
from zynnova.ml.mlff.poly_discovery import discover_sparse_dynamics

dynamics = discover_sparse_dynamics(
    time_ps,
    polarization_and_crystallinity,
    state_names=("polarization", "crystallinity"),
    backend="native",
    weak_form=True,
    polynomial_degree=2,
)
for equation in dynamics.equations:
    print(equation.expression, equation.train_r2)
```

混合二阶导数说明模型中存在非加和耦合，不自动等价于微观因果作用；符号式也必须在
未参与发现的聚合物家族、DFT 条件或实验批次上复现，才能用于生成约束。

### 推荐数据组合

`public_dataset_plan()` 返回用途和保真度明确的数据顺序，而不是自动下载并混合标签：

- PI1M：化学空间预训练和新颖性，不能作为物性证据。
- TransPolymer/polyVERSE：多任务表征和性质基线，需保留各源测量条件。
- Huan 高分子 DFT 数据：统一方法的带隙和介电机制基准。
- RadonPy：无定形全原子 MD 与跨模拟引擎验证。
- 高熵铁电和交联压电论文的 source data：窄域但包含可识别干预，适合机制验证。

公开数据的许可、下载地址、引用和接入说明见 `datasets.py` 与
`REFERENCES.md`。不同 DFT 泛函、实验温度、频率、电场和聚合物形态必须作为
`environment`、`fidelity` 或 `conditions` 保存，禁止把不相容标签直接拼接。

### 储能、高熵、基团和势垒的可检验路径

模块内置的是“待检验先验”，不是硬编码结论：

- 储能：直接从放电支路计算
  \(U_d=\int E\,dP\)，并分别追踪击穿场、最大/剩余极化和效率。
- 高熵：检验键型或组分熵是否通过势垒分布、随机场、延迟饱和和较低剩余极化产生
  非单调收益；同时控制辐照剂量、结晶度、电场和温度。
- 基团—能带：在同一主链/侧链骨架上做单一基团替换，使用同一 DFT/HSE 或实验
  测 HOMO、LUMO，而不只比较带隙；再控制堆积、共轭和结晶度。
- 能量势垒：对多个局域环境做构象扫描/NEB，比较均值、离散度和空间位置；单一路径
  的最高点不能代表高熵体系的势垒分布。
- 交联压电：检验交联密度的最优窗口，以及“局域构象异质性 → 转动势垒降低/相能差
  缩小 → d33”的介导链，而不是假定交联越多越好。

## 3. 依据已发现性质生成结构

生成仍由原有 PolyLoom 完成语法、端点和化学有效性检查；PolyPrism 提供独立性质
预测和不确定性。本模块只用跨环境稳定的机制项重排，并拒绝超出训练适用域的候选。
2026 年的 POLYT5 可作为近期聚合物条件生成外部基线；这里保留 PolyLoom，是因为它
已与项目的 pSMILES/PSELFIES、PolyPrism、物理描述符和训练工作空间原生一致。

```python
from zynnova.ml.generation.PolyLoom import load_poly_loom
from zynnova.ml.prediction.PolyPrism import load_poly_prism
from zynnova.ml.mlff.poly_discovery import (
    MechanismConstraint,
    MechanismGenerationConfig,
)

generator = load_poly_loom("checkpoints/polyloom.pt")
predictor = load_poly_prism("checkpoints/polyprism.pt")
campaign.generation_config = MechanismGenerationConfig(
    num_candidates=24,
    oversample_factor=10,
    maximum_applicability_distance=4.0,
    constraints=(
        MechanismConstraint(
            "bond_configurational_entropy_R",
            lower=0.7,
            required=False,
        ),
        MechanismConstraint("formal_charge_abs", upper=0.0),
    ),
)
generated = campaign.generate(
    generator,
    {
        "recoverable_energy_density_J_cm3": 30.0,
        "efficiency": 0.85,
    },
    process_conditions={"temperature_K": 298.15},
    predictor=predictor,
)

active_candidates = [
    candidate.to_active_candidate(candidate_id=f"gen-{index:04d}")
    for index, candidate in enumerate(generated.candidates)
]
next_calculations = campaign.propose_simulations(active_candidates)
```

输出分数拆分为生成器目标分、机制贡献、适用域距离、特征覆盖率、性质不确定性和
化学新颖性。每个候选保留各机制项的贡献，因此可以回答“为什么被选中”，也可以定位
是基团、熵、势垒还是过程变量驱动了排名。带隙目标会自动路由到电子结构 oracle；
势垒路由到 NEB；介电目标路由到带偶极的平衡 MD；储能目标路由到多尺度验证。

## 文件职责

| 文件 | 职责 |
|---|---|
| `potential.py` | JouleWeave 高分子预设、训练/加载、委员会 UQ、主动选帧 |
| `simulation.py` | ASE 计算、弛豫、MD、NEB、构象能和偶极涨落 |
| `observables.py` | P–E 储能、效率、介电涨落、MSD、CED 和势垒统计 |
| `features.py` | RDKit 基团、键型/组分熵和依赖缺失时的显式近似回退 |
| `datasets.py` | 公开数据目录及 ZynNova `MaterialSample` 适配 |
| `priors.py` | 文献机制路径、反证试验和适用 oracle |
| `discovery.py` | 稳定选择、跨环境检验、介导、匹配控制和证据分级 |
| `symbolic.py` | 小型、可审计、带交叉验证的经验关系搜索 |
| `physics_learning/dimensions.py` | SI 量纲、变量单位和 Buckingham Π 群 |
| `physics_learning/neural.py` | 稀疏 RBF-KAN 非线性 oracle |
| `physics_learning/interaction.py` | 混合 Hessian 相互作用图与分治 |
| `physics_learning/backends.py` | 原生、PSE、PySR、PhySO、PhyE2E 延迟适配器 |
| `physics_learning/engine.py` | 留环境验证、后端审计、Pareto 排序与统一报告 |
| `physics_learning/dynamics.py` | 积分弱形式稀疏动力学和 PySINDy 适配 |
| `active.py` | 信息增益/不确定性/新颖性/成本联合采样和反事实配对 |
| `generation.py` | PolyLoom + PolyPrism + 机制/适用域重排 |
| `campaign.py` | 数据—发现—模拟—生成的持久化闭环 |
| `reporting.py` | 版本化 JSON 与含反证条件的 Markdown 报告 |

## 最低验收标准

在把一条规律用于论文或新材料筛选前，至少检查：

- 势能在独立高分子家族上的能量、力、应力误差以及委员会外推告警；
- MD 平衡、有限尺寸、时间相关性和不同初态复现；
- NEB 端点原子映射、路径数、DFT 单点校准和势垒分布而非单值；
- P–E 回线的温度、频率、厚度、电极、最大场和击穿删失；
- 基团效应的匹配骨架、同一电子方法、HOMO/LUMO 分解和多重比较；
- 稳定项在至少一个未参与发现的环境中保持方向和量级；
- 生成候选在适用域内，并经过独立预测器、DFT/实验和合成可行性检查。
