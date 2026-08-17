# ZynSim 大规模跨尺度升级说明

本次升级针对原有 ZynSim 中“只有接口、采用紧凑代理、缺少超大规模路径或尚未形成闭环”的部分，新增了图像驱动微结构、通用混合有限元网格、流式 COMSOL 导出、孔尺度电化学、三维正极化学–力学–断裂、多速率全尺度耦合、单体 Newton–Krylov、分布式线性代数和更完整的材料本构。

> **工程边界**：这些实现提供了可运行的研究级/工程级数值内核、严格的状态检查和可扩展接口，但“工业级认证”仍需要针对目标电芯和材料完成参数标定、网格收敛、实验验证、软件 V&V、长期稳定性测试和目标 COMSOL 版本的导入验收。默认 NMC 参数是文献量级起点，不代表任何商业粉体的已标定参数。

## 1. 新增模块总览

| 路径 | 作用 |
|---|---|
| `zynsim/microstructure/imaging.py` | 单张图像阈值/颜色/自动多 Otsu 分相和相占比 |
| `zynsim/microstructure/reconstruction.py` | 单视图或 XY/XZ/YZ 三视图快速三维体素重建 |
| `zynsim/microstructure/morphology.py` | ZynMorph 形貌描述符和相比例精确重定向 |
| `zynsim/microstructure/pipeline.py` | 图像→体素→形貌→网格→COMSOL 一键工作流 |
| `zynsim/core/general_mesh.py` | Tet4、Hex8、Wedge6、Pyramid5、Tri3、Quad4 混合网格容器 |
| `zynsim/io/comsol_general.py` | 通用网格和超大体素网格的 COMSOL MPHTXT 导出 |
| `zynsim/io/mesh_exchange.py` | meshio 驱动的 VTU/XDMF/Exodus/Gmsh/Abaqus 等交换 |
| `zynsim/fem/distributed.py` | PETSc KSP/GAMG/Hypre 和 Morton 分区 |
| `zynsim/multiphysics/industrial.py` | 广义 Maxwell、J2 塑性、混合模态 cohesive/contact、热失控反应网络 |
| `zynsim/multiphysics/monolithic.py` | 矩阵自由单体 Newton–Krylov 求解器 |
| `zynsim/battery/pore_scale.py` | 显式体素孔尺度固相/电解液传输与 Butler–Volmer 反应 |
| `zynsim/battery/mechanics/*` | 三维 NMC 化学–力学–相场断裂和 RVE→P2D 闭环 |
| `zynsim/workflows/fast_full_scale.py` | 多速率全尺度推进、原子属性刷新、RVE 力学、数字孪生和无 pickle 断点 |
| `cpp/src/zynsim/voxel.cpp` | OpenMP 图像阈值、正交投影融合、界面计数和连接生成 |

## 2. 图像到体素

### 2.1 单图分相

显式阈值规则为

\[
L(\mathbf{x})=\ell_k,\qquad I_k^{\min}\le I(\mathbf{x})\le I_k^{\max}.
\]

RGB 图像还可以按颜色距离分类。未设置阈值时，使用确定性的多 Otsu 动态规划，最大化类间方差。输出包含：

- 整数相标签；
- 每相像素数；
- 每相面积占比；
- 实际阈值；
- 原生 C++ 后端使用情况。

### 2.2 单视图三维重建

单视图采用投影保持的相关挤出。二维标签作为硬投影约束，沿缺失方向生成具有指定相关长度的随机场，并只在界面附近进行扰动，因此不会退化为简单逐层复制。

### 2.3 三正交视图重建

对于 `xy[x,y]`、`xz[x,z]` 和 `yz[y,z]`，每个体素相标签通过投影一致性势选择：

\[
\ell^*(x,y,z)=\arg\max_{\ell}
\left[
 w_p\bigl(
 \mathbf{1}_{L_{xy}=\ell}+\mathbf{1}_{L_{xz}=\ell}+\mathbf{1}_{L_{yz}=\ell}
 \bigr)+w_m\log \pi_\ell+\xi_\ell
\right].
\]

其中 \(\pi_\ell\) 是三视图平均先验，\(\xi_\ell\) 是相关形貌扰动。随后执行投影松弛，降低三方向投影误差。大体素阵列可以写入 `numpy.memmap`，避免完整复制。

三张图只约束三个投影，并不能唯一决定真实三维结构。因此本算法输出的是满足投影与统计形貌约束的重建样本，而不是对唯一真实三维结构的数学恢复。使用 XCT/FIB-SEM 三维标签时应直接输入真实标签。

## 3. ZynMorph：可编辑的形貌参数表示

ZynMorph 是固定长度的谱–拓扑–几何描述符。对每个相 \(\alpha\)，包含：

\[
\mathbf{z}_\alpha=
[\phi_\alpha,
S_{v,\alpha},
\bar\ell_x,\bar\ell_y,\bar\ell_z,
\sigma_{\ell,x},\sigma_{\ell,y},\sigma_{\ell,z},
\ell_{90,x},\ell_{90,y},\ell_{90,z},
\bar k,\sigma_k,P_{\mathrm{low}},
N_c/V,
\Pi_x,\Pi_y,\Pi_z,
\lambda_1,\lambda_2,\lambda_3,
\mathbf{n}_1,
\mathbf{p}_{\mathrm{radial}}].
\]

其中：

- \(\phi\)：相体积分数；
- \(S_v\)：单位体积界面面积；
- \(\ell\)：三方向 chord-length 统计；
- \(\bar k,\sigma_k\)：功率谱质心和带宽；
- \(P_{\mathrm{low}}\)：低频结构能量；
- \(N_c/V\)：连通分量密度；
- \(\Pi_i\)：三方向贯通性；
- \(\lambda_i,\mathbf n_1\)：结构张量各向异性；
- \(\mathbf p_{\mathrm{radial}}\)：归一化径向功率谱。

最终指纹由描述向量和网格尺寸的 SHA-256 截断值生成，可用于缓存、数据版本和生成模型条件。

`retarget_phase_fractions()` 通过各相 signed-distance 分数的对偶偏置移动界面，再使用确定性的整数配额修复，使目标体积分数达到最近的一个体素精度，同时尽量保持原始界面和连通形貌。可用 `fixed_mask` 锁定集流体、隔膜或实验可信区域。

## 4. 通用与超大规模有限元网格

### 4.1 通用网格对象

`GeneralMesh` 支持一阶：

- 体单元：Tet4、Hex8、Wedge6、Pyramid5；
- 边界单元：Tri3、Quad4；
- 辅助单元：Line2、Point1。

每个 `ElementBlock` 可独立保存区域实体 ID、名称和元数据。Hex8/Wedge6/Pyramid5 可以一致地分解为 Tet4，以复用原来的 Tet4 求解器。`quality_report()` 给出有符号 Jacobian、倒置/退化子单元、总体积和 scaled-Jacobian 统计；`validate_positive_volume()` 不再用绝对值掩盖倒置。

### 4.2 体素网格

体素到 Hex8 是严格共形映射，每个体素对应一个 Hex8。Tet4 模式使用每个体素六个一致四面体。区域标签保留为 COMSOL geometric entity，外边界和材料界面可生成命名选择。Hex8 使用 Quad4 边界；Tet4 自动把每个体素面拆为两个共形 Tri3，避免四边形边界与四面体体单元不匹配。

### 4.3 Out-of-core 流式导出

`write_large_voxel_comsol_mphtxt()` 不创建完整的节点或连接矩阵。它按块依次写出：

1. 节点坐标；
2. Hex8/Tet4 连接；
3. 区域实体；
4. 六个外边界；
5. 不同相之间的内部界面；
6. COMSOL selection 对象。

额外内存复杂度约为

\[
M_{\mathrm{extra}}=O(N_{\mathrm{chunk}}),
\]

而不是 \(O(N_{\mathrm{nodes}}+N_{\mathrm{elements}})\)。因此可处理远大于内存容量的网格；实际极限由体素标签存储、磁盘空间、文件系统和 COMSOL 自身导入能力决定。

对规则体素：

\[
N_{\mathrm{node}}=(n_x+1)(n_y+1)(n_z+1),
\]

\[
N_{\mathrm{Hex8}}=n_xn_yn_z,\qquad
N_{\mathrm{Tet4}}=6n_xn_yn_z.
\]

在真正生成前应调用 `plan_large_voxel_mesh()` 获取节点数、单元数、界面数和预计输出规模。

### 4.4 COMSOL 的含义

导出结果是 COMSOL MPHTXT 网格及域/边界选择，可在 COMSOL 中导入。它不自动生成 `.mph` 中的材料、物理场、study 和 solver；这些仍应通过 COMSOL GUI、Java API 或 LiveLink 配置。

## 5. 孔尺度电化学

`PoreScaleElectrochemistry` 在真实体素界面上建立有限体积问题。

固相和电解液电势满足

\[
-\nabla\cdot(\sigma_s\nabla\phi_s)=a_{se}j,
\qquad
-\nabla\cdot(\kappa_e\nabla\phi_e)=-a_{se}j.
\]

界面反应采用对称 Butler–Volmer：

\[
j=2i_0\sinh\left(\frac{\alpha F\eta}{RT}\right),
\quad
\eta=\phi_s-\phi_e-U(\theta,T),
\]

\[
i_0=i_{0,\mathrm{ref}}
\sqrt{\frac{c_e}{c_{e,0}}\theta(1-\theta)}.
\]

固相占位和电解液浓度使用隐式扩散推进，并在每个相界面面上守恒地交换锂。求解器会拒绝不接触参考边界的悬浮固相/电解液连通分量，避免用数值正则化掩盖奇异电势问题。

## 6. 三维 Ni-rich 正极化学–力学–退化

### 6.1 总自由能

RVE 中的实现对应以下自由能结构：

\[
\Psi=\int_\Omega
\left[
 f_{\mathrm{mix}}(\theta,T)
 +\frac{\kappa}{2C_{\max}}|\nabla\theta|^2
 +g(d)\psi_e^+(\boldsymbol\varepsilon-\boldsymbol\varepsilon^c)
 +\psi_e^-
 +G_c(\theta,\alpha_f,g_b)
 \left(\frac{d^2}{2\ell}+\frac{\ell}{2}|\nabla d|^2\right)
\right]dV.
\]

其中 \(\theta=c/C_{\max}\)，\(d\in[0,1]\) 是损伤，\(g(d)=(1-d)^2+k_r\)。

正则溶液化学势为

\[
\mu_{\mathrm{chem}}=RT\ln\frac{\theta}{1-\theta}
+\Omega(1-2\theta).
\]

加入梯度和应力耦合后：

\[
\mu=\mu_{\mathrm{chem}}
-\frac{\kappa}{C_{\max}}\nabla^2\theta
-\frac{1}{C_{\max}}
\boldsymbol\sigma:\frac{\partial\boldsymbol\varepsilon^c}{\partial\theta}.
\]

### 6.2 各向异性扩散

层状 NMC 的扩散张量写为

\[
\mathbf D=D_{ab}(\theta,T)(\mathbf I-\mathbf n_c\otimes\mathbf n_c)
+D_c(\theta,T)\mathbf n_c\otimes\mathbf n_c,
\]

\[
D_c=r_DD_{ab},\qquad r_D\ll1,
\]

\[
\mathbf j=-\frac{\theta(1-\theta)}{RT}\mathbf D\nabla\mu,
\qquad
\dot\theta=-\nabla\cdot\mathbf j+s_R.
\]

默认状态函数包含高占位区超过两个数量级的扩散率下降和 Arrhenius 温度修正。可通过 `property_provider` 接入 JouleWeave、ZynForge、DFT 或实验表面。

### 6.3 晶格本征应变和横观各向同性弹性

以晶体 c 轴 \(\mathbf n_c\) 为方向：

\[
\boldsymbol\varepsilon^c=
\varepsilon_{ab}(\theta)(\mathbf I-\mathbf n_c\otimes\mathbf n_c)
+\varepsilon_c(\theta)\mathbf n_c\otimes\mathbf n_c.
\]

\(\varepsilon_c(\theta)\) 使用非单调插值表示高 SOC 附近的 c 轴变化。弹性使用正定横观各向同性能量，不需要逐体素存储完整 6×6 刚度矩阵。谱迭代通过 FFT 求解周期 RVE 的近似平衡

\[
\nabla\cdot\boldsymbol\sigma=\mathbf0.
\]

### 6.4 相场断裂、疲劳和晶界

历史场

\[
\mathcal H(t)=\max_{\tau\le t}\psi_e^+(\tau)
\]

保证裂纹不可逆。黏性 AT2 型演化为

\[
\eta_d\dot d=
2(1-d)\mathcal H
-G_c\left(\frac d\ell-\ell\nabla^2d\right),
\quad \dot d\ge0.
\]

有效断裂能同时受晶界、首次脱锂和循环疲劳控制：

\[
G_c^{\mathrm{eff}}=G_{c0}
(1-r_{gb}g_b)
[1-r_1(1-\theta_{\min})]
\max\left[
\frac{1}{1+(\alpha_f/\alpha_0)^m},r_{\min}
\right].
\]

这使晶界优先开裂、首次脱锂强度下降和后续疲劳累积能够同时进入模型。

### 6.5 裂纹润湿

裂纹和外表面建立湿润场 \(w\)：

\[
\dot w=\frac{w^*(|\nabla d|,S_v)-w}{\tau_w}+D_w\nabla^2w.
\]

反应源权重变为

\[
s_R\propto S_v+\chi_w w|\nabla d|,
\]

从而允许新生裂纹内部产生局部反应通量，而不是继续假定整个颗粒表面均匀通量。

### 6.6 塑性剪切和氧缺陷代理

超过临界剪应力时累计不可逆剪切：

\[
\dot\gamma_p=\frac{1}{\tau_p}
\left\langle\frac{\tau_{\max}}{\tau_c}-1\right\rangle^{m_p}.
\]

氧缺陷指标使用光滑阈值

\[
f_{O\!-\mathrm{def}}=\frac12\left[1+\tanh\left(\frac{\gamma_p-\gamma_c}{\Delta\gamma}\right)\right].
\]

它是用于跨尺度传输反馈的机制代理，不是显式氧空位扩散/表面重构反应网络。

### 6.7 高电压相失配、氧迁移与裂纹俘获

为描述深度脱锂时层状正极的 O3→O1/阳离子混排类转变，增加可标定的
高电压转变变量 \(\xi\)：

\[
\xi^*(\theta,T)=\frac12\left[
1+\tanh\left(\frac{\theta_c-\theta}{w_\theta}\right)
\right],
\]

\[
\tau_\xi\dot\xi=\xi^*-\xi,
\qquad \dot\xi\ge0.
\]

转变产生额外晶格失配本征应变：

\[
\boldsymbol\varepsilon^{\mathrm{tr}}=
\xi\left[
\varepsilon_{ab}^{\mathrm{tr}}
(\varepsilon_c^{\mathrm{tr}}-\varepsilon_{ab}^{\mathrm{tr}})
\mathbf n_c\otimes\mathbf n_c
\right].
\]

移动氧状态 \(c_O\) 和裂纹俘获氧状态 \(c_O^{\mathrm{trap}}\) 采用守恒的
扩散–生成–俘获闭合：

\[
\frac{\partial c_O}{\partial t}
=D_O\nabla^2c_O+\gamma_O\dot\xi
-k_t\left(1+\chi_t\mathcal S_d\right)c_O,
\]

\[
\frac{\partial c_O^{\mathrm{trap}}}{\partial t}
=k_t\left(1+\chi_t\mathcal S_d\right)c_O,
\qquad
\mathcal S_d\propto |\nabla d|+w\bigl(|\nabla d|+S_v\bigr).
\]

氧暴露进一步降低断裂韧度与固相传输：

\[
G_c^{\mathrm{eff}}\leftarrow
G_c^{\mathrm{eff}}(1-r_Oc_O^{\mathrm{trap}}),
\qquad
D_s^{\mathrm{eff}}\leftarrow
D_s^{\mathrm{eff}}(1-r_Dc_O^{\mathrm{trap}}).
\]

数值上，\(\xi\) 使用有界半隐式松弛；氧扩散在 Fourier 空间隐式推进，
空间变化的裂纹俘获再用解析指数衰减更新，因此在较长的 RVE 多速率步长下
仍不会受到显式扩散稳定性上限控制。该实现是层状氧化物的**可标定机制代理**，
用于表达相失配–氧迁移–裂纹协同，不宣称解析 O–O 成键、气泡成核或完整表面
重构反应网络。

## 7. 颗粒/RVE 到 P2D 闭环

多个 RVE 可并行推进并按代表权重均匀化。输出：

- 损伤体积分数；
- 裂纹面密度；
- 连通活性材料比例；
- 最大主应力；
- 氧缺陷比例；
- 高电压转变相比例；
- 移动氧和裂纹俘获氧比例；
- 堆叠压力局部应力放大；
- 容量、扩散、反应面积、电子电导、孔隙率和电解液传输倍率；
- 高压力析锂风险倍率。

闭环映射示例：

\[
\varepsilon_{s,+}^{\mathrm{eff}}=
\varepsilon_{s,+}m_{\mathrm{active}},
\]

\[
D_{s,+}^{\mathrm{eff}}=D_{s,+}m_D,
\quad
k_+^{\mathrm{eff}}=k_+m_R,
\quad
\sigma_+^{\mathrm{eff}}=\sigma_+m_\sigma.
\]

基线参数只缓存一次，因此多次更新不会产生重复乘法漂移。

堆叠压力使用可校准的力链闭合：配位数随压力饱和增长，低配位导致局部应力放大；高压力同时降低孔隙传输并增加析锂风险。默认最优压力起点为 12.5 bar，仅用于复现文献趋势，必须针对实际电芯重新标定。

## 8. 快速全尺度计算

`FastFullScaleWorkflow` 使用多速率调度：

- 电化学/热：短时间步；
- 原子/材料属性刷新：较长时间间隔或不确定度触发；
- 三维 RVE 力学：较长时间间隔；
- 数字孪生：独立间隔；
- 保存和输出：独立间隔。

这种方式避免在每个电化学小步都运行昂贵的三维 RVE，同时保持最新的微结构闭合反馈。支持原子写入的断点目录：

- `manifest.json`；
- `checksums.json`；
- `coupled_state.npz`；
- 每个 RVE 的独立 NPZ；
- 用户定义的电化学状态文件。

不使用 pickle，并在恢复时验证 SHA-256。

## 9. 单体和分布式求解

`MonolithicNewtonKrylovSolver` 对任意拼接后的残差 \(\mathbf R(\mathbf x)=0\) 使用矩阵自由有限差分 Jacobian-vector product：

\[
\mathbf J(\mathbf x)\mathbf v\approx
\frac{\mathbf R(\mathbf x+h\mathbf v)-\mathbf R(\mathbf x)}{h}.
\]

线性子问题使用 GMRES，外层使用非精确 Newton 和 Armijo 回溯。用户可以提供解析 Jv 和预条件器。

`PETScLinearSolver` 支持 CG、GMRES、BCGS、MINRES 与 GAMG/Hypre/Jacobi/ILU。`morton_partition()` 提供确定性空间局部性分区。PETSc/MPI 为可选依赖；未安装时不会影响单机 SciPy 路径。

## 10. 最新正极力学退化研究对应关系

实现依据公开论文支持的物理机制重新组织，但没有复制论文专有代码，也没有把论文中针对特定体系的参数当作通用常数：

1. **Zhao et al., Nature Communications, 2026-07-10**, DOI `10.1038/s41467-026-75373-2`：在高电压 LiCoO₂ 中识别出 O3→O1→阳离子混排转变产生的晶格失配应力，以及氧化晶格氧向内部裂纹迁移并在裂纹处积累的协同裂纹扩展机制。本升级据此增加 \(\xi\)、移动氧和裂纹俘获氧三个独立状态；由于论文体系和工况特定，默认参数只作为可运行起点。
2. **Wang et al., Nature Energy, 2026-06-29**, DOI `10.1038/s41560-026-02087-6`：支持堆叠压力存在双侧风险窗口——低压力加速正极开裂，高压力促进负极析锂。论文中约 `12.5 bar` 的最优值只适用于其测试的石墨||NMC811 电芯，因此代码把压力窗口设为可标定参数。
3. **Yin et al., Advanced Energy Materials, 2026-05-27**, DOI `10.1002/aenm.71110`：支持超高镍材料中由晶内应变积累与晶间副反应共同驱动的跨尺度晶内/晶间开裂。对应实现同时保留晶体取向、晶界弱化、疲劳和界面氧/润湿状态。
4. **Eum et al., Nature Energy, 2026-03-02**, DOI `10.1038/s41560-026-01988-w`：支持 Ni-rich 正极的化学–力学失效与孔结构非均匀性密切相关，均匀孔结构可通过耗散应变能减轻失效。ZynMorph、显式孔结构和 RVE 力学因此共享孔隙、界面和连通性描述，而不是只使用平均孔隙率。
5. **Chen et al., Advanced Functional Materials 2026（2025 年在线发表）**, DOI `10.1002/adfm.202517282`：支持裂纹表面电解液润湿、裂纹表面新增反应通量、润湿与裂纹传播模式/距离/方向之间的双向耦合。本升级据此加入独立湿润状态，而不是把裂纹只作为电导率折减系数。
6. **Liu, Roters & Raabe, Nature Communications 2024**, DOI `10.1038/s41467-024-52123-w`：支持晶粒级三维 RVE、各向异性且浓度相关的 Li 扩散、晶格尺寸变化、位错/塑性剪切、氧缺陷代理、接触损失以及 PETSc 大规模计算。该论文明确说明其模型没有显式机械断裂，因此本升级中的相场裂纹来自后续断裂研究，而不是伪称为该论文原实现。

文献提供的是机制与特定实验体系的证据。默认参数仅用于数值启动；NMC 化学计量、颗粒类型、粘结剂/电解质、孔结构、温度、截止电压和堆叠方式变化后，必须重新标定。

## 11. 推荐入口

```python
from zynnova.zynsim.microstructure import (
    ImageSegmentationConfig,
    ImageToVoxelConfig,
    ImageElectrodeFEMConfig,
    OrthogonalImages,
    PhaseThreshold,
    image_to_electrode_fem,
)
```

```python
from zynnova.zynsim.battery import (
    NMCCathodeMaterial,
    CathodeSpectralConfig,
    SpectralCathodeDegradationSolver,
    CathodeScaleConfig,
    CathodeMechanicalMultiscaleModel,
)
```

## 12. 验证要求

投产或论文定量结论前至少执行：

1. 图像分相人工标注对照和相占比误差；
2. 三维重建的两点相关、chord、连通性、界面面积和独立切片验证；
3. Tet/Hex 网格正体积、Jacobian、界面一致性和网格收敛；
4. COMSOL 导入、域选择和边界选择核验；
5. 扩散、弹性和断裂的解析/制造解；
6. NMC 晶格应变、扩散率、模量和断裂能的材料专属标定；
7. 裂纹长度/体积与 operando XCT/SEM 对照；
8. 容量衰减、EIS、温升和堆叠压力窗口的电芯级验证；
9. MPI/PETSc 强弱扩展测试；
10. 长循环断点恢复的一致性测试。
