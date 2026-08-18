# ZynNova — COMSOL HEX8 修复 + TetGen 自适应四面体网格任务说明

## 1. 总目标

本任务是在独立框架 **ZynNova** 中升级电池微结构/有限元网格能力，重点解决两个问题：

1. 修复 COMSOL MPHTXT 的 HEX8 拓扑错误。
2. 彻底替换“每个规则体素固定切成 6 个 Tet4”的生产网格方案，引入真实 TetGen C++ 内核，形成高质量、非均匀、界面适应的四面体网格。

框架总名称必须保持 **ZynNova / `zynnova`**，不要重新引入旧总包名称。

---

## 2. COMSOL HEX8 必须解决的问题

COMSOL 当前报错：

```text
域单元配置无效。
两个单元连接到共享单元面的同一侧。
请尝试在没有域单元的情况下导入。
详细信息: Coordinates: 2.25e-07, 4.5e-07, 2.25e-07
```

### 已定位的根因

旧实现把 VTK/常见环绕式 Hex8 节点次序直接写入 COMSOL MPHTXT。

旧顺序：

```text
000, 100, 110, 010, 001, 101, 111, 011
```

COMSOL 应使用张量积式顺序：

```text
000, 100, 010, 110, 001, 101, 011, 111
```

因此需要交换旧局部节点中的 3/4、7/8 位置（按一基编号理解）。Quad4 低维面同样必须使用与 COMSOL 一致的局部节点顺序。

### 必须保留/完成的审计

写出 MPHTXT 之前必须检查：

- Hex8 中心 Jacobian 的符号/非退化性；
- 六个局部面是否有效；
- 相邻 Hex8 是否共享完全相同的 4 个节点；
- 两个相邻体单元中心是否处在共享面的相反两侧；
- 是否出现超过 2 个 Hex8 共用同一面的非流形情况；
- 是否存在重复单元；
- 是否存在反向/退化单元；
- 针对 `(2.25e-07, 4.5e-07, 2.25e-07)` 附近的最小回归样例。

### 诊断导出模式

需要支持：

1. 完整 `HEX8 + 边界面 + selections/domain attributes`；
2. 仅写 3D HEX8 域单元、不写二维面元素的诊断模式；
3. 如有必要，可再提供纯表面诊断文件，但不能把“移除域”当作掩盖错误节点顺序的修复方案。

### Python/C++ 必须同时修

必须同步修改：

- Python 流式 MPHTXT Hex8 路径；
- C++ 高速体素连接生成路径；
- 对应 Python/C++ 回归测试。

不能出现 Python 修好了、原生 C++ 快路径仍输出错误顺序的情况。

---

## 3. TetGen 高质量非均匀四面体网格目标

当前规则方案：

```text
voxel -> cube -> 固定 6 个 Tet4
```

只能作为快速/调试后端，不能再作为生产默认。

生产路线应为：

```text
多相体素/标签场
  -> 多材料共形界面提取
  -> 高质量表面网格/PLC
  -> TetGen constrained Delaunay tetrahedralization
  -> 分区域/局部尺寸控制
  -> Tet4 质量审计
  -> COMSOL / VTK / Gmsh / Abaqus 导出
```

### 多材料表面/PLC 关键要求

不能把每个相独立 marching cubes 后直接拼接，否则容易产生双层界面、裂缝和不共享节点。

应保证：

- 每个材料界面只生成一次；
- 界面两侧材料 ID 可追踪；
- 三相/多相交线共享同一组节点；
- 内部界面与外边界 marker 可区分；
- 对对角接触、非流形边/点进行审计；
- 必要时做最小拓扑修复，并记录被重标记的体素数量与比例；
- 生产默认不能静默大规模改变材料体积分数。

### 网格尺寸必须空间可变

必须支持：

- 全局最大 Tet 体积；
- 按材料相设置最大 Tet 体积；
- 按界面设置最大三角面积；
- 局部 refinement zone，例如裂纹尖端、SEI/CEI、颗粒尖角、孔喉；
- TetGen radius-edge ratio；
- 最小二面角；
- 优化等级；
- Steiner point 控制；
- 质量门禁。

例如：

```text
SEI / CEI / crack tip -> 细网格
活性材料内部       -> 较粗
大体积电解液       -> 中等/较粗
```

最终网格必须是真实不规则 TetGen 风格网格，而不是规则立方体切分。

---

## 4. TetGen C++ 集成要求

用户要求把真实 TetGen C++ 内核纳入源码，并在 `pip install` 时编译为 Python 可调用扩展。

当前设计：

```text
TetGen C++ source
   -> CMake target
   -> pybind11 module
   -> zynnova._native._zynmorph_tetgen_native
   -> zynnova.zynmorph Python API
```

Python 层接口目标：

```python
from zynnova.zynmorph import TetGenMeshingConfig, LocalRefinementZone, mesh_microstructure

fem = mesh_microstructure(
    volume,
    method="tetgen",
    tetgen_config=config,
)
```

`method="structured"` 只保留为明确指定的快速后端，不能在 TetGen 缺失/失败时静默回退。

### TetGen vendoring

当前补丁中包含：

```text
cpp/third_party/tetgen/SOURCE_LOCK.json
scripts/vendor_tetgen.py
```

锁定真实上游版本/提交并复制：

```text
tetgen.cxx
tetgen.h
predicates.cxx
tetgen-license
```

进入：

```text
cpp/third_party/tetgen/source/
```

然后通过 CMake/pybind11 编译。

**重要：当前这个打包文件没有物理包含 TetGen 上游源码本身。**
上一轮运行环境没有把 GitHub 中的源码文件下载进工作目录，所以这里保留的是完整绑定、CMake、版本锁、vendoring 脚本与许可证说明。联网环境需先执行：

```bash
python scripts/vendor_tetgen.py --accept-agpl
```

再构建。

### 许可证边界

TetGen 1.6.x 的公开源码许可证必须单独遵守；不要把嵌入/链接后的 TetGen 二进制错误地宣称为纯 MIT。相关说明在：

```text
THIRD_PARTY_NOTICES.md
cpp/third_party/tetgen/SOURCE_LOCK.json
```

---

## 5. 本包中已经包含的文件

### COMSOL/Hex8

```text
src/zynnova/zynsim/io/comsol_general.py
cpp/src/zynsim/voxel.cpp
cpp/tests/test_zynsim_voxel.cpp
tests/zynnova/test_comsol_hex_orientation.py
tests/zynnova/test_zynmorph_comsol.py
```

### 自适应 TetGen 网格

```text
src/zynnova/zynmorph/surface.py
src/zynnova/zynmorph/tetgen.py
src/zynnova/zynmorph/meshing.py
src/zynnova/zynmorph/schema.py
src/zynnova/zynmorph/pipeline.py
src/zynnova/zynmorph/__init__.py
```

### C++/构建

```text
cpp/bindings/zynmorph_tetgen_module.cpp
cpp/CMakeLists.txt
CMakeLists.txt
pyproject.toml
scripts/vendor_tetgen.py
cpp/third_party/tetgen/SOURCE_LOCK.json
THIRD_PARTY_NOTICES.md
```

### 测试

```text
tests/zynnova/test_zynmorph_tetgen_surface.py
tests/zynnova/test_zynmorph_tetgen_build_contract.py
```

---

## 6. 上一轮已做的验证

上一轮工作区中报告的定向验证包括：

```text
ZynNova 定向 Python 测试：35 passed
C++ voxel/Hex8 原生测试：passed
Python compileall：passed
```

这些验证主要覆盖：

- COMSOL Hex8 局部节点顺序；
- 旧错误节点顺序必须被拒绝；
- 共享面两侧性；
- 纳米尺度/非等距体素；
- Python 与 C++ Hex8 路径一致性；
- 多材料 PLC 提取与拓扑审计；
- TetGen Python/C++ 构建契约。

但是必须明确：

1. 当时环境不能启动 COMSOL GUI，所以没有声称真实 COMSOL GUI 导入已经执行通过；
2. 当时没有把 TetGen 上游源码下载进本地，因此没有声称真正的 TetGen 1.6 原生扩展已经在该环境中完成编译和运行。

下一步在联网且有编译器的真实 ZynNova 工作环境中，应完成下面第 7 节的最终验收。

---

## 7. 你接下来应该做什么

### A. 将本补丁覆盖到当前 ZynNova 仓库

保持目录结构复制文件即可。

### B. 下载锁定的 TetGen 源码

在仓库根目录：

```bash
python scripts/vendor_tetgen.py --accept-agpl
```

确认出现：

```text
cpp/third_party/tetgen/source/tetgen.cxx
cpp/third_party/tetgen/source/tetgen.h
cpp/third_party/tetgen/source/predicates.cxx
cpp/third_party/tetgen/source/tetgen-license
```

### C. 重新 pip 编译 ZynNova

建议先卸载旧 editable build/清理缓存：

```bash
python -m pip uninstall -y zynnova
rm -rf build _skbuild .cache
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
```

然后：

```bash
python -m pip install -e ".[zynmorph-tetgen]"
```

如果项目 extras 名称不同，以当前 `pyproject.toml` 为准。

### D. 验证原生扩展可导入

```bash
python - <<'PY'
from zynnova._native import _zynmorph_tetgen_native
print(_zynmorph_tetgen_native)
PY
```

### E. 跑专项测试

```bash
PYTHONPATH=src pytest -q \
  tests/zynnova/test_comsol_hex_orientation.py \
  tests/zynnova/test_zynmorph_comsol.py \
  tests/zynnova/test_zynmorph_tetgen_surface.py \
  tests/zynnova/test_zynmorph_tetgen_build_contract.py
```

若 C++ tests 已接入 CTest，同时运行：

```bash
ctest --test-dir build --output-on-failure
```

### F. 实际 COMSOL 验收

首先导入最小 2-cell/小体素 HEX8 文件，确认不再出现：

```text
两个单元连接到共享单元面的同一侧
```

然后再导入复杂多相电池 RVE。

至少检查：

- Domain 数量；
- 每个材料 Domain 的实体编号；
- 内部界面；
- 外边界；
- selections；
- 是否存在重叠域；
- 是否存在 inverted/invalid elements。

### G. 真正运行 TetGen 后端

验证输出不是：

```text
n_tets == n_voxels * 6
```

并统计：

- Tet 体积分布；
- edge length 分布；
- radius-edge ratio；
- 最小二面角；
- 不同区域平均尺寸；
- 裂纹/SEI/CEI 附近是否明显细化；
- inverted = 0；
- degenerate = 0；
- 材料域映射正确。

### H. 最终复杂 Notebook

仍需完成一个最终生产 notebook，建议命名：

```text
notebooks/ZynNova_TetGen_Complex_Battery_RVE.ipynb
```

其内容应至少包括：

- 100+ 不规则正/负极颗粒；
- 电解液孔隙网络；
- CBD；
- SEI/CEI；
- 裂纹；
- 集流体；
- 多相共形表面；
- 表面质量审计；
- TetGen 区域/局部尺寸场；
- 非均匀 Tet4；
- COMSOL MPHTXT；
- VTK/Gmsh/Abaqus；
- 网格质量图；
- 相域/界面统计。

---

## 8. 最终验收标准

只有同时满足以下条件才算任务真正完成：

- [ ] COMSOL Hex8 最小回归案例实际导入成功；
- [ ] 报错坐标附近不再出现共享面同侧错误；
- [ ] Python/C++ Hex8 连接顺序一致；
- [ ] TetGen 上游源码真实存在于 vendored source 目录；
- [ ] `pip install` 能编译 `_zynmorph_tetgen_native`；
- [ ] Python 可以直接调用 C++ TetGen 内核；
- [ ] 复杂多相 PLC 无裂缝、重复面、错误非流形连接；
- [ ] TetGen 输出非规则、非均匀 Tet4；
- [ ] 不同材料/局部区域尺寸控制生效；
- [ ] inverted Tet = 0；
- [ ] degenerate Tet = 0；
- [ ] COMSOL 多材料 domain 与 interface 映射正确；
- [ ] 最终复杂电池 RVE notebook 可以完整执行。
