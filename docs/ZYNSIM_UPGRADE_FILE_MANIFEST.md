# ZynSim 大规模升级文件清单

本清单由原始上传工程与升级工程逐文件比较生成。构建目录、缓存、Python 字节码和示例输出未计入。

## 统计

- 新增功能/文档/测试文件：23
- 本清单文件：1
- 修改文件：9
- 删除文件：0
- 增量交付文件总数：33

## 新增文件

- `cpp/bindings/zynsim_voxel_module.cpp`
- `cpp/include/zynnova/zynsim/voxel.hpp`
- `cpp/src/zynsim/voxel.cpp`
- `cpp/tests/test_zynsim_voxel.cpp`
- `docs/ZYNSIM_INDUSTRIAL_UPGRADE.md`
- `examples/zynsim_image_to_comsol_and_cathode_degradation.py`
- `src/zynnova/zynsim/battery/mechanics/__init__.py`
- `src/zynnova/zynsim/battery/mechanics/materials.py`
- `src/zynnova/zynsim/battery/mechanics/multiscale.py`
- `src/zynnova/zynsim/battery/mechanics/spectral.py`
- `src/zynnova/zynsim/battery/pore_scale.py`
- `src/zynnova/zynsim/core/general_mesh.py`
- `src/zynnova/zynsim/fem/distributed.py`
- `src/zynnova/zynsim/io/comsol_general.py`
- `src/zynnova/zynsim/io/mesh_exchange.py`
- `src/zynnova/zynsim/microstructure/imaging.py`
- `src/zynnova/zynsim/microstructure/morphology.py`
- `src/zynnova/zynsim/microstructure/pipeline.py`
- `src/zynnova/zynsim/microstructure/reconstruction.py`
- `src/zynnova/zynsim/multiphysics/industrial.py`
- `src/zynnova/zynsim/multiphysics/monolithic.py`
- `src/zynnova/zynsim/workflows/fast_full_scale.py`
- `tests/test_zynsim_industrial_upgrade.py`

## 修改文件

- `cpp/CMakeLists.txt`
- `pyproject.toml`
- `src/zynnova/zynsim/battery/__init__.py`
- `src/zynnova/zynsim/core/__init__.py`
- `src/zynnova/zynsim/fem/__init__.py`
- `src/zynnova/zynsim/io/__init__.py`
- `src/zynnova/zynsim/microstructure/__init__.py`
- `src/zynnova/zynsim/multiphysics/__init__.py`
- `src/zynnova/zynsim/workflows/__init__.py`

## 删除文件

- 无

## 已执行验证

- Python：`PYTHONPATH=src python -m pytest -q tests/test_zynsim_industrial_upgrade.py`，10 项测试全部通过。
- Python：`python -m compileall -q src/zynnova/zynsim examples tests/test_zynsim_industrial_upgrade.py` 通过。
- 端到端示例：图像 → 体素 → ZynMorph → 共形 Hex/Tet 网格 → COMSOL MPHTXT → 三维正极退化 → RVE 闭合，运行通过。
- COMSOL 文本检查器成功读取流式 Tet4/Tri3 文件、几何实体和命名选择。
- C++：OpenMP 体素内核与 `cpp/tests/test_zynsim_voxel.cpp` 使用 `g++ -std=c++17 -O3 -fopenmp -Wall -Wextra -Wpedantic` 编译并运行通过。
- 2,097,152 体素 memmap 检查：分块网格规划和精确相比例重定向运行通过；该结果仅是当前容器的功能/内存路径检查，不是集群扩展基准。
- 完整 pybind11 CMake 扩展未在当前容器构建：运行环境没有安装 `pybind11`。`pyproject.toml` 已声明构建依赖，独立 C++ 核心测试已通过。

## 适用边界

本版本提供可运行的研究级/工程级大规模数值内核与集成路径，但不等同于已经针对某一商业电芯完成工业认证。投入定量研究或生产前仍需材料参数识别、网格/时间步收敛、实验校准、目标 COMSOL 版本导入验收、MPI/PETSc 集群扩展测试和正式软件 V&V。
