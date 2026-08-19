# ZynNova

<a id="zynnova-cn"></a>

<p align="center"><strong>ZynNova — materials intelligence & scientific workflow 框架</strong></p>

<p align="center">
  <img src="docs/assets/zynnova-overview.png" alt="ZynNova framework overview" width="900">
</p>

[English](#zynnova-en) | [中文](#zynnova-cn)

## 中文简介

ZynNova 是一个面向材料科学和多模态科学计算的工程化框架，采用 `zynnova` 单一命名空间组织。当前版本聚焦于可复用的结构化模块、流程化的数据与模拟接口，以及用于模型推理与实验脚本的统一入口，便于在同一工程中进行仿真、建模和扩展开发。

## 从零安装

```bash
# 1) 安装前置工具
python --version          # 建议 Python 3.10+
git --version
cmake --version

# 2) 拉取源码
git clone <your-repo-url>
cd ZynNova

# 3) 创建并激活虚拟环境
python -m venv .venv

# Windows（PowerShell）
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

# 4) 安装运行与构建依赖
python -m pip install -U pip setuptools wheel
python -m pip install -e .

# 5) 可选：安装开发依赖（测试、Lint、实验工具）
python -m pip install -e ".[dev]"
```

对于轻量级运行，也可将第 4 步替换为：

```bash
python -m pip install -e ".[zynnova]"
```

## 目录结构

```text
src/
 zynnova/
  core/           # 后端注册、通用序列化与公用基础设施
  structure/      # 分子结构与图表示模型
  data/           # 数据源、加载、变换与校验
  dynamics/       # 分子动力学与轨迹相关工具
  ml/
    generation/   # 生成式模型与构建工具
    mlff/         # 力场/势函数相关工作流
    prediction/   # 预测类任务与结构/性能推断
  zynsim/        # FEM、多尺度与逆问题相关流程
  zynmorph/      # 电池微结构生成与 Tet4 网格化
  zynvista/      # 图像/视频驱动的场景重建与渲染
  zynform/       # 对象生成、重建与网格化
  zynvox/        # 语音相关工具链
  geometry/      # 几何基础设施
  _native/       # C++ 侧扩展
  README.md
```

## 核心能力

- **核心工具**
  - `zynnova.core` 提供后端注册、运行时约定与工具链统一入口
  - 序列化、配置和可追溯性相关基础能力
- **结构与数据**
  - `zynnova.structure`、`zynnova.data`、`zynnova.geometry`
  - 数据清洗、过滤、增强与元数据跟踪
- **仿真能力**
  - `zynnova.dynamics`：积分器与轨迹工具链
  - `zynnova.zynsim`：多尺度仿真与逆向工作流
- **子系统能力**
  - `zynnova.zynmorph`：条件化微结构合成与网格生成
  - `zynnova.zynvista`：场景重建、可视化与网格/场景导出
  - `zynnova.zynform`：几何对象处理与修复重建
  - `zynnova.zynvox`：语音相关实验与模型服务能力
- **机器学习**
  - `zynnova.ml` 的 `generation`、`mlff`、`prediction` 命名空间

## 安装与校验

```bash
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m zynnova status
PYTHONPATH=src python scripts/zynnova/verify.py
python scripts/zynnova/static_audit.py
```

## 快速示例

```python
from zynnova import StructureData, backend_status
from zynnova.structure.molecular import stru2graph
import numpy as np

water = StructureData(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array([[0.0, 0.0, 0.0], [0.958, 0.0, 0.0], [-0.240, 0.927, 0.0]], dtype=float),
)

graph = stru2graph(water, backend="python", neighbor_mode="radius")
print(graph.num_nodes, graph.num_edges)
print(backend_status())
```

## 文档

- `docs/index.md`
- `docs/ZYNNOVA.md`
- `docs/ZYNNOVA_INSTALLATION.md`
- `src/zynnova/ARCHITECTURE.md` / `src/zynnova/BACKEND_CONTRACTS.md`
- `src/zynnova/SOURCE_LOCK.json`

## 许可证

本项目遵循 `LICENSE` 文件中的许可条款，第三方依赖许可见对应说明文件。

---

## English
<a id="zynnova-en"></a>

[中文](#zynnova-cn) | [English](#zynnova-en)

ZynNova is a production-oriented framework for materials intelligence and scientific workflows, organized under a single `zynnova` namespace. This version focuses on reusable structural modules, standardized data and simulation interfaces, and a unified entry point for model inference and experimental pipelines.

## Zero-installation setup

```bash
# 1) Install prerequisites
python --version          # Python 3.10+
git --version
cmake --version

# 2) Clone source
git clone <your-repo-url>
cd ZynNova

# 3) Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

# 4) Install build/runtime dependencies
python -m pip install -U pip setuptools wheel
python -m pip install -e .

# 5) Optional: install development dependencies (tests, linting, experimentation)
python -m pip install -e ".[dev]"
```

For lightweight usage, you can replace step 4 with:

```bash
python -m pip install -e ".[zynnova]"
```

## Package layout

```text
src/
 zynnova/
  core/           # backend registry, serialization and shared infrastructure
  structure/      # structure and molecular-graph data models
  data/           # data sources, loaders, transforms, validation
  dynamics/       # molecular dynamics and trajectory utilities
  ml/
    generation/   # model generation and builder tooling
    mlff/         # force-field / potential workflows
    prediction/   # prediction and inference pipelines
  zynsim/        # FEM, multiscale, and inverse workflows
  zynmorph/      # battery microstructure generation and Tet4 meshing
  zynvista/      # scene reconstruction from image/video and rendering
  zynform/       # object generation, reconstruction and meshing
  zynvox/        # speech experimentation stack
  geometry/      # geometric primitives and conversion helpers
  _native/       # C++ extension namespace
  README.md
```

## Core capabilities

- **Core utilities**
  - `zynnova.core` provides backend registration, runtime conventions and unified entry points
  - Shared serialization, configuration, and provenance utilities
- **Structure & data**
  - `zynnova.structure`, `zynnova.data`, `zynnova.geometry`
  - Data intake, filtering, augmentation and metadata tracking
- **Simulation stack**
  - `zynnova.dynamics` for integrators and trajectory workflows
  - `zynnova.zynsim` for multiscale and inverse workflows
- **Subsystem stack**
  - `zynnova.zynmorph` for conditional microstructure synthesis and meshing
  - `zynnova.zynvista` for reconstruction, visualization, and scene/mesh export
  - `zynnova.zynform` for object processing and reconstruction repair
  - `zynnova.zynvox` for speech-oriented model experiments
- **Machine learning**
  - `zynnova.ml` with `generation`, `mlff`, and `prediction` namespaces

## Install and smoke checks

```bash
python -m pip install -U pip
python -m pip install -e .
python -m pip install -e ".[dev]"
python -m zynnova status
PYTHONPATH=src python scripts/zynnova/verify.py
python scripts/zynnova/static_audit.py
```

## Quick example

```python
from zynnova import StructureData, backend_status
from zynnova.structure.molecular import stru2graph
import numpy as np

water = StructureData(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array([[0.0, 0.0, 0.0], [0.958, 0.0, 0.0], [-0.240, 0.927, 0.0]], dtype=float),
)

graph = stru2graph(water, backend="python", neighbor_mode="radius")
print(graph.num_nodes, graph.num_edges)
print(backend_status())
```

## Documentation

- `docs/index.md`
- `docs/ZYNNOVA.md`
- `docs/ZYNNOVA_INSTALLATION.md`
- `src/zynnova/ARCHITECTURE.md` / `src/zynnova/BACKEND_CONTRACTS.md`
- `src/zynnova/SOURCE_LOCK.json`

## License

This project follows the terms in `LICENSE`. Third-party licenses are documented in their respective files.
