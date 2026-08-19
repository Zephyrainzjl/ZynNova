<a id="top"></a>

<h1 align="center">ZynNova</h1>

<p align="center">
  <img src="docs/assets/zynnova-overview.png" alt="ZynNova framework overview" width="900">
</p>

<p align="center">
  <a href="#zh-cn"><kbd><strong> 中文 </strong></kbd></a>
  &nbsp;&nbsp;
  <a href="#en"><kbd><strong> English </strong></kbd></a>
</p>

---

<a id="zh-cn"></a>

# 中文文档

<p align="right"><a href="#en"><kbd>切换到 English</kbd></a></p>

## ZynNova 是什么？

**ZynNova** 是一个面向科学智能、材料计算、多尺度仿真、三维重建与生成、语音智能以及大模型 Agent 的可扩展 Python/C++ 框架。

当前主干将多个原本容易彼此割裂的能力统一在同一个包中：

- **ZynMorph**：复杂多相微结构、体素、表面网格、四面体有限元网格与工程格式导出；
- **ZynSim**：面向材料、电化学、电池与多物理场问题的数值模拟工作流；
- **ZynVista**：图像/视频条件下的度量场景重建、大世界生成、3DGS/mesh 保持、风格与 DCC 导出；
- **ZynForm**：图像到高保真三维物体、表面修复、物理尺度恢复、多格式导出与 FEM 网格；
- **ZynVox**：带同意记录的语音克隆、TTS、语音转换、数据集处理、训练编排、流式 API 与可选 UI；
- **ZynAstra**：位于 `src/zynnova/llm/zynastra/` 的首个完整 LLM/Agent 框架，可调用 ZynNova 的其他公共能力，并支持云端 API、本地模型、Skills、MCP、Tools、Memory 与 LoRA/QLoRA 微调。

ZynNova 的设计目标不是把所有大型模型权重塞进 Python 包，而是提供稳定的 **API、运行时、协议、质量门、资产管理、工作区管理与外部引擎适配层**。大型语音、LLM、3D/世界生成模型及其 checkpoint 均应下载到用户指定的 **包外工作区**。

> [!IMPORTANT]
> ZynNova 代码仓库应保持可维护和可安装。数 GB 到数百 GB 的模型仓库、训练集和 checkpoint 不应直接提交到 `src/zynnova/`。

---

## 1. 总体架构

```text
ZynNova/
├── src/zynnova/
│   ├── core/                 # 公共数据结构、异常、序列化、后端协议
│   ├── data/                 # 科学与材料数据
│   ├── dynamics/             # 动力学工作流
│   ├── geometry/             # 公共几何、点云、表面、相机、体网格
│   ├── llm/
│   │   └── zynastra/         # 第一个完整、独立的 LLM / Agent 框架
│   │       ├── providers/    # OpenAI-compatible / LiteLLM / 本地 Transformers
│   │       ├── tools/        # Tool registry + ZynNova API bridge
│   │       ├── skills/       # 可移植 SKILL.md 技能
│   │       ├── mcp/          # MCP stdio / Streamable HTTP
│   │       ├── memory.py     # SQLite 会话记忆
│   │       ├── models.py     # 包外模型下载与登记
│   │       ├── finetune.py   # LoRA / QLoRA SFT
│   │       ├── runtime.py    # 多步 Agent runtime
│   │       ├── server.py     # 可选 FastAPI 服务
│   │       └── cli.py        # CLI
│   ├── ml/                   # 机器学习模型与势能模型
│   ├── structure/            # 晶体、分子、高分子等结构表示
│   ├── tools/                # 通用工具
│   ├── visualization/        # 科学可视化
│   ├── zynmorph/             # 微结构、体素、表面、TetGen/FEM
│   ├── zynsim/               # 多物理场/FEM/电池模拟
│   ├── zynvista/             # 场景重建、世界生成、3DGS/mesh、DCC
│   │   ├── external.py       # 外部场景模型统一协议
│   │   ├── model_hub.py      # 外部模型工作区
│   │   └── studio.py         # SceneStudio
│   ├── zynform/              # 物体生成、修复、尺度与 FEM
│   │   ├── external.py       # 外部物体模型统一协议
│   │   ├── model_hub.py      # 外部模型工作区
│   │   └── studio.py         # ObjectStudio
│   └── zynvox/
│       └── studio/            # 数据、训练、推理、API、UI、外部语音引擎
├── tests/
├── pyproject.toml
└── README.md
```

未来新增第二套、第三套 LLM 框架时，推荐保持：

```text
src/zynnova/llm/
├── zynastra/       # 完整框架 1
├── <framework-2>/  # 完整框架 2
└── <framework-3>/  # 完整框架 3
```

每个目录都应该是可以独立演进的完整框架，而不是把所有 Agent 的状态、provider registry 或训练逻辑堆进一个全局对象。

---

## 2. 安装

### 2.1 基础安装

```bash
git clone https://github.com/Zephyrainzjl/ZynNova.git
cd ZynNova
python -m pip install -e .
```

### 2.2 常用科学模块

```bash
python -m pip install -e ".[zynmorph-all]"
python -m pip install -e ".[zynsim-all]"
python -m pip install -e ".[zynnova-scene]"
python -m pip install -e ".[zynnova-object]"
python -m pip install -e ".[zynnova-voice]"
```

### 2.3 ZynVox Studio

基础 Studio、HTTP API 与音频处理：

```bash
python -m pip install -e ".[voice-studio]"
```

增加自动 ASR 标注：

```bash
python -m pip install -e ".[voice-studio-asr]"
```

增加 Gradio UI：

```bash
python -m pip install -e ".[voice-ui]"
```

### 2.4 ZynAstra

云端 OpenAI-compatible Provider：

```bash
python -m pip install -e ".[llm]"
```

更广泛的 Provider 路由：

```bash
python -m pip install -e ".[llm-providers]"
```

MCP：

```bash
python -m pip install -e ".[llm-mcp]"
```

本地 Hugging Face 模型与 LoRA：

```bash
python -m pip install -e ".[llm-local]"
```

Linux / CUDA 下的 QLoRA 环境：

```bash
python -m pip install -e ".[llm-local-qlora]"
```

ZynAstra 全功能：

```bash
python -m pip install -e ".[llm-all]"
```

### 2.5 场景和三维对象外部模型辅助依赖

```bash
python -m pip install -e ".[scene-models,object-models]"
```

> [!NOTE]
> 基础安装不会强制拉取 Transformers、Faster-Whisper、Gradio、训练框架或大型 3D 生成模型。需要哪一组能力再安装对应 extra。

---

## 3. 包外工作区

推荐统一指定：

```bash
# Linux / WSL
export ZYNNOVA_WORKSPACE=/data/zynnova_workspace

# PowerShell
$env:ZYNNOVA_WORKSPACE = "D:\\zynnova_workspace"
```

推荐目录布局：

```text
/data/zynnova_workspace/
├── models/                 # ZynAstra 本地模型快照
├── finetunes/              # LoRA / QLoRA adapter
├── runs/                   # Agent 运行记录
├── skills/                 # 用户 Skills
├── memory/                 # Agent 会话数据库
├── zynvox/
│   ├── datasets/
│   ├── models/
│   ├── engines/
│   ├── voices/
│   ├── runs/
│   └── cache/
├── zynvista/
│   ├── models/
│   └── runs/
└── zynform/
    ├── models/
    └── runs/
```

语音模块也可以单独指定：

```bash
export ZYNNOVA_VOICE_WORKSPACE=/data/voice_workspace
```

场景和对象模块可以分别设置自己的工作区，避免大模型互相污染。

---

## 4. ZynMorph —— 微结构与有限元网格

ZynMorph 面向复杂多相材料和电极微结构，负责从参数化几何、体素或外部几何到可计算网格的转换。

典型能力包括：

- 多相体素结构；
- 正极/负极颗粒、电解液、隔膜等区域；
- 不规则表面提取与平滑；
- 复杂连通拓扑；
- 表面质量修复；
- TetGen 四面体体网格；
- 区域/边界标签保留；
- VTK、MSH、INP、COMSOL `mphtxt` 等工程导出；
- 与 ZynSim 的 FEM/多物理场求解衔接。

ZynMorph 的原则是：**几何标签、区域标签和物理区域必须在体素 → 表面 → 四面体 → 导出的全过程中可追踪。**

---

## 5. ZynSim —— 多尺度与多物理场模拟

ZynSim 是数值求解与工作流层，可用于组织：

- FEM；
- 电化学；
- 传热；
- 传质；
- 多孔介质；
- 电池电极/隔膜结构；
- 相场与相关多物理场流程；
- 网格、材料区域、边界条件和结果导出。

ZynSim 与 ZynMorph 分工明确：ZynMorph 负责“结构怎样变成高质量计算网格”，ZynSim 负责“在这些网格/区域上如何定义并求解物理问题”。

---

## 6. ZynVista —— 场景重建与大世界生成

ZynVista 面向图像/视频条件下的三维场景任务。

核心目标：

- 图像条件场景恢复；
- 视频条件场景重建；
- 度量尺度和相机信息管理；
- 大场景/世界生成；
- 3D Gaussian Splatting 资产；
- mesh 保持与质量检查；
- 深度、法线、相机、材质等辅助资产；
- 风格转换后的几何资产保持；
- Blender、Maya、Houdini 等 DCC 使用的通用导出链。

### 6.1 外部模型协议

快速发展的世界模型不应写死在 ZynVista 主干。`external.py` 和 `SceneStudio` 负责稳定协议：

```text
输入
  ↓
ZynVista SceneRequest
  ↓
外部生成/重建引擎
  ↓
mesh / PBR / 3DGS / cameras / depth / normals / metadata
  ↓
ZynVista 质量检查、尺度、资产整理、导出
```

模型仓库和 checkpoint 位于包外工作区，因此可以随时替换底层模型而不破坏 ZynNova 上层 API。

---

## 7. ZynForm —— 高保真图像到三维物体

ZynForm 聚焦单体物品与工程几何：

- image-to-3D；
- multi-view-to-3D；
- mesh/PBR 资产；
- 缺陷/孔洞/非流形表面修复；
- 物理尺度恢复；
- 重采样与拓扑质量控制；
- OBJ / PLY / STL / GLB 等多格式导出；
- 四面体 FEM 网格；
- 与 ZynMorph/ZynSim 的工程计算链连接。

外部生成模型通过 `ObjectStudio` 统一接入，后处理和工程导出仍由 ZynNova 控制。

---

## 8. ZynVox —— 语音智能框架

ZynVox 由两层组成：

1. 原有语音核心能力；
2. 新增的 **ZynVox Studio**，负责数据、训练、推理、API、UI、外部引擎和工作区。

ZynVox Studio 的目标是让上层调用保持 ZynNova 风格，而不是要求业务代码直接依赖某一个第三方 WebUI。

### 8.1 主要能力

- 同意记录与 voice profile；
- few-shot / zero-shot TTS 工作流；
- 语音转换；
- 数据切片；
- Faster-Whisper 可选自动标注；
- 外部训练阶段编排；
- 外部声学引擎注册；
- GPT-SoVITS 本地服务适配；
- 流式 HTTP 输出；
- FastAPI；
- Gradio 可选 UI；
- 模型/数据/运行产物全部进入包外 workspace。

> [!CAUTION]
> 克隆或转换真实人物声音前，请确认你有相应授权。ZynVox 提供同意记录结构，但技术上的“可执行”不等于你已经获得使用许可。

### 8.2 创建 Studio

```python
from zynnova.zynvox import VoiceWorkspace, ZynVoxStudio

workspace = VoiceWorkspace("/data/zynnova_workspace")
studio = ZynVoxStudio(workspace=workspace)
```

### 8.3 注册一个声音

```python
from zynnova.zynvox import ConsentBasis, ConsentRecord

consent = ConsentRecord(
    confirmed=True,
    basis=ConsentBasis.SELF,
    purpose="personal voice model",
)

profile = studio.enroll_voice(
    voice_id="my_voice",
    reference_audio="/data/reference.wav",
    reference_text="这是一段参考语音。",
    language="zh",
    consent=consent,
)
```

### 8.4 TTS

```python
from zynnova.zynvox import GenerationRequest

result = studio.synthesize(
    GenerationRequest(
        text="欢迎使用 ZynNova。",
        voice_id="my_voice",
        language="zh",
        output_name="hello",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        repetition_penalty=1.35,
        batch_size=1,
        streaming=False,
        parallel_infer=True,
    )
)

print(result.audio)
```

### 8.5 语音转换

当当前 engine 支持 VC 时：

```python
result = studio.voice_convert(
    source_audio="/data/source.wav",
    voice_id="my_voice",
    output_name="converted",
)

print(result.audio)
```

### 8.6 数据集切片

```python
from zynnova.zynvox import DatasetPrepareConfig, prepare_dataset

manifest = prepare_dataset(
    DatasetPrepareConfig(
        dataset_name="speaker_a",
        input_audio="/data/raw_recording.wav",
        language="zh",
        min_segment_s=1.0,
        max_segment_s=15.0,
        transcribe=True,
        whisper_model="large-v3",
        whisper_device="auto",
    ),
    workspace=workspace,
)

print(manifest)
```

不需要 ASR 时可设置 `transcribe=False`。

### 8.7 训练编排

ZynVox Studio 不把第三方训练仓库复制进包内，而是将训练阶段委托给外部 engine/driver：

```python
from zynnova.zynvox import TrainingConfig, VoiceEngineProfile, train_voice_model

engine_profile = VoiceEngineProfile(
    name="my-speech-engine",
    root="/data/external_voice_engine",
    python="python",
)

config = TrainingConfig(
    dataset_manifest=manifest,
    run_name="speaker_a_v1",
    stages=("prepare-text", "ssl-features", "semantic", "acoustic"),
    batch_size=4,
    epochs_semantic=15,
    epochs_acoustic=8,
    precision="bf16",
    device="cuda",
)

result = train_voice_model(config, engine_profile, workspace)
print(result)
```

通过 `stage_commands` 可以为每一个训练阶段指定第三方仓库自己的命令。

### 8.8 GPT-SoVITS 本地引擎适配

当 GPT-SoVITS 仓库已经位于包外目录时：

```python
from zynnova.zynvox import GPTSoVITSLocalConfig, GPTSoVITSLocalEngine

engine = GPTSoVITSLocalEngine(
    GPTSoVITSLocalConfig(
        root="/data/external/GPT-SoVITS",
        python="/data/envs/gpt-sovits/bin/python",
        host="127.0.0.1",
        port=9880,
        gpt_weights="/data/models/gpt.ckpt",
        sovits_weights="/data/models/sovits.pth",
    )
)

studio = ZynVoxStudio(workspace=workspace, engine=engine)
```

该适配器负责管理外部 `api_v2.py` 服务并把 `GenerationRequest` 映射到外部 TTS 参数，包括 top-k/top-p、temperature、batch、速度、seed、并行推理、重复惩罚、流式模式等。

> [!NOTE]
> `GPTSoVITSLocalEngine` 当前是 TTS adapter。语音转换请使用 ZynVox 原有 VC backend，或注册一个同时实现 `synthesize()` 和 `convert()` 的组合/自定义 engine。

### 8.9 自定义外部 Engine

`CommandVoiceEngine` 使用稳定的 JSON job contract 启动任意外部程序。

```python
from zynnova.zynvox import CommandVoiceEngine, VoiceEngineProfile

engine = CommandVoiceEngine(
    VoiceEngineProfile(
        name="custom-engine",
        root="/data/external/custom_voice",
        python="python",
        infer_command=["python", "infer.py"],
        vc_command=["python", "convert.py"],
    )
)
```

ZynNova 写入 job JSON，外部 engine 只需读取 job 并把最终 WAV 写到约定输出路径。

### 8.10 ZynVox 自有 REST API

启动：

```bash
zynvox-studio serve --host 0.0.0.0 --port 8765
```

核心端点：

```text
GET  /v1/health
GET  /v1/voices
POST /v1/voices/enroll
GET  /v1/models
POST /v1/audio/speech
POST /v1/audio/voice-conversion
POST /v1/datasets/prepare
POST /v1/training/run
```

因此客户端只依赖 **ZynVox API**，而不需要绑定第三方 WebUI 的页面逻辑。

### 8.11 Gradio UI

```bash
zynvox-studio ui --host 127.0.0.1 --port 7860
```

UI 是可选入口；Python API 和 REST API 才是稳定主干。

### 8.12 完整测试 Notebook

仓库外可直接运行：

```text
ZynVox_Studio_Full_Test.ipynb
```

Notebook 默认使用合成音频 + dummy/command engine，因此不需要真实人物语音也能覆盖大部分 Studio contract；真实 GPT-SoVITS、真实 VC、ASR、Gradio 被设计成显式开关。

---

## 9. ZynAstra —— LLM / Agent 框架

`zynnova.llm.zynastra` 是 `src/zynnova/llm/` 下的第一套完整 Agent 框架。

### 9.1 核心能力

- OpenAI Responses 路径；
- 通用 OpenAI-compatible Chat Completions；
- SiliconFlow、ModelScope、vLLM、LM Studio 等兼容端点；
- 可选 LiteLLM Provider；
- 本地 Transformers 推理；
- Tool calling；
- ZynNova 公共 API 自动工具化；
- JSON → `dataclass` / `Enum` / `Path` / 容器自动类型转换；
- SQLite session memory；
- Skills；
- MCP stdio；
- MCP Streamable HTTP；
- 多步 Agent loop；
- 多 session 并行执行；
- 结构化输出；
- 本地模型下载；
- LoRA / QLoRA；
- FastAPI；
- CLI。

### 9.2 Provider 与 Agent 解耦

Provider 负责“模型怎样被调用”；Agent runtime 负责：

```text
Prompt
  ↓
Provider
  ↓
Tool Call?
  ├─ No  → Final response
  └─ Yes → Tool Registry → ZynNova/API/MCP/Skill
                    ↓
                 Result
                    ↓
                 Provider
```

因此更换供应商不需要重写 ZynNova tools。

### 9.3 OpenAI-compatible 示例

```python
from zynnova.llm.zynastra import Agent, AgentConfig, ProviderConfig, Workspace
from zynnova.llm.zynastra.providers import create_provider

workspace = Workspace("/data/zynnova_workspace")

provider = create_provider(
    ProviderConfig(
        provider="openai-compatible",
        model="your-model",
        base_url="https://your-compatible-endpoint/v1",
        api_key_env="YOUR_API_KEY",
    )
)

agent = Agent(
    provider,
    workspace,
    config=AgentConfig(name="ZynAstra"),
)

result = await agent.run("分析这个任务并选择合适的工具。")
print(result)
```

### 9.4 SiliconFlow / ModelScope 等

只需要更换：

- `base_url`；
- `model`；
- API key 环境变量。

不要把密钥硬编码到 Python 文件、Notebook 或 Git 仓库。

### 9.5 调用 ZynNova 功能

ZynAstra tool bridge 可以把公共 ZynNova Python API 注册成 Agent tool。复杂参数会从 JSON 自动转换成对应 Python 类型，因此 `SceneRequest`、`ObjectRequest`、FEM 配置等不需要再维护一套重复的数据模型。

### 9.6 Skills

推荐每个技能单独目录：

```text
skills/
└── battery-meshing/
    ├── SKILL.md
    └── ...
```

Skill 用于保存稳定的工作方法、工具约束、领域规范和可复用流程，而不是取代 Python API。

### 9.7 MCP

MCP 适合作为外部工具和数据源协议。ZynAstra 将 MCP 与内部 Python tools 放在同一个 Agent loop 中，上层不需要知道工具来自本地函数还是 MCP server。

### 9.8 本地模型下载

模型必须下载到包外：

```python
from zynnova.llm.zynastra.models import download_model

path = download_model(
    repo_id="your-org/your-model",
    workspace="/data/zynnova_workspace",
)
print(path)
```

### 9.9 LoRA / QLoRA

微调输出同样进入外部 workspace：

```text
/data/zynnova_workspace/finetunes/<run>/
```

ZynNova 负责训练配置、数据入口和产物登记，不把 adapter 或基础模型提交回 Python 包目录。

---

## 10. CLI

### ZynNova

```bash
zynnova --help
```

### ZynAstra

```bash
zynnova-llm --help
```

### ZynVox Studio

```bash
zynvox-studio --help
zynvox-studio speak "你好，ZynNova" --voice my_voice --language zh
zynvox-studio serve --host 0.0.0.0 --port 8765
zynvox-studio ui --host 127.0.0.1 --port 7860
```

---

## 11. 开发原则

### 11.1 不把大型模型写死进主干

主干定义稳定 contract，模型作为外部 engine。

### 11.2 保持科学模块可独立使用

安装 LLM 或语音依赖不应成为运行 ZynMorph/ZynSim 的前提。

### 11.3 保留单位和 provenance

几何、物理量、模型版本、输入、输出、工作区和外部 engine 都应有可追踪元数据。

### 11.4 显式区分参考实现和真实后端

测试中的 dummy engine 只验证协议正确性，不应被表述为真实高质量生成模型。

### 11.5 所有重资产进入外部 workspace

包括：

- LLM checkpoint；
- 语音模型；
- ASR 模型；
- 3D/世界生成模型；
- 训练 checkpoint；
- 大型数据集；
- 运行资产。

---

## 12. 测试

基础测试：

```bash
pytest -q
```

建议分层：

1. **Contract smoke test**：dummy engine、tool loop、JSON contract；
2. **Optional backend test**：有对应依赖时测试真实第三方 engine；
3. **GPU/model integration test**：显式启用，不应作为基础安装的强制测试；
4. **Long-running benchmark**：独立运行，不阻塞普通 CI。

ZynVox 推荐先运行本仓库配套的：

```text
ZynVox_Studio_Full_Test.ipynb
```

---

## 13. 安全、同意与数据治理

ZynNova 的语音、LLM、场景和对象功能可能处理敏感或受授权限制的数据。

请遵循：

- 对真实人物语音，保存明确 consent basis；
- 不把 API key 写入 Git；
- 不把私有训练数据提交到公开仓库；
- 对外部模型遵守其许可证；
- 对生成资产保留模型、版本、输入与处理 provenance；
- 对科学模拟保留单位、网格、材料区域和求解配置。

---

## 14. 项目状态

ZynNova 仍处于持续快速开发阶段。公共 API 会尽量保持稳定，但前沿生成模型、外部模型仓库与第三方服务会继续变化，因此它们应通过 adapter/contract 层接入，而不是成为核心包内部不可替换的实现细节。

---

## 15. License

请以仓库根目录 `LICENSE` 为准。第三方模型、数据集和外部引擎可能拥有不同许可证；使用它们时应分别遵守对应条款。

<p align="right"><a href="#top"><kbd>返回顶部</kbd></a> · <a href="#en"><kbd>English</kbd></a></p>

---
---

<a id="en"></a>

# English Documentation

<p align="right"><a href="#zh-cn"><kbd>切换到中文</kbd></a></p>

## What is ZynNova?

**ZynNova** is an extensible Python/C++ framework for scientific intelligence, materials computation, multiscale simulation, 3D reconstruction and generation, speech intelligence, and tool-using LLM agents.

The current architecture unifies several capabilities that would otherwise become disconnected projects:

- **ZynMorph** — multiphase microstructures, voxels, surfaces, tetrahedral FEM meshes, and engineering exports;
- **ZynSim** — numerical workflows for materials, electrochemistry, batteries, FEM, and multiphysics;
- **ZynVista** — image/video-conditioned metric scene reconstruction, large-world generation, 3DGS/mesh preservation, style workflows, and DCC export;
- **ZynForm** — high-fidelity image-to-object generation, surface repair, physical scaling, multi-format export, and FEM meshing;
- **ZynVox** — consent-aware voice cloning, TTS, voice conversion, dataset preparation, training orchestration, streaming APIs, and an optional UI;
- **ZynAstra** — the first complete LLM/Agent framework under `src/zynnova/llm/zynastra/`, with hosted APIs, local models, Skills, MCP, tools, memory, and LoRA/QLoRA fine-tuning.

ZynNova deliberately does **not** embed every large checkpoint inside the Python package. The repository provides stable **APIs, runtimes, contracts, quality gates, asset management, workspaces, and adapters**. Large speech, LLM, 3D, and world-generation repositories and checkpoints belong in a user-selected workspace outside the package.

> [!IMPORTANT]
> Keep the ZynNova source tree installable and maintainable. Multi-gigabyte model repositories, datasets, and checkpoints should not be committed into `src/zynnova/`.

---

## 1. Architecture

```text
ZynNova/
├── src/zynnova/
│   ├── core/                 # shared structures, errors, serialization, backend contracts
│   ├── data/                 # scientific and materials data
│   ├── dynamics/             # dynamics workflows
│   ├── geometry/             # shared geometry, point clouds, surfaces, cameras, volume meshes
│   ├── llm/
│   │   └── zynastra/         # first complete independent LLM / Agent framework
│   │       ├── providers/    # OpenAI-compatible / LiteLLM / local Transformers
│   │       ├── tools/        # tool registry + ZynNova API bridge
│   │       ├── skills/       # portable SKILL.md skills
│   │       ├── mcp/          # MCP stdio / Streamable HTTP
│   │       ├── memory.py     # SQLite session memory
│   │       ├── models.py     # external-workspace model downloads
│   │       ├── finetune.py   # LoRA / QLoRA SFT
│   │       ├── runtime.py    # multi-step agent runtime
│   │       ├── server.py     # optional FastAPI service
│   │       └── cli.py        # CLI
│   ├── ml/                   # ML models and potentials
│   ├── structure/            # crystal, molecule, polymer representations
│   ├── tools/                # shared utilities
│   ├── visualization/        # reusable scientific visualization
│   ├── zynmorph/             # microstructures, voxels, surfaces, TetGen/FEM
│   ├── zynsim/               # multiphysics / FEM / battery simulation
│   ├── zynvista/             # scene reconstruction, worlds, 3DGS/mesh, DCC
│   │   ├── external.py       # external scene-model contract
│   │   ├── model_hub.py      # external model workspace
│   │   └── studio.py         # SceneStudio
│   ├── zynform/              # object generation, repair, scale, FEM
│   │   ├── external.py       # external object-model contract
│   │   ├── model_hub.py      # external model workspace
│   │   └── studio.py         # ObjectStudio
│   └── zynvox/
│       └── studio/            # data, training, inference, API, UI, external voice engines
├── tests/
├── pyproject.toml
└── README.md
```

Future LLM frameworks should remain independent siblings:

```text
src/zynnova/llm/
├── zynastra/       # complete framework 1
├── <framework-2>/  # complete framework 2
└── <framework-3>/  # complete framework 3
```

Each directory can evolve as a complete framework without sharing mutable global provider registries or agent state.

---

## 2. Installation

### 2.1 Core

```bash
git clone https://github.com/Zephyrainzjl/ZynNova.git
cd ZynNova
python -m pip install -e .
```

### 2.2 Scientific subsystems

```bash
python -m pip install -e ".[zynmorph-all]"
python -m pip install -e ".[zynsim-all]"
python -m pip install -e ".[zynnova-scene]"
python -m pip install -e ".[zynnova-object]"
python -m pip install -e ".[zynnova-voice]"
```

### 2.3 ZynVox Studio

```bash
python -m pip install -e ".[voice-studio]"
```

Optional ASR labeling:

```bash
python -m pip install -e ".[voice-studio-asr]"
```

Optional Gradio UI:

```bash
python -m pip install -e ".[voice-ui]"
```

### 2.4 ZynAstra

Hosted OpenAI-compatible APIs:

```bash
python -m pip install -e ".[llm]"
```

Broad provider routing:

```bash
python -m pip install -e ".[llm-providers]"
```

MCP:

```bash
python -m pip install -e ".[llm-mcp]"
```

Local Hugging Face models and LoRA:

```bash
python -m pip install -e ".[llm-local]"
```

QLoRA on a suitable Linux/CUDA environment:

```bash
python -m pip install -e ".[llm-local-qlora]"
```

Combined ZynAstra feature set:

```bash
python -m pip install -e ".[llm-all]"
```

### 2.5 External 3D model helpers

```bash
python -m pip install -e ".[scene-models,object-models]"
```

> [!NOTE]
> The core installation does not force Transformers, Faster-Whisper, Gradio, training stacks, or large 3D-model dependencies onto scientific-only users.

---

## 3. External workspaces

Set a common root:

```bash
# Linux / WSL
export ZYNNOVA_WORKSPACE=/data/zynnova_workspace

# PowerShell
$env:ZYNNOVA_WORKSPACE = "D:\\zynnova_workspace"
```

Recommended layout:

```text
/data/zynnova_workspace/
├── models/
├── finetunes/
├── runs/
├── skills/
├── memory/
├── zynvox/
│   ├── datasets/
│   ├── models/
│   ├── engines/
│   ├── voices/
│   ├── runs/
│   └── cache/
├── zynvista/
│   ├── models/
│   └── runs/
└── zynform/
    ├── models/
    └── runs/
```

Speech can use a dedicated location through `ZYNNOVA_VOICE_WORKSPACE`. Scene and object subsystems can likewise use subsystem-specific workspaces.

---

## 4. ZynMorph — microstructures and FEM meshes

ZynMorph handles complex multiphase material structures and turns parameterized geometry, voxels, or imported geometry into computational meshes.

Typical capabilities include:

- multiphase voxel structures;
- electrode particles, electrolyte, separators, and other material regions;
- irregular surface extraction and smoothing;
- complex connected topology;
- surface repair;
- TetGen tetrahedralization;
- region and boundary label preservation;
- VTK, MSH, INP, COMSOL `mphtxt`, and related exports;
- integration with ZynSim FEM/multiphysics workflows.

The key rule is traceability: geometry labels and physical regions should survive voxel → surface → tetrahedra → export transformations.

---

## 5. ZynSim — multiscale and multiphysics simulation

ZynSim organizes numerical workflows for areas such as:

- FEM;
- electrochemistry;
- heat transfer;
- mass transport;
- porous media;
- battery electrodes and separators;
- phase-field and related multiphysics workflows;
- meshes, materials, boundary conditions, and result export.

ZynMorph answers “how does a structure become a high-quality computational mesh?” while ZynSim answers “how is a physical problem defined and solved on those regions and meshes?”

---

## 6. ZynVista — scene reconstruction and large worlds

ZynVista targets image/video-conditioned 3D scenes:

- image-conditioned scene recovery;
- video-conditioned reconstruction;
- metric scale and camera handling;
- large-scene/world generation;
- 3D Gaussian Splatting assets;
- mesh preservation and geometry audits;
- depth, normals, cameras, materials, and auxiliary assets;
- style workflows that preserve usable geometry;
- asset export for Blender, Maya, Houdini, and other DCC tools.

### 6.1 External-model contract

Fast-moving world models should not be hard-coded into the ZynVista core:

```text
Input
  ↓
ZynVista SceneRequest
  ↓
external reconstruction / generation engine
  ↓
mesh / PBR / 3DGS / cameras / depth / normals / metadata
  ↓
ZynVista QA, metric handling, asset organization, export
```

Repositories and checkpoints remain outside the package, so the underlying model can be replaced without changing the upper-level ZynNova API.

---

## 7. ZynForm — high-fidelity image-to-object workflows

ZynForm focuses on individual objects and engineering-ready geometry:

- image-to-3D;
- multi-view-to-3D;
- mesh/PBR assets;
- hole/non-manifold/surface repair;
- physical scaling;
- remeshing and topology quality control;
- OBJ / PLY / STL / GLB and related export;
- tetrahedral FEM meshes;
- integration with ZynMorph and ZynSim.

External generators enter through `ObjectStudio`; engineering post-processing remains controlled by ZynNova.

---

## 8. ZynVox — speech intelligence

ZynVox contains the original speech functionality plus **ZynVox Studio**, which provides dataset preparation, training orchestration, inference, APIs, UI, external-engine management, and workspaces.

The goal is for application code to depend on a stable ZynNova API rather than directly depending on one third-party WebUI.

### 8.1 Capabilities

- consent records and voice profiles;
- few-shot / zero-shot TTS workflows;
- voice conversion;
- dataset segmentation;
- optional Faster-Whisper transcription;
- external training-stage orchestration;
- external speech-engine registration;
- managed local GPT-SoVITS TTS adapter;
- streaming HTTP output;
- FastAPI service;
- optional Gradio UI;
- model/data/run artifacts in an external workspace.

> [!CAUTION]
> Before cloning or converting a real person's voice, make sure you have appropriate authorization. A technical consent-record structure is not a substitute for permission.

### 8.2 Create a Studio

```python
from zynnova.zynvox import VoiceWorkspace, ZynVoxStudio

workspace = VoiceWorkspace("/data/zynnova_workspace")
studio = ZynVoxStudio(workspace=workspace)
```

### 8.3 Enroll a voice

```python
from zynnova.zynvox import ConsentBasis, ConsentRecord

consent = ConsentRecord(
    confirmed=True,
    basis=ConsentBasis.SELF,
    purpose="personal voice model",
)

profile = studio.enroll_voice(
    voice_id="my_voice",
    reference_audio="/data/reference.wav",
    reference_text="This is a reference recording.",
    language="en",
    consent=consent,
)
```

### 8.4 TTS

```python
from zynnova.zynvox import GenerationRequest

result = studio.synthesize(
    GenerationRequest(
        text="Welcome to ZynNova.",
        voice_id="my_voice",
        language="en",
        output_name="hello",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        repetition_penalty=1.35,
        batch_size=1,
        streaming=False,
        parallel_infer=True,
    )
)

print(result.audio)
```

### 8.5 Voice conversion

When the selected engine supports VC:

```python
result = studio.voice_convert(
    source_audio="/data/source.wav",
    voice_id="my_voice",
    output_name="converted",
)

print(result.audio)
```

### 8.6 Dataset preparation

```python
from zynnova.zynvox import DatasetPrepareConfig, prepare_dataset

manifest = prepare_dataset(
    DatasetPrepareConfig(
        dataset_name="speaker_a",
        input_audio="/data/raw_recording.wav",
        language="en",
        min_segment_s=1.0,
        max_segment_s=15.0,
        transcribe=True,
        whisper_model="large-v3",
        whisper_device="auto",
    ),
    workspace=workspace,
)
```

Use `transcribe=False` when ASR is not required.

### 8.7 Training orchestration

Training remains an external-engine responsibility while ZynVox owns the stable job model:

```python
from zynnova.zynvox import TrainingConfig, VoiceEngineProfile, train_voice_model

engine_profile = VoiceEngineProfile(
    name="my-speech-engine",
    root="/data/external_voice_engine",
    python="python",
)

config = TrainingConfig(
    dataset_manifest=manifest,
    run_name="speaker_a_v1",
    stages=("prepare-text", "ssl-features", "semantic", "acoustic"),
    batch_size=4,
    epochs_semantic=15,
    epochs_acoustic=8,
    precision="bf16",
    device="cuda",
)

result = train_voice_model(config, engine_profile, workspace)
```

Use `stage_commands` when the external repository requires custom commands for individual training stages.

### 8.8 Managed local GPT-SoVITS adapter

If a GPT-SoVITS repository is already present outside the package:

```python
from zynnova.zynvox import GPTSoVITSLocalConfig, GPTSoVITSLocalEngine

engine = GPTSoVITSLocalEngine(
    GPTSoVITSLocalConfig(
        root="/data/external/GPT-SoVITS",
        python="/data/envs/gpt-sovits/bin/python",
        host="127.0.0.1",
        port=9880,
        gpt_weights="/data/models/gpt.ckpt",
        sovits_weights="/data/models/sovits.pth",
    )
)

studio = ZynVoxStudio(workspace=workspace, engine=engine)
```

The adapter manages the external `api_v2.py` TTS service and maps `GenerationRequest` fields such as top-k/top-p, temperature, batch size, speed, seed, parallel inference, repetition penalty, and streaming mode.

> [!NOTE]
> `GPTSoVITSLocalEngine` is currently a TTS adapter. Use the existing ZynVox VC backend or a composite/custom engine implementing both `synthesize()` and `convert()` for voice conversion.

### 8.9 Custom external engine

`CommandVoiceEngine` provides a stable JSON job contract:

```python
from zynnova.zynvox import CommandVoiceEngine, VoiceEngineProfile

engine = CommandVoiceEngine(
    VoiceEngineProfile(
        name="custom-engine",
        root="/data/external/custom_voice",
        python="python",
        infer_command=["python", "infer.py"],
        vc_command=["python", "convert.py"],
    )
)
```

ZynNova writes a job document; the external driver reads it and writes the resulting WAV to the requested output path.

### 8.10 First-party REST API

Start the server:

```bash
zynvox-studio serve --host 0.0.0.0 --port 8765
```

Core endpoints:

```text
GET  /v1/health
GET  /v1/voices
POST /v1/voices/enroll
GET  /v1/models
POST /v1/audio/speech
POST /v1/audio/voice-conversion
POST /v1/datasets/prepare
POST /v1/training/run
```

Clients therefore depend on the **ZynVox API**, not a third-party WebUI page contract.

### 8.11 Optional Gradio UI

```bash
zynvox-studio ui --host 127.0.0.1 --port 7860
```

The Python API and REST API remain the stable core; the UI is optional.

### 8.12 Full test notebook

Use:

```text
ZynVox_Studio_Full_Test.ipynb
```

The notebook defaults to synthetic audio plus dummy/command engines, so most Studio contracts can be tested without using a real person's voice. Real GPT-SoVITS, ASR, legacy VC, and Gradio tests are explicit opt-in sections.

---

## 9. ZynAstra — LLM / Agent framework

`zynnova.llm.zynastra` is the first complete framework under `src/zynnova/llm/`.

### 9.1 Core capabilities

- OpenAI Responses path;
- generic OpenAI-compatible Chat Completions;
- compatible SiliconFlow, ModelScope, vLLM, LM Studio, and private endpoints;
- optional LiteLLM provider;
- local Transformers inference;
- tool calling;
- automatic ZynNova public-API tools;
- JSON → `dataclass` / `Enum` / `Path` / container conversion;
- SQLite session memory;
- Skills;
- MCP stdio;
- MCP Streamable HTTP;
- multi-step agent loops;
- concurrent sessions;
- structured output;
- local model downloads;
- LoRA / QLoRA;
- FastAPI;
- CLI.

### 9.2 Provider/runtime separation

Providers answer “how is the model called?” while the agent runtime owns the tool loop:

```text
Prompt
  ↓
Provider
  ↓
Tool Call?
  ├─ No  → Final response
  └─ Yes → Tool Registry → ZynNova/API/MCP/Skill
                    ↓
                 Result
                    ↓
                 Provider
```

Changing providers therefore does not require rewriting ZynNova tools.

### 9.3 OpenAI-compatible example

```python
from zynnova.llm.zynastra import Agent, AgentConfig, ProviderConfig, Workspace
from zynnova.llm.zynastra.providers import create_provider

workspace = Workspace("/data/zynnova_workspace")

provider = create_provider(
    ProviderConfig(
        provider="openai-compatible",
        model="your-model",
        base_url="https://your-compatible-endpoint/v1",
        api_key_env="YOUR_API_KEY",
    )
)

agent = Agent(
    provider,
    workspace,
    config=AgentConfig(name="ZynAstra"),
)

result = await agent.run("Analyze this task and choose the appropriate tools.")
print(result)
```

### 9.4 SiliconFlow, ModelScope, and compatible endpoints

Change only the endpoint, model ID, and API-key environment variable when the service implements the expected OpenAI-compatible contract.

Never hard-code API keys into Python files, notebooks, or Git repositories.

### 9.5 ZynNova tools

The ZynAstra bridge can expose public ZynNova Python APIs as agent tools. Complex JSON arguments are converted into their Python types, avoiding duplicate schemas for scene requests, object requests, FEM configuration objects, and similar APIs.

### 9.6 Skills

Recommended structure:

```text
skills/
└── battery-meshing/
    ├── SKILL.md
    └── ...
```

Skills encode reusable workflows, domain rules, and tool instructions. They complement rather than replace Python APIs.

### 9.7 MCP

MCP tools can participate in the same agent loop as local Python tools, allowing external services and data sources to be integrated without changing the high-level agent interface.

### 9.8 Local model downloads

Keep snapshots outside the package:

```python
from zynnova.llm.zynastra.models import download_model

path = download_model(
    repo_id="your-org/your-model",
    workspace="/data/zynnova_workspace",
)
print(path)
```

### 9.9 LoRA / QLoRA

Fine-tuning artifacts likewise stay outside the source tree:

```text
/data/zynnova_workspace/finetunes/<run>/
```

---

## 10. CLI

### ZynNova

```bash
zynnova --help
```

### ZynAstra

```bash
zynnova-llm --help
```

### ZynVox Studio

```bash
zynvox-studio --help
zynvox-studio speak "Hello ZynNova" --voice my_voice --language en
zynvox-studio serve --host 0.0.0.0 --port 8765
zynvox-studio ui --host 127.0.0.1 --port 7860
```

---

## 11. Development principles

### 11.1 Do not hard-code large models into the core

The core defines stable contracts; models remain replaceable external engines.

### 11.2 Keep scientific modules independently usable

LLM and speech dependencies must not become prerequisites for ZynMorph or ZynSim.

### 11.3 Preserve units and provenance

Geometry, physical quantities, model versions, inputs, outputs, workspaces, and external engines should remain traceable.

### 11.4 Distinguish smoke-test engines from production models

A dummy engine validates contracts; it does not claim production-quality speech or 3D generation.

### 11.5 Put heavyweight artifacts in external workspaces

This includes LLM checkpoints, speech models, ASR models, 3D/world-generation models, training checkpoints, large datasets, and generated assets.

---

## 12. Testing

Run the basic suite with:

```bash
pytest -q
```

Recommended test layers:

1. **Contract smoke tests** — dummy engines, tool loops, JSON contracts;
2. **Optional backend tests** — real third-party engines when dependencies are present;
3. **GPU/model integration tests** — explicitly enabled, never mandatory for a lightweight installation;
4. **Long-running benchmarks** — separate from normal CI.

For ZynVox, start with:

```text
ZynVox_Studio_Full_Test.ipynb
```

---

## 13. Safety, consent, and data governance

ZynNova speech, LLM, scene, and object workflows may process sensitive or licensed data.

Recommended practices:

- retain an explicit consent basis for real-person speech;
- never commit API keys;
- do not publish private training data by accident;
- respect external model licenses;
- retain model/version/input/processing provenance for generated assets;
- retain units, meshes, material regions, and solver configuration for scientific simulations.

---

## 14. Project status

ZynNova is under active development. Public APIs are intended to remain reasonably stable, while frontier generative models, external model repositories, and hosted services will continue to change. Those volatile dependencies should enter through adapters and contracts rather than becoming irreplaceable implementation details inside the core package.

---

## 15. License

See the repository-root `LICENSE`. Third-party models, datasets, and external engines may use different licenses and must be handled under their respective terms.

<p align="right"><a href="#top"><kbd>Back to top</kbd></a> · <a href="#zh-cn"><kbd>中文</kbd></a></p>
