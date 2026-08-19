# ZynNova 0.3.0

**ZynNova** is an extensible Python/C++ framework for materials intelligence, scientific simulation, microstructure/FEM workflows, modern 3D reconstruction and generation, consent-aware speech, and tool-using large-language-model agents.

Version 0.3.0 keeps the existing scientific stack intact and adds three large capabilities without forcing heavyweight models into the Python package:

1. **ZynAstra** — a complete provider-neutral LLM/agent framework under `src/zynnova/llm/zynastra/`.
2. **ZynVox Studio** — a first-party voice API and local training/inference orchestration layer designed for GPT-SoVITS-class workflows.
3. **ZynVista / ZynForm Studio** — stable contracts for fast-moving world and object generators, while retaining ZynNova's metric geometry, repair, DCC export, and FEM stack.

> Heavy checkpoints and external model repositories are intentionally stored in a user-selected workspace outside the installed package. ZynNova contains APIs, orchestration, contracts, quality gates, provenance, and adapters—not multi-gigabyte model weights.

---

## 1. Package architecture

```text
src/zynnova/
├── core/                 shared manifests, exceptions, backend contracts, serialization
├── data/                 materials/scientific datasets
├── dynamics/             atomistic/dynamics workflows
├── geometry/             common point-cloud, surface, camera and volume-mesh geometry
├── llm/
│   └── zynastra/         first complete independent LLM/agent framework
│       ├── providers/    OpenAI-compatible, LiteLLM, local Transformers
│       ├── tools/        typed tool registry + whole-ZynNova bridge
│       ├── skills/       portable SKILL.md packages
│       ├── mcp/          Model Context Protocol client bridge
│       ├── memory.py     persistent SQLite sessions
│       ├── models.py     external-workspace model download
│       ├── finetune.py   LoRA / QLoRA SFT
│       ├── runtime.py    multi-step tool-using agent loop
│       ├── server.py     optional FastAPI service
│       └── cli.py        command-line interface
├── ml/                   machine-learning models and potentials
├── structure/            crystal/molecule/polymer representations and conversions
├── tools/                common utilities
├── visualization/        reusable scientific visualization
├── zynmorph/             microstructure generation, voxel/surface/volume conversion, TetGen
├── zynsim/               multiphysics / FEM / battery-scale simulation
├── zynvista/             metric scene reconstruction, world generation, 3DGS/mesh, DCC export
│   ├── external.py       external world-generator contract
│   ├── model_hub.py      external model workspace helper
│   └── studio.py         high-level SceneStudio
├── zynform/              image/object generation, repair, physical scaling, FEM meshing
│   ├── external.py       external object-generator contract
│   ├── model_hub.py      external model workspace helper
│   └── studio.py         high-level ObjectStudio
└── zynvox/               consent-aware voice conversion and TTS
    └── studio/            first-party dataset/training/inference/API/UI layer
```

Every future LLM framework should live beside ZynAstra:

```text
src/zynnova/llm/
├── zynastra/       complete framework #1
├── <future-2>/     complete framework #2
└── <future-3>/     complete framework #3
```

A future framework does not need to inherit ZynAstra's mutable global state or provider registry. This is intentional: each framework can evolve independently and coexist in one ZynNova installation.

---

## 2. Installation

### Lightweight core

```bash
git clone https://github.com/Zephyrainzjl/ZynNova.git
cd ZynNova
pip install -e .
```

### Scientific / existing subsystems

Use the existing extras as before, for example:

```bash
pip install -e ".[zynmorph-all]"
pip install -e ".[zynsim-all]"
pip install -e ".[zynnova-scene]"
pip install -e ".[zynnova-object]"
pip install -e ".[zynnova-voice]"
```

### ZynAstra LLM

Hosted OpenAI-compatible endpoints only:

```bash
pip install -e ".[llm]"
```

Broad provider coverage with LiteLLM:

```bash
pip install -e ".[llm-providers]"
```

MCP:

```bash
pip install -e ".[llm-mcp]"
```

FastAPI server:

```bash
pip install -e ".[llm-server]"
```

Local Hugging Face models + LoRA training:

```bash
pip install -e ".[llm-local]"
```

Linux QLoRA environment:

```bash
pip install -e ".[llm-local-qlora]"
```

Combined ZynAstra feature set:

```bash
pip install -e ".[llm-all]"
```

### ZynVox Studio

```bash
pip install -e ".[voice-studio]"
```

Add automatic ASR labeling:

```bash
pip install -e ".[voice-studio-asr]"
```

Add Gradio UI:

```bash
pip install -e ".[voice-ui]"
```

### External 3D model download helpers

```bash
pip install -e ".[scene-models,object-models]"
```

`all` deliberately remains a scientific convenience extra rather than silently pulling every LLM, speech-ASR, UI, and generative-model dependency into every installation.

---

## 3. External workspaces

Set one common root:

```bash
# Linux / WSL
export ZYNNOVA_WORKSPACE=/data/zynnova_workspace

# PowerShell
$env:ZYNNOVA_WORKSPACE = "D:\\zynnova_workspace"
```

The new systems create their own children under that root:

```text
/data/zynnova_workspace/
├── models/               ZynAstra model snapshots
├── finetunes/            LoRA/QLoRA adapters
├── runs/                 agent runs
├── skills/               user-installed skills
├── memory/               agent session database
├── zynvox/
│   ├── datasets/
│   ├── models/
│   ├── engines/
│   ├── voices/
│   └── runs/
├── zynvista/
│   ├── models/
│   └── runs/
└── zynform/
    ├── models/
    └── runs/
```

You can instead set subsystem-specific roots with `ZYNNOVA_VOICE_WORKSPACE`, `ZYNNOVA_SCENE_WORKSPACE`, or `ZYNNOVA_OBJECT_WORKSPACE`.

---

# 4. ZynAstra — complete LLM / Agent framework

ZynAstra is the first self-contained framework under `zynnova.llm`. It is not tied to one provider.

Core capabilities:

- OpenAI Responses API path for OpenAI endpoints.
- Generic OpenAI-compatible Chat Completions path for SiliconFlow, ModelScope, vLLM, LM Studio, compatible gateways, and private endpoints.
- Optional LiteLLM provider for broad provider routing.
- Local `transformers` inference from a model directory in the external workspace.
- Multi-step function/tool calling.
- Provider-native Responses tools when a provider supports them.
- Structured-output / JSON-schema configuration hooks.
- Multimodal message blocks through the provider-neutral `Message` representation.
- Parallel independent sessions with `Agent.run_many(...)`.
- Automatic JSON → Python `dataclass`, `Enum`, `Path`, list/tuple/dict conversion when calling public ZynNova APIs.
- Persistent session memory with SQLite.
- Portable `SKILL.md` skills.
- MCP stdio and Streamable HTTP clients through the official Python MCP SDK when installed.
- Local Hugging Face snapshot download into a chosen workspace.
- LoRA / QLoRA supervised fine-tuning into a chosen workspace.
- Optional FastAPI service.
- CLI.
- No API keys are written to workspace metadata; keys are read from environment variables.

## 4.1 OpenAI

```python
import asyncio
from zynnova.llm.zynastra import Agent, AgentConfig, ProviderConfig, Workspace

async def main():
    provider = ProviderConfig.openai(
        model="gpt-5.6",
        reasoning_effort="high",
        # Optional OpenAI Responses-native tools stay provider-specific:
        # native_tools=({"type": "web_search"},),
    )

    agent = Agent.create(
        provider,
        Workspace("/data/zynnova_workspace"),
        config=AgentConfig(max_steps=20),
    )

    try:
        result = await agent.run(
            "Inspect the available ZynNova scene APIs and tell me how to reconstruct a video."
        )
        print(result.text)
        print(result.session_id)
    finally:
        await agent.aclose()

asyncio.run(main())
```

Set:

```bash
export OPENAI_API_KEY=...
```

## 4.2 SiliconFlow

```python
provider = ProviderConfig.siliconflow(
    model="<your-siliconflow-model-id>",
)
agent = Agent.create(provider, "/data/zynnova_workspace")
```

```bash
export SILICONFLOW_API_KEY=...
```

## 4.3 ModelScope inference API

```python
provider = ProviderConfig.modelscope(
    model="<your-modelscope-model-id>",
)
```

```bash
export MODELSCOPE_API_KEY=...
```

## 4.4 Any OpenAI-compatible endpoint

```python
from zynnova.llm.zynastra import ProviderConfig

provider = ProviderConfig(
    kind="openai-compatible",
    model="my-model",
    base_url="http://127.0.0.1:8000/v1",
    api_key_env="MY_API_KEY",
    api_style="chat-completions",
)
```

This is the preferred path for vLLM, private gateways, self-hosted OpenAI-compatible servers, and vendors that expose compatible endpoints.

## 4.5 LiteLLM

```python
provider = ProviderConfig(
    kind="litellm",
    model="<litellm-provider>/<model>",
    base_url="litellm://",
    api_key_env="MY_PROVIDER_KEY",
)
```

Provider-specific model strings and credentials remain the responsibility of the provider configuration; ZynAstra does not hard-code every vendor into the core runtime.

For ensemble/delegated workloads, independent sessions can run concurrently:

```python
results = await agent.run_many(
    ["analyze route A", "analyze route B", "audit the assumptions"],
    max_concurrency=3,
)
```

---

## 4.6 Calling all public ZynNova functions

Every ZynAstra instance can install two generic scientific tools:

- `zynnova_list_api` — inspect callable APIs in a namespace.
- `zynnova_call` — invoke a public function beneath an allowlisted ZynNova subsystem.

Example agent intent:

```text
1. list zynnova.zynvista
2. construct SceneRequest and SceneConfig from JSON
3. call zynnova.zynvista.run_scene
4. inspect the returned run directory / manifest
5. summarize the result
```

The tool bridge performs best-effort conversion from JSON objects to annotated Python dataclasses and enums. It rejects private attributes and namespaces outside the configured allowlist.

You can restrict tool access:

```python
from zynnova.llm.zynastra import AgentConfig

config = AgentConfig(
    allowed_zynnova_roots=("zynvista", "zynform", "zynmorph"),
)
```

---

## 4.7 Skills

A skill is a directory containing `SKILL.md` and optional `manifest.json`:

```text
/data/zynnova_workspace/skills/electrode-meshing/
├── SKILL.md
└── manifest.json
```

Example `manifest.json`:

```json
{
  "name": "electrode-meshing",
  "version": "1.0"
}
```

The skill instructions are discovered by `SkillManager` and incorporated into the agent system context. This format is deliberately file-based, inspectable, and easy to version-control outside the package.

---

## 4.8 MCP

```python
from zynnova.llm.zynastra import AgentConfig, MCPServerConfig

config = AgentConfig(
    mcp_servers=(
        MCPServerConfig(
            name="my_server",
            transport="stdio",
            command="python",
            args=("/data/mcp/my_server.py",),
        ),
    )
)
```

Streamable HTTP:

```python
MCPServerConfig(
    name="remote_tools",
    transport="streamable-http",
    url="https://example.internal/mcp",
)
```

MCP tool names are namespaced as `mcp__<server>__<tool>` to prevent collisions with ZynNova-native tools and other MCP servers.

---

## 4.9 Download a model outside the package

```python
from zynnova.llm.zynastra import Workspace, download_model

workspace = Workspace("/data/zynnova_workspace").ensure()
local = download_model(
    "<hugging-face-model-id>",
    workspace,
)
print(local.path)
```

CLI:

```bash
zynnova-llm --workspace /data/zynnova_workspace \
  download-model <hugging-face-model-id>
```

The implementation uses Hugging Face snapshot download with `local_dir=...`; the resulting snapshot is under the external workspace, not `site-packages/zynnova` and not your git repository.

---

## 4.10 Local inference

```python
from zynnova.llm.zynastra import Agent, ProviderConfig

provider = ProviderConfig.local(
    "/data/zynnova_workspace/models/my-local-model"
)
agent = Agent.create(provider, "/data/zynnova_workspace")
```

The local provider uses `AutoTokenizer` + `AutoModelForCausalLM` and `device_map="auto"`.

---

## 4.11 LoRA / QLoRA fine-tuning

Input data can be a Hugging Face dataset name or local JSON/JSONL/CSV.

```python
from zynnova.llm.zynastra import SFTConfig, Workspace, finetune_lora

adapter = finetune_lora(
    "/data/zynnova_workspace/models/base-model",
    SFTConfig(
        dataset="/data/my_task/train.jsonl",
        text_field="text",
        output_name="battery-agent",
        epochs=2,
        learning_rate=1e-4,
        lora_r=32,
        lora_alpha=64,
        qlora_4bit=True,
    ),
    Workspace("/data/zynnova_workspace"),
)
print(adapter)
```

All checkpoints and final adapters are written below `workspace/finetunes`.

---

## 4.12 ZynAstra API server

```bash
zynnova-llm \
  --workspace /data/zynnova_workspace \
  serve \
  --provider openai-compatible \
  --model gpt-5.6 \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --api-style responses \
  --host 127.0.0.1 \
  --port 8765
```

Endpoints:

```text
GET  /v1/health
GET  /v1/tools
POST /v1/agent/run
```

---

# 5. ZynVox Studio — GPT-SoVITS-class workflow surface with ZynNova's own API

The previous ZynVox APIs remain available:

- consent-gated voice conversion,
- zero/few-shot TTS backends,
- GPT-SoVITS API adapter,
- CosyVoice / IndexTTS adapters,
- benchmarking,
- provenance and disclosure markers.

0.3.0 adds a first-party **ZynVox Studio** above those backends.

The Studio API covers the workflow surface expected from a GPT-SoVITS-class system:

- reference-voice enrollment,
- few/zero-shot reference-conditioned TTS,
- voice conversion,
- language and reference transcript conditioning,
- seed / top-k / top-p / temperature / repetition penalty,
- speed control,
- batching and parallel-inference hints,
- streaming transport,
- long-text split-mode hints,
- dataset slicing and normalization,
- optional Faster-Whisper transcription,
- staged local training orchestration,
- external model/checkpoint registry,
- Python API,
- REST API,
- optional Gradio UI,
- existing ZynVox consent / provenance boundary.

**Important quality boundary:** ZynVox Studio supplies the complete first-party workflow and API contract, but speech quality is determined by the selected acoustic/semantic engine and checkpoints. ZynNova does not pretend that a wrapper by itself recreates a neural architecture. To reach GPT-SoVITS-level acoustic quality, attach a compatible local engine/checkpoint through `CommandVoiceEngine` or use one of the existing mature ZynVox backends.

## 5.1 Enroll an authorized voice

```python
from zynnova.zynvox import ConsentBasis, ConsentRecord
from zynnova.zynvox.studio import ZynVoxStudio

studio = ZynVoxStudio("/data/zynnova_workspace")

consent = ConsentRecord(
    confirmed=True,
    basis=ConsentBasis.SELF,
    purpose="my local TTS model",
)

studio.enroll_voice(
    "my_voice",
    "/data/voice/reference.wav",
    consent,
    reference_text="This is the transcript of the reference clip.",
    language="zh",
)
```

For non-self voices, the existing ZynVox policy requires concrete authorization/license/source evidence.

## 5.2 Synthesize

```python
from zynnova.zynvox.studio import GenerationRequest

result = studio.synthesize(
    GenerationRequest(
        text="你好，这是 ZynVox Studio。",
        voice_id="my_voice",
        language="zh",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        speed=1.0,
        seed=1234,
    )
)
print(result.audio)
```

By default the Studio uses the legacy ZynVox adapter so existing configured backends continue to work.

---

## 5.3 Managed local GPT-SoVITS engine

For an external GPT-SoVITS checkout, ZynNova can own the process lifecycle and translate `GenerationRequest` directly to the current local `api_v2.py` `/tts` contract. Your application still talks only to the ZynVox Studio API.

```python
from zynnova.zynvox.studio import (
    GPTSoVITSLocalConfig,
    GPTSoVITSLocalEngine,
    ZynVoxStudio,
)

engine = GPTSoVITSLocalEngine(
    GPTSoVITSLocalConfig(
        root="/data/external/GPT-SoVITS",
        python="/data/envs/gpt-sovits/bin/python",
        host="127.0.0.1",
        port=9880,
        tts_config="GPT_SoVITS/configs/tts_infer.yaml",
        # optional: switch trained weights after startup
        # gpt_weights="/data/zynnova_workspace/zynvox/models/voice/s1.ckpt",
        # sovits_weights="/data/zynnova_workspace/zynvox/models/voice/s2.pth",
    )
)

studio = ZynVoxStudio("/data/zynnova_workspace", engine=engine)
```

This adapter exposes upstream controls including top-k, top-p, temperature, split method, batch size, speed, fragment interval, seed, parallel inference, repetition penalty, VITS sampling steps, super-sampling, and streaming modes through the ZynVox request/`extra` fields. It can reuse an already-running local `api_v2.py` port or start/stop the process itself.

## 5.4 Attach any other GPT-SoVITS-class engine

External voice repositories live outside ZynNova:

```text
/data/zynnova_workspace/zynvox/engines/my_engine/
```

Configure a stable command driver:

```python
from zynnova.zynvox.studio import (
    CommandVoiceEngine,
    VoiceEngineProfile,
    ZynVoxStudio,
)

profile = VoiceEngineProfile(
    name="my-gpt-sovits-class-engine",
    root="/data/zynnova_workspace/zynvox/engines/my_engine",
    python="/data/envs/voice/bin/python",
    infer_module="zynnova_voice_driver.infer",
    vc_module="zynnova_voice_driver.vc",
)

studio = ZynVoxStudio(
    "/data/zynnova_workspace",
    engine=CommandVoiceEngine(profile),
)
```

The driver receives:

```bash
python -m zynnova_voice_driver.infer --zynnova-job /path/to/output.wav.job.json
```

The job is JSON and contains a stable `zynnova-zynvox-studio-v1` contract, including:

- requested output path,
- target voice/reference audio,
- reference transcript,
- language,
- model identifier,
- sampling controls,
- streaming/parallel hints,
- additional engine-specific options.

The external driver must write the requested WAV path and exit with status 0. This isolates GPT-SoVITS-version-specific internals from ZynNova's public API.

---

## 5.5 Dataset preparation

```python
from pathlib import Path
from zynnova.zynvox.studio import (
    DatasetPrepareConfig,
    VoiceWorkspace,
    prepare_dataset,
)

manifest = prepare_dataset(
    DatasetPrepareConfig(
        dataset_name="speaker_a",
        input_audio=(Path("/data/raw/a.wav"), Path("/data/raw/b.wav")),
        language="zh",
        sample_rate=32000,
        min_segment_s=1.0,
        max_segment_s=15.0,
        silence_db=-42,
        transcribe=True,
        whisper_model="large-v3",
    ),
    VoiceWorkspace("/data/zynnova_workspace"),
)
```

Output:

```text
zynvox/datasets/speaker_a/
├── wavs/
└── manifest.csv   # audio, language, text
```

---

## 5.6 Training orchestration

ZynNova intentionally does not copy external training code into its package. Training is executed against an external engine checkout through explicit stage commands or the standard driver contract.

```python
from zynnova.zynvox.studio import TrainingConfig, train_voice_model

result = train_voice_model(
    TrainingConfig(
        dataset_manifest=manifest,
        run_name="speaker_a_v1",
        stages=("prepare-text", "ssl-features", "semantic", "acoustic"),
        batch_size=4,
        epochs_semantic=15,
        epochs_acoustic=8,
    ),
    profile,
    studio.workspace,
)
```

Each stage writes a log, the full job configuration is retained, and model outputs are located under the external `zynvox/models` directory.

---

## 5.7 ZynVox REST API

```bash
zynvox-studio --workspace /data/zynnova_workspace serve \
  --host 127.0.0.1 --port 8770
```

Endpoints:

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

The speech endpoint intentionally resembles common `/v1/audio/speech` clients, but it is a ZynNova-owned API and is independent of GPT-SoVITS WebUI's HTTP API.

Example:

```bash
curl http://127.0.0.1:8770/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "ZynVox Studio test",
    "voice": "my_voice",
    "language": "en",
    "temperature": 1.0,
    "stream": false
  }' \
  --output speech.wav
```

---

## 5.8 Gradio UI

```bash
zynvox-studio --workspace /data/zynnova_workspace ui \
  --host 127.0.0.1 --port 7861
```

The UI calls exactly the same `ZynVoxStudio` Python API as the REST server.

---

# 6. ZynVista — scene reconstruction and large-world generation

Existing ZynVista remains the authoritative metric reconstruction path. It already supports:

- image-conditioned reconstruction,
- video frame sampling,
- dense-view fusion,
- point-cloud and mesh construction,
- geometry audits,
- geometry-locked style transfer,
- cross-DCC export,
- COLMAP text export,
- world hierarchy / LOD chunk export,
- preservation of native backend assets including Gaussian-splat outputs when supplied.

0.3.0 adds `SceneStudio` and an external generator contract so large models can evolve independently from the stable scientific geometry layer.

## 6.1 Existing metric reconstruction

```python
from pathlib import Path
from zynnova.zynvista import SceneConfig, SceneRequest, run_scene

result = run_scene(
    SceneRequest(
        images=(Path("view_01.png"), Path("view_02.png")),
        mode="reconstruct",
        backend="auto",
    ),
    SceneConfig(
        build_mesh=True,
        build_world_hierarchy=True,
        export_formats=("ply", "glb", "usd"),
    ),
)
```

---

## 6.2 Modern external world generators

The 0.3.0 contract is designed to accommodate text/image/multiview/video-conditioned engines and mixed representations such as mesh + 3D Gaussian splats.

```python
from zynnova.zynvista import (
    CommandSceneEngine,
    GenerativeSceneRequest,
    SceneEngineProfile,
    SceneStudio,
)

studio = SceneStudio("/data/zynnova_workspace")

profile = SceneEngineProfile(
    name="hy-world-2",
    root="/data/external/HY-World-2",
    python="/data/envs/world/bin/python",
    module="zynnova_world_driver",
)

studio.register_engine("hy-world-2", CommandSceneEngine(profile))

bundle = studio.generate(
    GenerativeSceneRequest(
        prompt="a physically plausible battery research laboratory",
        model="my-world-checkpoint",
        geometry_mode="mesh+3dgs",
        texture_mode="pbr",
        seed=7,
    ),
    engine="hy-world-2",
)

print(bundle.assets)
```

A command engine receives `--zynnova-job job.json` and writes `result.json`:

```json
{
  "assets": {
    "mesh": "scene.glb",
    "gaussian_splat": "scene.spz",
    "point_cloud": "scene.ply",
    "cameras": "cameras.json",
    "depth": "depth.npz",
    "normals": "normals.npz"
  },
  "metadata": {
    "metric": true
  }
}
```

All referenced files are validated and copied into the ZynNova run export directory.

Good external targets for custom drivers include current large-world generators, Nerfstudio/3DGS reconstruction stacks, and other models that can return explicit geometry or native splat assets. Their repositories and checkpoints remain outside ZynNova.

---

## 6.3 Download scene checkpoints outside the package

```python
from zynnova.zynvista import download_scene_model

path = download_scene_model(
    "<hugging-face-scene-model-id>",
    "/data/zynnova_workspace",
)
```

---

# 7. ZynForm — high-fidelity object generation, repair and FEM

Existing ZynForm keeps:

- image-conditioned 3D object generation,
- native textured/PBR asset preservation,
- surface audit,
- surface cleanup/repair,
- physical scaling,
- TetGen/Gmsh/voxel tetrahedralization,
- FEM-ready Tet4 quality checks,
- DCC and FEM export.

0.3.0 adds `ObjectStudio` for modern image/multiview/text-to-3D engines.

```python
from pathlib import Path
from zynnova.zynform import (
    CommandObjectEngine,
    GenerativeObjectRequest,
    ObjectEngineProfile,
    ObjectStudio,
)

studio = ObjectStudio("/data/zynnova_workspace")

profile = ObjectEngineProfile(
    name="trellis2",
    root="/data/external/TRELLIS.2",
    python="/data/envs/3d/bin/python",
    module="zynnova_object_driver",
)

studio.register_engine("trellis2", CommandObjectEngine(profile))

bundle = studio.generate(
    GenerativeObjectRequest(
        images=(Path("object.png"),),
        texture_mode="pbr",
        topology_mode="production",
        target_extent_m=0.012,
    ),
    engine="trellis2",
    repair=True,
    generate_fem=True,
)
```

The external object driver must provide at least a `mesh` role in `result.json`; it may additionally provide `pbr`, `glb`, texture maps, multiview renderings, or other native assets.

After external generation, `ObjectStudio` can:

1. load the mesh into ZynNova geometry,
2. clean/repair it,
3. apply a physical metric scale,
4. preserve/scale the native PBR scene when possible,
5. tetrahedralize it with the existing ZynForm/ZynMorph FEM path,
6. retain a reproducible run manifest.

This makes modern generative models a front end to the existing scientific/FEM pipeline rather than a replacement for it.

Current external driver targets can include image-to-3D PBR systems such as TRELLIS.2 or Hunyuan3D-2.1; exact third-party CLIs are deliberately isolated in adapters because those projects change faster than ZynNova's public API.

---

# 8. ZynMorph and ZynSim

The 0.3.0 upgrade does not remove or replace the existing scientific modules.

## ZynMorph

Use it for:

- voxel microstructures,
- multiphase electrode structures,
- surface extraction and repair,
- free-form regions,
- tetrahedral meshing,
- native TetGen integration,
- region/material mappings,
- mesh export suitable for downstream FEM workflows.

## ZynSim

Use it for:

- battery/multiphysics simulations,
- FEM-related studies,
- scientific IO,
- image and geometry workflows,
- optional HPC backends.

ZynAstra can discover and invoke these APIs through the same tool layer as ZynVista, ZynForm, and ZynVox.

---

# 9. Agent + scientific workflow example

```python
import asyncio
from zynnova.llm.zynastra import Agent, ProviderConfig

async def main():
    agent = Agent.create(
        ProviderConfig.siliconflow("<model-id>"),
        "/data/zynnova_workspace",
    )
    try:
        result = await agent.run(
            """
            Inspect zynnova.zynmorph and zynnova.zynsim.
            Find the public APIs needed to create a multiphase electrode geometry,
            tetrahedralize it, export it, and prepare it for a simulation.
            Use tools to verify function signatures instead of guessing.
            """
        )
        print(result.text)
    finally:
        await agent.aclose()

asyncio.run(main())
```

The same agent can call `zynvista`, `zynform`, and `zynvox` functions when those roots remain in `allowed_zynnova_roots`.

---

# 10. CLI summary

```text
zynnova                 existing ZynNova CLI
zynnova-llm             ZynAstra chat / local-model download / fine-tune / server
zynvox-studio           ZynVox synthesis / API server / UI
```

Examples:

```bash
zynnova-llm --workspace /data/zynnova_workspace chat \
  "List the ZynVista generation APIs" \
  --provider openai-compatible \
  --model gpt-5.6 \
  --base-url https://api.openai.com/v1 \
  --api-style responses

zynnova-llm --workspace /data/zynnova_workspace \
  download-model <model-id>

zynvox-studio --workspace /data/zynnova_workspace \
  speak "test" --voice my_voice --language en
```

---

# 11. Extension contracts

## New LLM framework

Create a sibling package:

```text
src/zynnova/llm/my_new_framework/
```

Keep its providers, tools, memory, MCP and runtime internal unless you explicitly want interoperability. `zynnova.llm.__init__` may export selected stable entry points.

## New ZynAstra provider

Implement the `ModelProvider` protocol:

```python
class MyProvider:
    name = "mine"

    async def complete(self, messages, tools=()):
        ...

    async def aclose(self):
        ...
```

Then add construction logic to `providers/registry.py`.

## New Agent tool

```python
from zynnova.llm.zynastra import ToolRegistry

registry = ToolRegistry()
registry.add(
    "my_tool",
    lambda value: {"square": value * value},
    description="Square a number.",
    parameters={
        "type": "object",
        "properties": {"value": {"type": "number"}},
        "required": ["value"],
        "additionalProperties": False,
    },
)
```

## New voice engine

Implement:

```python
class MyVoiceEngine:
    name = "my-engine"
    def synthesize(self, request, profile, output): ...
    def convert(self, source, profile, output, **options): ...
```

or use `CommandVoiceEngine` and keep the engine-specific implementation entirely outside ZynNova.

## New scene/object generator

Use `CommandSceneEngine` / `CommandObjectEngine`, or implement their small `run(request, output_dir)` protocol directly.

---

# 12. Reproducibility and provenance

ZynNova's design rule is that difficult workflows should leave inspectable artifacts rather than only returning in-memory tensors.

Existing scientific and 3D pipelines already use run manifests and quality files. New external Studio contracts also retain:

- input request,
- selected engine/model,
- output assets,
- elapsed time,
- external engine log,
- job JSON,
- result JSON.

ZynVox continues to enforce authorization at the public voice boundary and the legacy pipeline continues to produce provenance/disclosure records.

API keys are not written into these manifests by ZynAstra.

---

# 13. Security boundaries

LLM tool execution is powerful. For production deployments:

- keep `allowed_zynnova_roots` minimal,
- expose only trusted MCP servers,
- do not place secrets in skill files,
- keep API keys in environment variables or a proper secret manager,
- run untrusted external model repositories in dedicated environments/containers,
- review model `trust_remote_code` requirements before enabling local inference,
- bind development FastAPI servers to localhost unless authentication/network policy is configured externally.

ZynAstra blocks private (`_...`) ZynNova attributes in the generic tool bridge, but this is an application boundary, not a general OS sandbox.

---

# 14. Voice authorization

ZynVox is intentionally consent-aware. The project supports legitimate voice conversion and speech synthesis where the target voice is:

- the user's own voice,
- directly authorized,
- licensed for this use, or
- valid public-domain/source material with evidence where required by the existing policy.

The authorization record is part of the API contract rather than a UI-only checkbox.

---

# 15. Third-party models and licenses

ZynNova does not automatically redistribute the checkpoints or repositories of GPT-SoVITS, HY-World, TRELLIS, Hunyuan3D, Nerfstudio, or other optional external projects. Install/download them separately into an external workspace and follow each project's license and model terms.

The existing ZynNova license metadata also notes the separate license requirements of linked TetGen components. Keep third-party notices and source-license obligations intact when distributing binaries.

---

# 16. Development checks

```bash
python -m compileall src/zynnova
pytest -q
ruff check src tests
```

For editable native builds:

```bash
pip install -v -e .
```

For the full heavy subsystem you are actively developing, install only its needed extras rather than every optional dependency at once.

---

# 17. 0.3.0 upgrade summary

### Added

- `src/zynnova/llm/`
- complete `src/zynnova/llm/zynastra/` agent framework
- OpenAI Responses + generic OpenAI-compatible provider
- LiteLLM provider
- local Transformers provider
- ZynNova-wide typed tool bridge
- skills
- MCP client bridge
- SQLite session memory
- external-workspace model download
- LoRA / QLoRA SFT
- ZynAstra FastAPI + CLI
- `src/zynnova/zynvox/studio/`
- ZynVox first-party REST API
- voice enrollment/model listing
- TTS/VC Studio API
- dataset preprocessing + optional ASR
- staged external-engine training orchestration
- Gradio UI
- ZynVista external model hub / engine contract / SceneStudio
- ZynForm external model hub / engine contract / ObjectStudio
- modern world/object generation → existing geometry/FEM bridge

### Modified

- package-level `zynnova.__init__`
- `zynvox.__init__`
- `zynvista.__init__`
- `zynform.__init__`
- full `pyproject.toml`
- full `README.md`

The scientific core and existing public pipelines remain available; the 0.3.0 work adds higher-level capabilities rather than replacing them.
