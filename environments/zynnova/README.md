# ZynNova environments

Use `core.yml` for portable geometry, meshing, audio I/O and tests. Learned backends are
intentionally not merged into one environment: their CUDA, PyTorch, compiler and model
requirements conflict.

Recommended layout:

```text
external/zynnova/
├── mapanything/
├── hy-world-2/
├── instruct-gs2gs/
├── pixal3d/
├── trellis2/
├── meanvc2/
├── xvc/
├── cosyvoice/
├── indextts/
└── gpt-sovits/
```

Create each backend environment by following the official repository instructions after
pinning a commit with `scripts/zynnova/source_bootstrap.py`. Do not blindly install from a
third-party tutorial or an unversioned repack. Record the Python executable, repository,
checkpoint/model path and accepted license in the workflow configuration.

MapAnything offers differently licensed checkpoints; select the Apache model when the
use case requires that licensing. HY-World and IndexTTS have custom upstream terms.
Pixal3D/TRELLIS.2 also have separately licensed renderer dependencies. Verify all terms
before deployment.
