# ZynNova validation record

Validation date: **2026-08-17**.

The checked source tree is validated in two deliberately separate layers.

## Deterministic local validation

Run:

```bash
PYTHONPATH=src python scripts/zynnova/verify.py \
  --output validation/zynnova_validation_2026-08-17.json
```

The recorded run reports:

A separate reproducible static source audit (`python scripts/zynnova/static_audit.py --output validation/zynnova_static_audit_2026-08-17.json`) is recorded in `validation/zynnova_static_audit_2026-08-17.json`. It parsed 108 Python files (11,951 lines), the project TOML and the source lock; checked for syntax errors, `shell=True`, `os.system`, `subprocess.Popen`, built-in `eval`/`exec`, unsafe pickle loads, user-specific absolute paths, duplicate source IDs and unregistered optional-dependency names; all audited checks passed.

- package byte-code compilation: passed;
- ZynNova pytest suite: **13 passed**;
- backend status CLI: passed;
- examples and verification scripts compilation: passed;
- overall deterministic result: passed.

The tests exercise backend discovery and failure reasons, nanometre-scale Tet4 quality,
safe volume round-trips, exact multiphase fractions, hard percolation constraints, scene
fusion and exports, object surface/FEM generation, voice consent/provenance and audio
benchmarking. The tests use small deterministic fixtures; they do not present a fallback
as a learned-model result.

## Standalone namespace audit

`validation/zynnova_rebrand_audit_2026-08-17.json` verifies that every retained
source-snapshot file maps to the standalone repository, the two package initializers
were deliberately merged, no obsolete project identifier remains in a path or file,
there is no duplicated package namespace, 358 notebooks remain valid JSON, the CMake
configuration succeeds, and six portable C++ translation units pass syntax checks.

## External-model contract validation

MapAnything, HY-World 2.0, Instruct-GS2GS, Pixal3D, TRELLIS.2, MeanVC2, X-VC,
CosyVoice 3, IndexTTS-2.5 and GPT-SoVITS remain isolated upstream installations. The
local validation checks their typed request/output contracts, path/command construction,
license gates, diagnostics and failure manifests. It does **not** claim that large-model
inference was executed without the corresponding repositories, checkpoints, CUDA stack
and accepted upstream terms.

For a production deployment, pin each checked-out commit with
`scripts/zynnova/source_bootstrap.py`, run the upstream model's own tests, then repeat the
ZynNova pipeline and same-hardware quality benchmark on representative private data.

## Retained test scope

The retained source snapshot contains pre-existing tests and optional model imports outside the difficult-task subframeworks. Its original global test fixture imported a missing `zynforge` module during
collection; this was changed to a lazy fixture so independent test groups can run. A
broader exploratory run also exposed one pre-existing SCF-gate expectation outside the
new modules. It is therefore reported separately from the validated ZynNova difficult-task suite; this delivery does not claim that every historical optional-model test passes when its corresponding source module is absent.

The machine-readable evidence is
`validation/zynnova_validation_2026-08-17.json`.
