# Public-source and literature audit

Observed on **2026-08-17**. `SOURCE_LOCK.json` is machine-readable and intentionally
leaves revisions unset until installation; `scripts/zynnova/source_bootstrap.py` records
the exact checked-out commit. No third-party model source or weight is copied into this
repository.

## Selection criteria

A backend is included only when an author/institution repository or primary paper is
available, its input/output can be mapped to a stable contract, and missing weights or
licenses are reported instead of bypassed. Blog posts, unofficial repacks and anonymous
checkpoints are not accepted as implementation sources.

## Battery microstructures

The design starts from multiphase periodic reconstruction demonstrated by *Pores for
Thought* and upgrades it with exact phase counts, arbitrary phase schemas, descriptor
conditioning, enforced percolation, physical voxel spacing, quality-checked Tet4 meshes,
and a trainable conditional 3-D flow. The optional DSD contract targets intensity/mass
preserving discrete diffusion. Because the LANL repository does not expose a stable
package API and has a nonstandard notice, it remains isolated rather than being copied.
MicroLad is listed only as a primary-paper design reference: no author-maintained source
repository was verified during this audit, so ZynNova does not pretend to provide a
MicroLad code backend.

## 3-D scenes

MapAnything is used for feed-forward metric reconstruction because its official model
returns factored scene geometry/cameras across multiple reconstruction tasks. HY-World
2.0 is kept as a separate large-world pipeline because it produces persistent meshes and
Gaussian splats from text, images, or video through multiple official stages. Text-guided
3DGS editing is integrated through the official Instruct-GS2GS commands.

## 3-D objects

Pixal3D is the preferred pixel-aligned single-image path and TRELLIS.2 is the alternative
for complex topology and PBR output. The adapters preserve native GLB/PBR assets before
surface conversion, repair and volume meshing. A CPU silhouette baseline exists only for
deterministic tests and is explicitly named as a baseline.

## Voice

MeanVC2 and X-VC are true speech-to-speech conversion engines. CosyVoice 3 and
IndexTTS-2.5 cover multilingual zero-shot cloning and controllable synthesis. The official
GPT-SoVITS `api_v2.py` client is included solely as a same-machine comparison backend.
ZynNova does not claim perceptual superiority from architecture names: the supplied
benchmark must demonstrate it with identical references, texts, hardware and evaluators.

## License boundary

Code and checkpoints can have different terms. MapAnything, for example, publishes
both a noncommercial and an Apache-licensed model option. IndexTTS uses a custom model
use agreement. Every external backend therefore requires an explicit local installation;
ZynNova redistributes neither repositories nor weights and records the selected paths and
hashes in manifests.
