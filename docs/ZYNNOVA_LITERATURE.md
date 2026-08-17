# ZynNova primary literature and official implementations

The machine-readable repository list is `src/zynnova/SOURCE_LOCK.json`.
Engineering choices are based on primary papers and author/institution repositories,
including:

- Gayon-Lombardo et al., *Pores for thought*, npj Computational Materials (2020), with
  the authors' `pores4thought` repository.
- Santos et al., *Discrete Spatial Diffusion: Intensity-Preserving Diffusion Modeling*,
  NeurIPS 2025 Spotlight, with the LANL repository.
- Lee and Ahmed, *MicroLad: 2D-to-3D Microstructure Reconstruction and Generation via Latent Diffusion and Score Distillation*; retained as a paper reference because no author-maintained source repository was verified on 2026-08-17.
- Keetha et al., *MapAnything: Universal Feed-Forward Metric 3D Reconstruction*, with
  the Meta research repository.
- Tencent Hunyuan, *HY-World 2.0*, official world-generation/reconstruction repository.
- Li et al., *Pixal3D: Pixel-Aligned 3D Generation from Images*, SIGGRAPH 2026.
- Microsoft Research, *TRELLIS.2: Native and Compact Structured Latents for 3D
  Generation*.
- Vachha and Haque, *Instruct-GS2GS*, official Nerfstudio extension.
- Liu et al., *StyleGaussian*, and Liu et al., *StylOS*, retained as isolated 3DGS style backends through the external style contract.
- Ma et al., *MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion*.
- *X-VC: Zero-shot Streaming Voice Conversion in Codec Space*.
- Du et al., *CosyVoice 3: Towards In-the-wild Speech Generation*.
- *IndexTTS-2.5 Technical Report* and the official IndexTTS repository.
- The official GPT-SoVITS repository and `api_v2.py`, used as a comparison baseline.

Repository state changes quickly. Pin a commit and preserve the upstream license and
model-card files with every production deployment.
