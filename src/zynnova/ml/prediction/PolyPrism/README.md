# PolyPrism

`PolyPrism` is the next-generation companion to `PolyPrediction`. It keeps the
same `MaterialSample`, property specification, normalization and family-split
contracts, while replacing early pooled fusion with property-query fusion.

Key differences:

- PSMILES tokens, periodic graph, process conditions and source fidelity remain
  distinct views until property-specific cross-attention.
- Sparse top-k experts specialize for thermal, electronic, dielectric,
  breakdown and morphology targets.
- Normal-Inverse-Gamma evidence separates aleatoric and epistemic uncertainty.
- Simulation, literature and experimental values can be trained jointly through
  the `sample.metadata["fidelity"]` field.
- Existing energy-storage targets and physics-consistency loss remain available.

The public API follows the same lifecycle as `PolyPrediction`:

```python
from zynnova.ml.prediction.PolyPrism import (
    PolyPrismConfig,
    load_poly_prism,
    predict_poly_prism,
    train_poly_prism,
)

config = PolyPrismConfig()
result = train_poly_prism(config, samples=my_material_samples)
predictor = load_poly_prism(result.best_checkpoint, device="auto")
predictions = predict_poly_prism(predictor, candidate_psmiles)
```

Set fidelity with one of `unknown`, `simulation`, `literature`, or `experiment`.
Aliases such as `dft`, `computed`, `paper`, and `measured` are normalized.
