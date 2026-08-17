# PolyLoom

`PolyLoom` is the next-generation companion to `PolyGen`. It retains the
Polymer-SELFIES representation, ZynNova dataset interface, target normalizers,
chemical validation and two-port repeat-unit contract.

Key differences:

- cosine-scheduled discrete flow with log-SNR time encoding;
- self-conditioning to reduce trajectory drift during iterative unmasking;
- classifier-free multi-property and processing-condition guidance;
- condition-routed experts for different regions of the design space;
- auxiliary property, length and polymer-endpoint objectives;
- oversampling followed by chemical validation and ranked candidate return.

The public API follows the same lifecycle as `PolyGen`:

```python
from zynnova.ml.generation.PolyLoom import (
    PolyLoomConfig,
    generate_poly_loom,
    load_poly_loom,
    train_poly_loom,
)

config = PolyLoomConfig()
result = train_poly_loom(config, samples=my_material_samples)
generator = load_poly_loom(result.best_checkpoint, device="auto")
generation = generate_poly_loom(
    generator,
    {"dielectric_constant": 6.0, "configurational_entropy_R": 1.5},
    config=config.sampling,
)
```

The sampler accepts normalized target properties and processing conditions so
it can reuse the normalizers stored in each training checkpoint.
