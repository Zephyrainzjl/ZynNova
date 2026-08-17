# ZynNova polymer integration

Integrated package path:

```text
src/zynnova/structure/polymer/
```

Public namespace:

```python
from zynnova.structure import polymer
from zynnova.structure.polymer import stru2record, make_view, view2record
```

Conversion pairs:

```text
stru2record <-> record2stru
stru2simple <-> simple2stru
stru2graph  <-> graph2stru
record      <-> chemical/single-chain/multiscale/transformer/generative views
```

The default source-to-view conversion retains a lossless record payload for an
exact inverse. Set `include_reconstruction=False` for compact training storage.
Generated unit-level graphs decode with a `RepresentationSchema` and unit
library; generated atom-level graphs decode directly from atom, bond, and
coordinate classes.

Validation performed:

- Python source compilation completed.
- Full pytest suite: 12 passed, 2 skipped because optional PyG/native test
  discovery dependencies were not active in the source-test environment.
- Binary wheel build completed after correcting the existing CMake binding path.
- Installed wheel import and native C++ backend smoke test completed.