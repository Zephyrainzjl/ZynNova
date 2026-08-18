# ZynNova TetGen material-region recovery final fix

## What this fixes

The old free-form TetGen path treated `raw["region_attributes"]` as physical
material IDs and immediately required:

```python
set(np.unique(raw_region_attributes)) == configured_material_ids
```

That is the exact source of:

```text
requested=[1, 2, 3], output=[1, 2, 3, 4, 5, 6, 7, 8]
```

The corrected implementation separates:

- **TetGen raw/topological region attributes**
- **ZynNova physical material region IDs**

If TetGen returns extra topology IDs, ZynNova uses the marked PLC interface
graph to recover the physical material on each side.

For example:

```text
raw TetGen IDs:
1, 2, 3, 4, 5, 6, 7, 8

physical materials:
1 = active
2 = electrolyte
3 = CBD
```

may become:

```text
raw -> material
1 -> 1
2 -> 2
3 -> 3
4 -> 1
5 -> 3
6 -> 2
7 -> 1
8 -> 3
```

The exact mapping is derived from PLC facet markers and adjacency; it is NOT
derived from the magnitude of raw IDs.

The final FEM `cell_regions` and COMSOL domains therefore contain only the
declared physical materials.

## Important: your traceback proves you are loading the old file

Your current traceback still contains:

```python
output_ids = set(map(int, np.unique(cell_regions)))
if output_ids != configured_ids:
    raise GeometryError(
        "TetGen free-form region coverage mismatch: "
        f"requested={sorted(configured_ids)}, output={sorted(output_ids)}"
    )
```

The fixed source no longer performs that raw-output gate. It calls:

```python
recover_tetgen_material_regions(...)
```

first.

## Apply

Extract this ZIP over the ZynNova repository root.

Then completely restart the Jupyter kernel.

Because this patch is Python-only, the TetGen C++ ABI does not need another
rebuild if ABI 2 is already installed. Re-running editable install is safe:

```powershell
python -m pip install -e ".[zynmorph-all]" -v
```

## Verify BEFORE opening the notebook

From a fresh PowerShell:

```powershell
python scripts\verify_tetgen_region_recovery.py
```

Expected:

```text
"region_recovery_api": "facet-material-v1"
"has_material_region_recovery": true
"has_old_raw_coverage_gate": false

PASS: TetGen raw topology-region recovery is active.
```

Also inspect the path printed for `freeform.py`. It must be the repository you
just patched.

## Run regression

```powershell
pytest -q tests\zynnova\test_tetgen_region_recovery.py
```

Expected:

```text
6 passed
```

The complete artifact source regression is:

```text
132 passed
```

## Notebook

Use only:

```text
notebooks/ZynNova_MCS_MCR_Heterogeneous_TetGen_RegionRecovery_v2.ipynb
```

The notebook has a source gate in its first cells. If the kernel still loads
the old `freeform.py`, it stops immediately before spending time on MCS/MCR or
TetGen.

After TetGen it prints:

```python
fem.metadata["tetgen_region_recovery"]
```

including:

- requested material IDs
- raw TetGen region IDs
- automatic/extra raw IDs
- raw -> physical material mapping
- cell counts before/after recovery

The final assertion remains strict:

```python
set(np.unique(fem.mesh.cell_regions)) == {1, 2, 3}
```

Do not weaken or delete this assertion.
