from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import zynnova
import zynnova.zynmorph.freeform as freeform
import zynnova.zynmorph.region_recovery as recovery
from zynnova.zynmorph import TETGEN_REGION_RECOVERY_API, tetgen_native_status

source = inspect.getsource(freeform.mesh_freeform_tetgen)
result = {
    "python": sys.executable,
    "zynnova": str(Path(zynnova.__file__).resolve()),
    "freeform": str(Path(freeform.__file__).resolve()),
    "region_recovery": str(Path(recovery.__file__).resolve()),
    "region_recovery_api": TETGEN_REGION_RECOVERY_API,
    "has_material_region_recovery": "recover_tetgen_material_regions" in source,
    "has_old_raw_coverage_gate": (
        "TetGen free-form region coverage mismatch" in source
        and "recover_tetgen_material_regions" not in source
    ),
}
status = tetgen_native_status()
result["tetgen_native"] = {
    "available": status.available,
    "version": status.version,
    "reason": status.reason,
    "module_path": None if status.module_path is None else str(status.module_path),
}
print(json.dumps(result, indent=2, ensure_ascii=False))
if not result["has_material_region_recovery"]:
    raise SystemExit(
        "FAIL: 当前 Python/Jupyter 仍在加载旧版 zynmorph/freeform.py。"
        "覆盖补丁并完全重启 kernel 后再试。"
    )
print("PASS: TetGen raw topology-region recovery is active.")
