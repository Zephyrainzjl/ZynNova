#!/usr/bin/env python
"""Print detailed ZynMorph TetGen native-extension diagnostics."""
from __future__ import annotations

import json

from zynnova.zynmorph import tetgen_native_diagnostics


def main() -> int:
    report = dict(tetgen_native_diagnostics())
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
