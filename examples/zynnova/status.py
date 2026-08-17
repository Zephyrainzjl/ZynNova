from __future__ import annotations

import json

from zynnova import backend_status

print(json.dumps(backend_status(), indent=2, ensure_ascii=False))
