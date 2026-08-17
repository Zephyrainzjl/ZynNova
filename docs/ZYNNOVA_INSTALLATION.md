# ZynNova installation and backend isolation

## Portable core

Python 3.10–3.13 is supported by the ZynNova package. For the most compatible external
3-D/audio stacks, use Python 3.10 or 3.11 in their own environments.

```bash
python -m pip install -e ".[zynnova]"
```

Optional groups are `zynnova-geometry`, `zynnova-fem`, `zynnova-morph`,
`zynnova-morph-train`, `zynnova-scene`, `zynnova-object`, `zynnova-voice`,
`zynnova-voice-eval`, and `zynnova-ui`.

## External repositories

Do not install every research repository into the ZynNova environment. Clone and pin it:

```bash
python scripts/zynnova/source_bootstrap.py mapanything \
  --destination external/zynnova/mapanything \
  --accept-upstream-terms
```

Then follow that repository's official installation instructions in a dedicated Conda
environment. Pass `repository`, `python_executable`, checkpoint/model directory and any
required explicit license flag through `backend_options`.

## Diagnostics

```bash
zynnova status
python scripts/zynnova/verify.py --output validation/zynnova_validation.json
```

An unavailable backend is normal until its repository and weights are configured. The
status report states the missing requirement rather than importing the heavy stack.
