# External backend contracts

All contracts use argv lists and do not invoke a shell. Placeholder substitution is
performed per argv token, so paths containing spaces remain safe.

## Battery microstructure (`zynnova.zynmorph.external.v1`)

Construct `ExternalMicrostructureBackend(name, command, cwd=...)`. ZynNova appends:

```text
--request REQUEST.json --output OUTPUT.npz
```

The process must write an NPZ containing a 3-D integer array named `labels`. The label
shape and phase IDs are validated by the normal pipeline; exact fractions and requested
percolation constraints are enforced after generation.

## Scene (`zynnova.scene-request.v1`)

`ExternalSceneBackend` accepts an argv template with `{request}`, `{output}`, and
`{work}`. `output_files` maps semantic roles to paths relative to `{output}`. Typical
roles are `mesh`, `point_cloud`, `gaussian_splat`, `cameras`, and `texture`.

## Object (`zynnova.object-request.v1`)

`ExternalObjectBackend` supports `{request}`, `{image}`, `{output}`, and `{work}`.
`output_mesh` is relative to `{output}` and must name an existing mesh after the process
exits. The standard repair, export, scale, and FEM stages then run normally.

## Voice conversion

`ExternalVoiceBackend` supports `{source}`, `{target}`, `{output}`, `{workdir}`, and
`{mode}`. It must create the WAV path represented by `{output}`.

## Text-to-speech (`zynnova.tts-request/1.0`)

`ExternalTTSBackend` supports `{request}`, `{reference}`, `{output}`, and `{workdir}`.
The request JSON contains text, language, transcript, emotion controls, duration factor,
style instruction and streaming flag. The program must create `{output}` as readable
audio.

## Reliability requirements

Wrappers should return non-zero on failure, write outputs atomically, avoid modifying
input files, report model/version in metadata, and pin their own repository revision and
weights. Never register a wrapper whose output contract has not been exercised by an
integration test.
