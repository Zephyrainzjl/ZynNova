# Attribution

This ZynNova extension provides an optional compatibility bridge for:

- AutoFigure: https://github.com/ResearAI/AutoFigure
- AutoFigure-Edit: https://github.com/ResearAI/AutoFigure-Edit

Both upstream repositories identify their source code as MIT licensed.  Their
names, marks, papers, datasets and upstream implementations remain the property
of their respective authors.  No upstream source file is vendored in this
patch.  The bridge imports the separately installed `autofigure` Python package
and forwards original API calls without changing their arguments.
