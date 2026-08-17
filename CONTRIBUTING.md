# Contributing to ZynNova

Thank you for your interest in contributing to **ZynNova**.

ZynNova is a unified framework for materials representation, machine learning, and simulation. Contributions of all sizes are welcome, including bug fixes, documentation improvements, new datasets, neural-network models, simulation backends, and C++ performance optimizations.

Please read this guide before opening an issue or pull request.

## Code of Conduct

By participating in this project, you agree to follow the project’s [Code of Conduct](CONDUCT.md).

Please communicate respectfully and keep technical discussions constructive and focused.

## Ways to Contribute

You can contribute to ZynNova in several ways:

- Report reproducible bugs.
- Fix existing issues.
- Improve documentation and tutorials.
- Add tests and validation cases.
- Implement new materials representations.
- Add datasets, encoders, or preprocessing methods.
- Add neural-network architectures.
- Add prediction, generation, or force-field models.
- Add molecular-dynamics calculators or backends.
- Improve Python or C++ performance.
- Improve interoperability with external scientific packages.

Before beginning a substantial change, please open an issue to discuss its scope and intended design.

## Reporting Bugs

Bug reports should be submitted through the issue tracker of the repository that hosts this source tree.

A useful bug report should include:

- A clear and descriptive title.
- The ZynNova version or Git commit.
- Python version.
- Operating system and version.
- Installation method.
- Relevant optional dependencies and their versions.
- Minimal code that reproduces the problem.
- Complete exception and traceback.
- Expected behavior.
- Actual behavior.

When applicable, also include:

- Structure format and system size.
- Input tensor shapes and data types.
- Units used by the input data.
- CPU or GPU information.
- CUDA and PyTorch versions.
- Whether the native C++ backend is available.

Please remove private datasets, credentials, access tokens, and confidential paths before submitting logs.

## Requesting Features

Feature requests should explain:

- The scientific or engineering problem being addressed.
- The proposed public API.
- Expected inputs and outputs.
- Required units, shapes, and data types.
- Relevant references or established methods.
- Required optional dependencies.
- Compatibility with existing ZynNova modules.
- A minimal example of the intended usage.

Keep each request focused on one feature. Large proposals should be divided into smaller, independently reviewable components.

## Development Requirements

Local development requires:

- Python 3.10 or later
- Git
- A C++17-compatible compiler
- CMake
- A supported Python build environment

Optional machine-learning functionality may additionally require PyTorch, PyTorch Geometric, RDKit, ASE, or other scientific packages.

## Development Setup

### 1. Clone the repository

Clone the repository URL supplied by the project owner or your own fork:

```bash
git clone <repository-url> ZynNova
cd ZynNova
```

When a canonical upstream remote is available, add it explicitly:

```bash
git remote add upstream <upstream-repository-url>
git remote -v
```

### 2. Create a development environment

Using Conda:

```bash
conda create -n zynnova-dev python=3.12
conda activate zynnova-dev
```

Alternatively, using Python `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the package

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip
```

Install ZynNova in editable development mode:

```bash
python -m pip install -e ".[dev]"
```

This builds the native C++ extension and links the installed package to the local source tree.

To install all optional components:

```bash
python -m pip install -e ".[dev,all]"
```

You may also install only the components required for your work:

```bash
python -m pip install -e ".[dev,graph,dynamics]"
```

### 4. Verify the installation

Check the Python package:

```bash
python -c "import zynnova; print(zynnova.__version__)"
```

Check the native C++ structure backend:

```bash
python -c "from zynnova.structure.common import native_available; print(native_available())"
```

A result of `True` indicates that the native extension was built and imported successfully.

## Project Structure

The main development directories are:

```text
ZynNova/
├── src/zynnova/          # Python package
├── cpp/                   # C++ source, headers, and pybind11 bindings
├── tests/                 # Automated tests
├── docs/                  # Documentation and tutorials
├── CMakeLists.txt         # Native build configuration
└── pyproject.toml         # Python package and build configuration
```

New files should be placed in the appropriate existing module. Avoid creating parallel implementations outside the established package structure.

## Development Workflow

Create a focused branch for your contribution:

```bash
git switch -c feature/short-description
```

Examples of useful branch names:

```text
feature/equivariant-potential
feature/new-polymer-encoder
fix/periodic-neighbor-search
docs/dynamics-tutorial
perf/cpp-graph-builder
```

Keep your branch synchronized with the upstream repository:

```bash
git fetch upstream
git rebase upstream/main
```

Make small, logically related commits. Avoid including unrelated formatting changes, generated files, build directories, caches, checkpoints, or local datasets.

## Coding Guidelines

### General requirements

Contributed code should:

- Be clear, maintainable, and appropriately documented.
- Preserve existing functionality unless a breaking change has been approved.
- Use explicit validation for public inputs.
- Provide useful and actionable error messages.
- Clearly document units, tensor shapes, data types, and conventions.
- Avoid unnecessary dependencies.
- Keep optional dependencies isolated from the core package.
- Include tests for normal behavior and important edge cases.
- Avoid unrelated refactoring in the same pull request.

### Python code

Python contributions should:

- Follow the existing package organization.
- Use type annotations for public interfaces.
- Include docstrings for public classes, methods, and functions.
- Import optional dependencies only where required.
- Raise a clear installation message when an optional dependency is missing.
- Expose supported public APIs through the appropriate package namespace.
- Keep implementation-only helpers private.
- Avoid embedding machine-specific paths or configuration.

Public APIs should clearly define:

- Accepted input types.
- Input and output shapes.
- Physical units.
- Default values.
- Possible exceptions.
- Optional dependencies.
- Numerical precision expectations.

### C++ code

C++ contributions should:

- Use C++17-compatible code.
- Separate public headers, implementation files, and Python bindings.
- Validate array dimensions, data types, indices, and memory layout.
- Avoid undefined behavior and unsafe ownership.
- Use const references where appropriate.
- Release the Python GIL for sufficiently expensive native computations when safe.
- Translate C++ errors into useful Python exceptions.
- Provide Python and C++ consistency tests.
- Include benchmarks when the contribution is performance-related.

After changing C++ code or pybind11 bindings, rebuild the editable installation:

```bash
python -m pip install -e .
```

Run the native backend tests:

```bash
python -m pytest tests/test_cpp_backend.py
```

### Machine-learning models

A new neural-network model should include:

- A configuration object or clearly defined constructor.
- Explicit input and output contracts.
- Support for CPU execution.
- A minimal deterministic smoke test.
- Reproducible random-seed handling.
- Training and inference interfaces.
- Checkpoint saving and loading.
- Device and data-type handling.
- Clear optional-dependency boundaries.
- Documentation and a public example.
- Registration through the appropriate model registry, when applicable.

Models used as interatomic potentials must additionally document:

- Predicted energy conventions.
- Force calculation and gradient behavior.
- Stress or virial conventions, if supported.
- Atomic and structure-level output shapes.
- Length, energy, force, and stress units.
- Periodic-boundary handling.
- Neighbor-list requirements.
- Differentiability expectations.
- Numerical precision and stability assumptions.

### New datasets and encoders

Dataset contributions should document:

- Data source and license.
- Download procedure.
- Expected files and checksums, when applicable.
- Structure representation.
- Field names, roles, shapes, and units.
- Missing-value behavior.
- Train, validation, and test splitting.
- Reproducibility requirements.
- Required optional dependencies.

Do not commit restricted or redistributable datasets without explicit permission.

## Testing

Run the complete test suite before submitting a pull request:

```bash
python -m pytest
```

Run a specific test module:

```bash
python -m pytest tests/test_structure_types.py
```

Run an individual test:

```bash
python -m pytest tests/test_structure_types.py::test_name
```

New functionality should include tests that cover:

- Expected behavior.
- Invalid inputs.
- Boundary conditions.
- Shape and unit validation.
- Serialization or checkpoint recovery, when applicable.
- CPU behavior.
- Python and C++ consistency, when applicable.
- Periodic structures, when applicable.

Tests should be small and should not download large datasets or train expensive models.

## Code Quality

Run Ruff before submitting your changes:

```bash
ruff check .
```

Check formatting:

```bash
ruff format --check .
```

Apply automatic formatting when needed:

```bash
ruff format .
```

Automatically fix safe linting issues:

```bash
ruff check . --fix
```

Always review automatically generated changes before committing them.

## Documentation

Changes to public behavior must include corresponding documentation.

Documentation contributions may include:

- Public API docstrings.
- Installation instructions.
- Workflow tutorials.
- API notebooks.
- Architecture documentation.
- Examples for new models or simulation methods.

Documentation files are located under:

```text
docs/
```

End-to-end tutorials are located under:

```text
docs/notebooks/workflows/
```

Public API notebooks are located under:

```text
docs/notebooks/api/
```

Public examples should use supported package imports and should not depend on private implementation helpers.

Expensive training, downloads, and simulations should be disabled by default using an explicit flag such as:

```python
RUN_TRAINING = False
```

## Building the Package

Build source and wheel distributions with:

```bash
python -m build
```

The build should complete without modifying tracked source files.

Do not commit local build artifacts such as:

```text
build/
dist/
*.egg-info/
__pycache__/
.pytest_cache/
.ruff_cache/
```

## Pull Request Guidelines

Before opening a pull request, confirm that:

- The branch contains one focused change.
- Existing functionality remains compatible.
- New functionality includes appropriate tests.
- The complete test suite passes.
- Ruff checks pass.
- Public APIs include docstrings and type information.
- Units, shapes, and conventions are documented.
- Documentation and notebooks are updated.
- Optional dependencies are handled correctly.
- No large datasets, checkpoints, credentials, or generated artifacts are included.
- Performance claims are supported by reproducible benchmarks.
- Breaking changes are clearly identified.

The pull request description should include:

1. A concise summary.
2. The problem being solved.
3. The implementation approach.
4. Public API changes.
5. Tests performed.
6. Documentation changes.
7. Performance or accuracy results, when relevant.
8. Known limitations or follow-up work.

## Reviewing Contributions

Maintainers may request changes related to:

- API consistency.
- Numerical correctness.
- Physical conventions.
- Performance.
- Test coverage.
- Documentation.
- Dependency scope.
- Backward compatibility.
- Long-term maintainability.

Review feedback is part of the collaborative development process. Please keep discussions technical, specific, and respectful.

## Licensing

By submitting a contribution to ZynNova, you agree that your contribution may be distributed under the terms of the project’s [MIT License](LICENSE).