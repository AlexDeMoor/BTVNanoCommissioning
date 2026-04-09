# CLAUDE.md

## Project Overview

CMS BTV (b-tag validation) commissioning framework built on **coffea 0.7.30** for analyzing PFNano/NanoAOD data. Contains 40+ physics workflows (ttbar, c-tag, QCD, BTA, etc.) registered in `src/BTVNanoCommissioning/workflows/__init__.py`.

## Tech Stack

- **Language**: Python 3.8–3.10
- **Core framework**: coffea 0.7.30 (processor class pattern — `process()` / `postprocess()`)
- **Key libraries**: awkward, uproot, vector, hist, correctionlib, dask, parsl
- **Package manager**: pip + setuptools (conda/micromamba for environment via `test_env.yml`)

## Repository Layout

```
runner.py                          # Main entry point for all workflows
src/BTVNanoCommissioning/
  workflows/                       # Processor classes (one per physics channel)
  helpers/                         # Physics helpers (definitions, cross-sections, scale factors)
  utils/
    selection.py                   # Event/object selection functions
    correction.py                  # Scale factor & correction loading
    histogramming/                 # Histogram axes, templates, and filling
    array_writer.py                # ROOT array I/O
    AK4_parameters.py              # Jet parameter definitions
  data/                            # Correction files and lookup tables
scripts/                           # Plotting, hadding, validation scripts
metadata/                          # Dataset JSON configs (per campaign/year)
condor/ & condor_lxplus/           # Batch job submission
docs/                              # Sphinx documentation
testfile/                          # Test coffea output files
```

## Commands

```bash
# Install (editable)
pip install -e .
pip install -e .[dev,doc]          # with dev/doc extras

# Lint (matches CI)
black --check --verbose ./
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --ignore=E50,F401,F403,F824

# Format
black ./

# Run a workflow (integration test pattern)
python runner.py --workflow ttdilep_sf --json metadata/test_bta_run3.json --limit 1 --executor iterative --year 2023 --campaign Summer23

# Build docs
cd docs && make html
```

## Coding Conventions

- **Formatter**: Black (default settings, line length 88)
- **Linter**: Flake8 (config in `setup.cfg` — ignores E203, E231, E501, E722, W503, B950)
- **No type annotations** — this is physics analysis code; keep it concise
- **Import style**: Match the existing pattern in each file (no strict ordering enforced)
- **Coffea 0.7 patterns only**: Use `BaseProcessor` subclass with `process()`/`postprocess()` methods and `accumulate()`. Do not use coffea 2024+ dask-native patterns.
- **Array libraries**: Use awkward-array or numpy as appropriate for the task — no strict preference
- **No unit tests**: The project uses workflow-based integration tests via `runner.py` with `--limit 1`

## CI

GitHub Actions workflows (`.github/workflows/`):
- `python_linting.yml` — Black + Flake8 on every PR/push to master
- `ttbar_workflow.yml`, `ctag_*.yml`, `BTA_workflow.yml`, `QCD_workflow.yml` — integration tests per physics channel
- CI skip flags in commit messages: `[skip ci]`, `ci:skip array`, `ci:skip syst`, `ci:JERC_split`, `ci:weight_only`

## Key Patterns

- **Workflow registration**: Add new workflows to the dict in `src/BTVNanoCommissioning/workflows/__init__.py`
- **Selections**: Reusable selection functions live in `utils/selection.py` (HLT triggers, jet ID, muon ID/iso, etc.)
- **Corrections**: Scale factors and corrections loaded via `utils/correction.py` using correctionlib JSONs
- **Histogramming**: Axis definitions in `utils/histogramming/axes/`, histogram templates in `utils/histogramming/histograms/`
- **Runner flags**: `--isSyst all|weight_only|JERC_split` for systematics, `--isArray` for awkward array output, `--noHist` to skip histograms
