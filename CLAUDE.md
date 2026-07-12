# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **research fork of vLLM (based on v0.9.1)** that adds an **SLA-aware / ELRAR scheduling** system on top of the upstream V1 engine. The upstream vLLM codebase is unchanged in spirit; almost all original work lives in a small set of modules under `vllm/v1/core/sched/` plus supporting research artifacts (`modeling/`, `exp/`, `paper_figs/`). When making changes, assume the goal is research reproducibility for the accompanying paper ("Performance Modeling and SLA-Aware Scheduling for LLM Inference Systems"), not upstream contribution.

Note: much of the custom code and its docstrings are written in Chinese. Match the surrounding language when editing those files.

## The custom scheduling system

Three cooperating modules are wired into the stock scheduler at `vllm/v1/core/sched/scheduler.py`. All three are imported defensively (try/except → availability flag) and the scheduler falls back to legacy load-aware behavior if any is missing or errors. Understanding the data flow across them requires reading them together:

- **`vllm/v1/core/sched/sla_aware/`** — the SLA scheduler package. Key classes:
  - `SLAScheduler` (`sla_scheduler.py`) — top-level entry. `scheduler.py` calls `compute_schedule_decision(...)` each scheduling step to get `{token_budget, target_latency, prioritize_decode}`, and feeds back measured latency via `add_observation(...)`. Owns a `PerformancePredictor` and an `SLAOptimizer`.
  - `PerformancePredictor` (`performance_predictor.py`) — wraps the latency model; supports pretrained-only, hybrid, and online-training modes; tracks MAPE and triggers online re-fits.
  - `ThroughputSaturationModel` / `StableClusterModel` (`throughput_model.py`) — the physics-inspired latency model. Core form: `Thr(B,S) = P_max·(1−e^{−k_B·B})·(1−e^{−k_S·S})`, `T = w1·S/Thr + τ_B·B + τ_S·S` (6-parameter). Fitted via two-stage `scipy.optimize.curve_fit`; serialized as `.pkl`.
  - `SLAOptimizer` (`optimizer.py`) — the three-phase scheduling algorithm (search batch size → solve optimal tokens → greedy allocation), priority order decode > prefill > waiting, plus `compute_adaptive_target_latency(queue_length)`.
  - `config.py` — `SLASchedulerConfig`, populated **entirely from `VLLM_SLA_*` / `VLLM_SLO_*` environment variables** (see the package `README.md` for the full list). There is no config-file path.
  - `*.pkl` files (e.g. `fitted_model_h100.pkl`, `fitted_model_h100_6param.pkl`, `stable_model_*.pkl`) are pretrained model artifacts loaded by the predictor; pick via `VLLM_SLA_PRETRAINED_PATH`.
- **`vllm/v1/core/sched/engine_agent.py`** — `EngineAgent` (the "ELRAR" agent). Pushes live engine state (predicted latency, KV-cache occupancy, pending tokens, capacity) over UDP to an external State Gateway / cluster router. Configured via `VLLM_ENABLE_ELRAR` / `VLLM_ELRAR_*` env vars. Independent of the SLA scheduler.

Enable the system with `VLLM_SLA_SCHEDULER_ENABLED=true`; see `vllm/v1/core/sched/sla_aware/README.md` for the deployment stages (data-collection → pretrained → production) and the full env-var reference.

### Modeling & experiment artifacts (not part of the served engine)
- `modeling/` — standalone derivation of the latency model (`THEORY_DOCUMENT.md`, `METHODOLOGY_PAPER.md`), offline fitting code, and the `.pkl` models it produces.
- `exp/` — scheduler profiling harness (`scheduler_profiling_example.py`, `SCHEDULER_PROFILING_README.md`) and per-GPU profiling results (`profiling_result_{a100,a6000,h100,h100_qwen32b}`).
- `paper_figs/` — plotting scripts (`plot_*.py`) and figure outputs per experiment (`adaptive_model_exp`, `ablation_exp`, `e2e_exp`, `heterogeneous_exp`, etc.). These read the JSON result files alongside them; editing a plot usually means editing the `plot_*.py` and its input JSON together.
- `create_pretrained_model.py` (inside `sla_aware/`) — CLI to fit a `.pkl` from profiling logs (`--data`, `--output`, `--demo`, `--analyze-only`).
- `exp_discard/` — deprecated experiments; do not build on it.

## Build

Standard vLLM build. This fork is CUDA-oriented (H100/A100/A6000 profiling artifacts).

```bash
# Editable install against an existing PyTorch (fast iteration on Python code)
python use_existing_torch.py
pip install -r requirements/build.txt
pip install -e . --no-build-isolation
```

Pure-Python changes to the scheduling modules do **not** require a rebuild — the `.so`/CUDA kernels are unaffected. Just restart the server/process.

## Lint & format

Linting is via **pre-commit** (not `format.sh`, which is now a stub). Config: `.pre-commit-config.yaml`; rules: `pyproject.toml` (ruff line-length 80, yapf, isort, mypy, codespell).

```bash
pip install -r requirements/lint.txt
pre-commit install
pre-commit run --all-files          # run all hooks
pre-commit run --files <path>...     # lint specific files
```

## Tests

Tests live in `tests/` and run under pytest.

```bash
pytest tests/v1/core/                              # V1 scheduler/core tests
pytest tests/v1/core/test_scheduler.py             # one file
pytest tests/v1/core/test_scheduler.py::test_name  # one test
pytest -v -s <path>                                # verbose, no capture (see .buildkite/test-pipeline.yaml for CI groupings)
```

Many tests require a GPU and/or download model weights; prefer running a single targeted test file locally rather than the full suite.

## Working conventions specific to this fork

- The SLA/ELRAR modules are intentionally **minimally invasive** and **fail-open**: preserve the try/except-import + availability-flag + fallback pattern in `scheduler.py` when touching integration points. Never let an SLA-path exception crash the scheduling loop.
- All tuning happens through environment variables, not code edits — when adding a knob, thread it through `config.py` (`SLASchedulerConfig` + `from_env`) rather than hardcoding.
- `.pkl` model artifacts are committed and referenced by path; if you retrain a model, keep the filename/GPU naming convention (`fitted_model_<gpu>[_<variant>].pkl`) so existing configs and plots keep resolving.
- Changes to a paper experiment generally span three places: the runtime module that emits data, the JSON results under `paper_figs/<exp>/`, and the `plot_*.py` that renders it. Keep them consistent.
