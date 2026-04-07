# Code Folder — Developer Reference

This folder contains all implementation, configs, scripts, and generated outputs for the battery optimization project.

For setup instructions, configuration reference, and how to run the project, see the [repository root README](../README.md).

---

## Package Layout (`battery_opt/`)

### Core modules

| Module | Purpose |
|---|---|
| `config.py` | Dataclasses for all config objects (`CaseConfig`, `BatteryConfig`, `SolverConfig`, etc.) and JSON loaders (`load_case_config`, `load_assignment4_config`, …) |
| `data.py` | Loads the CSV dataset, validates required columns, infers the hourly time step, and applies scenario-level data overrides (e.g. price multipliers) |
| `solve.py` | Main solve entry point: picks the right formulation builder, instantiates the HiGHS solver via PuLP, extracts the schedule, and returns an `OptimizationResult` |
| `results.py` | `OptimizationResult` dataclass — objective value, status, runtime, schedule DataFrame, and summary metrics |
| `metrics.py` | Post-solve KPI calculation: grid import/export volumes, battery throughput, equivalent cycles, utilization rate, self-consumption proxy |
| `rolling_horizon.py` | Receding-horizon solver: iterates over overlapping data windows and stitches together a full-horizon schedule with configurable terminal-state constraints |
| `cli.py` | `argparse`-based CLI with subcommands: `solve`, `benchmark`, `assignment4`–`assignment7` |

### Formulation modules (`battery_opt/formulations/`)

| Module | Purpose |
|---|---|
| `common.py` | Builds the shared LP/MIP model: decision variables (`soc`, `charge_kw`, `discharge_kw`, `mode`, `net_grid_kwh`), balance constraints, SOC bounds, power limits, and the energy-arbitrage objective |
| `basic.py` | **Basic formulation** — binary `mode` variable controls whether the battery charges or discharges; power limits are indexed by mode |
| `tighter.py` | **Tighter formulation** — adds explicit SOC upper/lower bounds at each time step derived from the current mode, yielding a stronger LP relaxation and faster MIP solving |

### Analysis modules (`battery_opt/analysis/`)

| Module | Purpose |
|---|---|
| `common.py` | Shared utilities: safe solve wrapper (catches infeasibility), scenario config overrides, fractional data window builder |
| `assignment4.py` | Formulation benchmark; horizon-fraction scaling study (10 %–100 %); LP-relaxation gap analysis across parameter scenarios |
| `assignment5.py` | Grid-fee sweep; tracks profit, grid import/export volumes, and self-consumption proxy |
| `assignment6.py` | Degradation-cost sweep; tracks battery throughput and equivalent cycles against a zero-degradation baseline |
| `assignment7.py` | Rolling-horizon experiment runner; compares perfect foresight against limited-foresight policies (weekly, 2-day, day-ahead) |

---

## Config Files (`configs/`)

| File | Contents |
|---|---|
| `base_case.json` | Battery specs, dataset path, and solver settings (shared by all assignments) |
| `assignment4.json` | Benchmark settings, fractional window sizes, window positions, LP-relaxation scenario list |
| `assignment5.json` | Grid-fee values to sweep |
| `assignment6.json` | Degradation-cost values to sweep |
| `assignment7.json` | Rolling-horizon window sizes and extension scenario hooks |

---

## Scripts (`scripts/`)

Each script in `scripts/` is a thin wrapper that loads configs from `configs/` and calls the corresponding analysis runner. Run them from the **repository root**:

```bash
python code/scripts/run_base_case.py
python code/scripts/run_assignment4.py
python code/scripts/run_assignment5.py
python code/scripts/run_assignment6.py
python code/scripts/run_assignment7.py
```

---

## Extending the Project

### Adding a new formulation

1. Create `battery_opt/formulations/my_formulation.py`.
2. Import `build_common_model` from `formulations/common.py` and call it to get the base `ModelArtifacts`.
3. Add any additional constraints or variables to `artifacts.problem`.
4. Export your builder function from `formulations/__init__.py`.
5. Register it in the `BUILDERS` dict in `solve.py`.
6. Add the new value to the `Formulation` enum in `config.py`.

### Adding a new analysis

1. Create `battery_opt/analysis/my_analysis.py` with a `run_my_analysis_suite(case_config, analysis_config, output_dir)` function.
2. Add a corresponding config dataclass and loader to `config.py`.
3. Add a new subcommand to `cli.py` following the pattern of the existing assignment parsers.
4. Add a `code/scripts/run_my_analysis.py` entry-point script.
