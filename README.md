# Battery Optimization Repository

This repository is split into two layers:

- repository root: case material and shared project files
- `code/`: all implementation, configs, scripts, and generated results

The battery optimization code covers:

- both formulations from the case paper: `basic` and `tighter`
- assignment 4 analysis
- assignment 5 grid-fee analysis
- assignment 6 degradation analysis
- assignment 7 extension skeleton

## Repository Layout

At the repository root:

- `Description Case1.pdf`
- `Discussion Logistics Schedule.pdf`
- `Elgersma et al. (2024).pdf`
- `Slides Case1.pdf`
- `net_demand_and_price.csv`
- `README.md`

Inside `code/`:

- `battery_opt/`: Python package with models and analysis runners
- `configs/`: JSON configs for the base case and assignments
- `scripts/`: simple entry-point scripts
- `outputs/`: generated CSVs and plots

## Important Path Convention

The dataset and assignment PDFs stay at the repository root.

The code and outputs stay under `code/`.

The base case config at `code/configs/base_case.json` points to the dataset using a relative path to the repository root:

```json
"csv_path": "../../net_demand_and_price.csv"
```

The config loader resolves this path relative to the config file itself, not relative to your current working directory. That means the scripts keep working whether you run them from the repository root or from inside `code/`.

## Code Structure

`code/battery_opt/config.py`
Shared config objects and JSON loaders.

`code/battery_opt/data.py`
Dataset loading, validation, and scenario-level data modifications.

`code/battery_opt/formulations/basic.py`
Basic Operation MIP.

`code/battery_opt/formulations/tighter.py`
Tighter Operation MIP.

`code/battery_opt/solve.py`
Shared solve entry point for both formulations and LP relaxations.

`code/battery_opt/analysis/common.py`
Shared utilities for scenario overrides, safe solves, and fractional data windows.

`code/battery_opt/analysis/assignment4.py`
Assignment 4 benchmark, fraction-window horizon study, and LP-relaxation study.

`code/battery_opt/analysis/assignment5.py`
Assignment 5 grid-fee sweep and utilization metrics.

`code/battery_opt/analysis/assignment6.py`
Assignment 6 degradation-cost sweep.

`code/battery_opt/analysis/assignment7.py`
Assignment 7 extension skeleton with scenario hooks.

## Config Files

`code/configs/base_case.json`
Main battery and solver settings.

`code/configs/assignment4.json`
Assignment 4 settings:

- full-horizon benchmark
- fractional windows such as `10%`, `20%`, `50%`, `75%`, `100%`
- window positions: `start`, `middle`, `end`
- LP-relaxation scenarios

`code/configs/assignment5.json`
Assignment 5 grid-fee sweep.

`code/configs/assignment6.json`
Assignment 6 degradation sweep.

`code/configs/assignment7.json`
Assignment 7 extension scenarios.

## Scripts

The simplest way to run things from the repository root is through the scripts in `code/scripts/`.

Base solve:

```powershell
python code/scripts/run_base_case.py
```

Assignment 4:

```powershell
python code/scripts/run_assignment4.py
```

Assignment 5:

```powershell
python code/scripts/run_assignment5.py
```

Assignment 6:

```powershell
python code/scripts/run_assignment6.py
```

Assignment 7:

```powershell
python code/scripts/run_assignment7.py
```

## CLI Usage

If you want to use the package CLI directly, run it from inside `code/`:

```powershell
cd code
python -m battery_opt.cli solve --config configs/base_case.json --formulation basic
```

Other CLI commands:

```powershell
cd code
python -m battery_opt.cli assignment4 --config configs/base_case.json --analysis-config configs/assignment4.json --output-dir outputs/assignment4
python -m battery_opt.cli assignment5 --config configs/base_case.json --analysis-config configs/assignment5.json --output-dir outputs/assignment5
python -m battery_opt.cli assignment6 --config configs/base_case.json --analysis-config configs/assignment6.json --output-dir outputs/assignment6
python -m battery_opt.cli assignment7 --config configs/base_case.json --analysis-config configs/assignment7.json --output-dir outputs/assignment7
```

## Assignment Coverage

Assignment 4 includes:

- formulation benchmark
- fraction-based horizon scaling
- windows from different parts of the dataset
- LP-relaxation vs MIP comparison
- parameter experiments for prices, efficiencies, limits, and capacity

Assignment 5 includes:

- grid-fee sweep
- objective tracking
- grid import/export and total grid exchange
- battery throughput and equivalent cycles
- self-consumption proxy metrics

Assignment 6 includes:

- linear degradation-cost model based on throughput
- degradation sweeps from config
- comparison against the zero-degradation baseline

Assignment 7 includes:

- scenario skeleton for future extensions
- optional fractional windows
- optional schedule export
- isolated extension metric hook in `code/battery_opt/analysis/assignment7.py`

## Outputs

Generated outputs are written under:

- `code/outputs/assignment4`
- `code/outputs/assignment5`
- `code/outputs/assignment6`
- `code/outputs/assignment7`

## Notes

- `code/configs/base_case.json` still contains example battery values. Replace them with the exact case values before using final report results.
- Assignment 5 uses a self-consumption proxy because the dataset only gives net demand, not separate load and PV time series.
- Assignment 6 currently uses a linear degradation-cost formulation because it is easy to interpret and keeps the model linear.
- If you switch some studies from the configured case subset to the full dataset, larger windows may exceed the local size-limited Gurobi license.
