from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import CaseConfig, Formulation
from ..rolling_horizon import solve_rolling_horizon
from ..solve import solve_case
from .common import load_analysis_data


FORESIGHT_SCENARIOS = [
    {"name": "perfect_foresight", "window": None, "step": None, "overlap": None},
    {"name": "weekly", "window": 168, "step": 24, "overlap": 144},
    {"name": "two_day", "window": 48, "step": 24, "overlap": 24},
    {"name": "day_ahead", "window": 24, "step": 24, "overlap": 0},
]

FORMULATIONS = [Formulation.BASIC, Formulation.TIGHTER]

GRID_FEE = 0.02
DEGRADATION_COST = 0.01


def run_assignment7_rolling_horizon(
    case_config: CaseConfig,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_data = load_analysis_data(case_config, "configured_case")

    rh_config = replace(
        case_config,
        battery=replace(
            case_config.battery,
            grid_fee_eur_per_kwh=GRID_FEE,
            degradation_cost_eur_per_kwh_throughput=DEGRADATION_COST,
        ),
    )

    records = []

    for formulation in FORMULATIONS:
        for scenario in FORESIGHT_SCENARIOS:
            if scenario["window"] is None:
                record = _run_perfect_foresight(rh_config, formulation, base_data, scenario["name"])
            else:
                record = _run_rolling(
                    rh_config, formulation, base_data,
                    scenario["name"], scenario["window"], scenario["step"],
                    "hard_floor",
                )
            records.append(record)

        record_soft = _run_rolling(
            rh_config, formulation, base_data,
            "two_day_soft_valuation", 48, 24,
            "soft_valuation",
        )
        records.append(record_soft)

    results = pd.DataFrame(records)

    perfect = results[results["scenario"] == "perfect_foresight"][
        ["formulation", "total_profit_eur"]
    ].rename(columns={"total_profit_eur": "perfect_profit_eur"})
    results = results.merge(perfect, on="formulation", how="left")
    results["vpi_eur"] = results["perfect_profit_eur"] - results["total_profit_eur"]
    results["vpi_pct"] = results["vpi_eur"] / results["perfect_profit_eur"].abs() * 100

    results.to_csv(output_path / "assignment7_rolling_horizon.csv", index=False)

    _plot_profit_comparison(results, output_path)
    _plot_vpi(results, output_path)
    _plot_solve_time(results, output_path)

    return {"assignment7_rolling_horizon": results}


def _run_perfect_foresight(
    case_config: CaseConfig,
    formulation: Formulation,
    data: pd.DataFrame,
    scenario_name: str,
) -> dict:
    result = solve_case(case_config, formulation, prepared_data=data)
    schedule = result.schedule
    dt = case_config.time_step_hours or 1.0

    return {
        "scenario": scenario_name,
        "formulation": formulation.value,
        "terminal_strategy": "n/a",
        "window": len(data),
        "step": len(data),
        "overlap": 0,
        "n_subproblems": 1,
        "total_profit_eur": result.objective_value_eur,
        "total_solve_time_seconds": result.runtime_seconds,
        "equivalent_cycles": result.summary["equivalent_cycles"],
        "throughput_kwh": result.summary["throughput_kwh"],
        "total_import_kwh": result.summary["total_import_kwh"],
        "total_export_kwh": result.summary["total_export_kwh"],
        "total_grid_exchange_kwh": result.summary["total_grid_exchange_kwh"],
        "self_consumption_rate": result.summary["self_consumption_rate"],
        "periods": len(data),
    }


def _run_rolling(
    case_config: CaseConfig,
    formulation: Formulation,
    data: pd.DataFrame,
    scenario_name: str,
    window: int,
    step: int,
    terminal_strategy: str,
) -> dict:
    rh = solve_rolling_horizon(
        case_config, formulation, data,
        window=window, step=step,
        terminal_strategy=terminal_strategy,
    )
    return {
        "scenario": scenario_name,
        "formulation": formulation.value,
        "terminal_strategy": terminal_strategy,
        "window": window,
        "step": step,
        "overlap": window - step,
        "n_subproblems": rh["n_subproblems"],
        "total_profit_eur": rh["total_profit_eur"],
        "total_solve_time_seconds": rh["total_solve_time_seconds"],
        "equivalent_cycles": rh["summary"]["equivalent_cycles"],
        "throughput_kwh": rh["summary"]["throughput_kwh"],
        "total_import_kwh": rh["summary"]["total_import_kwh"],
        "total_export_kwh": rh["summary"]["total_export_kwh"],
        "total_grid_exchange_kwh": rh["summary"]["total_grid_exchange_kwh"],
        "self_consumption_rate": rh["summary"]["self_consumption_rate"],
        "periods": rh["periods_implemented"],
    }


def _plot_profit_comparison(results: pd.DataFrame, output_path: Path) -> None:
    hard_floor = results[results["terminal_strategy"].isin(["n/a", "hard_floor"])].copy()
    pivot = hard_floor.pivot(index="scenario", columns="formulation", values="total_profit_eur")
    order = ["perfect_foresight", "weekly", "two_day", "day_ahead"]
    pivot = pivot.reindex([s for s in order if s in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Total Profit by Foresight Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Profit (EUR)")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path / "assignment7_profit_comparison.png", dpi=150)
    plt.close(ax.figure)


def _plot_vpi(results: pd.DataFrame, output_path: Path) -> None:
    hard_floor = results[results["terminal_strategy"].isin(["n/a", "hard_floor"])].copy()
    rh_only = hard_floor[hard_floor["scenario"] != "perfect_foresight"]
    if rh_only.empty:
        return
    pivot = rh_only.pivot(index="scenario", columns="formulation", values="vpi_eur")
    order = ["weekly", "two_day", "day_ahead"]
    pivot = pivot.reindex([s for s in order if s in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Value of Perfect Information (VPI)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("VPI (EUR)")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path / "assignment7_vpi.png", dpi=150)
    plt.close(ax.figure)


def _plot_solve_time(results: pd.DataFrame, output_path: Path) -> None:
    hard_floor = results[results["terminal_strategy"].isin(["n/a", "hard_floor"])].copy()
    pivot = hard_floor.pivot(index="scenario", columns="formulation", values="total_solve_time_seconds")
    order = ["perfect_foresight", "weekly", "two_day", "day_ahead"]
    pivot = pivot.reindex([s for s in order if s in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Total Solve Time by Foresight Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Time (seconds)")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path / "assignment7_solve_time.png", dpi=150)
    plt.close(ax.figure)
