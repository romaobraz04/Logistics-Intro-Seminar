from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import Assignment5Config, CaseConfig, ScenarioOverrideConfig
from .common import apply_scenario_overrides, load_analysis_data, safe_solve


def run_assignment5_grid_fee_sweep(case_config: CaseConfig, assignment5_config: Assignment5Config) -> pd.DataFrame:
    base_data = load_analysis_data(case_config, assignment5_config.dataset_scope)
    records = []

    total_hours = len(base_data)

    for formulation in assignment5_config.formulations:
        for grid_fee in assignment5_config.grid_fee_values:
            hours_below_fee = int((base_data["price_eur_per_kwh"] < grid_fee).sum())
            hours_below_fee_pct = hours_below_fee / total_hours * 100

            scenario = ScenarioOverrideConfig(grid_fee_eur_per_kwh=grid_fee)
            scenario_case, scenario_data = apply_scenario_overrides(case_config, base_data, scenario)
            result, error = safe_solve(scenario_case, formulation, prepared_data=scenario_data)
            if error is not None:
                records.append(
                    {
                        "formulation": formulation.value,
                        "grid_fee_eur_per_kwh": grid_fee,
                        "status": error,
                        "periods": len(scenario_data),
                        "hours_below_fee": hours_below_fee,
                        "hours_below_fee_pct": hours_below_fee_pct,
                    }
                )
                continue

            records.append(
                {
                    "formulation": formulation.value,
                    "grid_fee_eur_per_kwh": grid_fee,
                    "status": result.status_name,
                    "periods": len(scenario_data),
                    "objective_eur": result.objective_value_eur,
                    "runtime_seconds": result.runtime_seconds,
                    "total_grid_fee_paid_eur": grid_fee * result.summary["total_grid_exchange_kwh"],
                    "total_grid_exchange_kwh": result.summary["total_grid_exchange_kwh"],
                    "total_import_kwh": result.summary["total_import_kwh"],
                    "total_export_kwh": result.summary["total_export_kwh"],
                    "throughput_kwh": result.summary["throughput_kwh"],
                    "equivalent_cycles": result.summary["equivalent_cycles"],
                    "average_soc_utilization": result.summary["average_soc_utilization"],
                    "local_self_consumed_kwh": result.summary["local_self_consumed_kwh"],
                    "self_consumption_rate": result.summary["self_consumption_rate"],
                    "fee_as_fraction_of_profit": ((grid_fee * result.summary["total_grid_exchange_kwh"]) / result.objective_value_eur if result.objective_value_eur and result.objective_value_eur > 0 else None),
                    "hours_below_fee": hours_below_fee,
                    "hours_below_fee_pct": hours_below_fee_pct,
                }
            )

    return pd.DataFrame(records)


def summarize_assignment5(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    valid = results[results["status"] == "OPTIMAL"].copy()
    if valid.empty:
        return pd.DataFrame()
    baseline = valid[valid["grid_fee_eur_per_kwh"] == valid["grid_fee_eur_per_kwh"].min()][
        ["formulation", "objective_eur", "total_grid_exchange_kwh", "self_consumption_rate"]
    ].rename(
        columns={
            "objective_eur": "baseline_objective_eur",
            "total_grid_exchange_kwh": "baseline_grid_exchange_kwh",
            "self_consumption_rate": "baseline_self_consumption_rate",
        }
    )
    summary = valid.merge(baseline, on="formulation", how="left")
    summary["objective_change_from_baseline_eur"] = (
        summary["objective_eur"] - summary["baseline_objective_eur"]
    )
    summary["total_profit_loss_eur"] = (
        summary["baseline_objective_eur"] - summary["objective_eur"]
    )
    summary["direct_fee_burden_eur"] = (
        summary["grid_fee_eur_per_kwh"] * summary["total_grid_exchange_kwh"]
    )
    summary["behavioural_cost_eur"] = (
        summary["total_profit_loss_eur"] - summary["direct_fee_burden_eur"]
    )
    summary["grid_exchange_change_from_baseline_kwh"] = (
        summary["total_grid_exchange_kwh"] - summary["baseline_grid_exchange_kwh"]
    )
    summary["self_consumption_rate_change"] = (
        summary["self_consumption_rate"] - summary["baseline_self_consumption_rate"]
    )
    return summary


def run_assignment5_suite(
    case_config: CaseConfig,
    assignment5_config: Assignment5Config,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = run_assignment5_grid_fee_sweep(case_config, assignment5_config)
    summary = summarize_assignment5(results)

    results.to_csv(output_path / "assignment5_grid_fee_sweep.csv", index=False)
    summary.to_csv(output_path / "assignment5_grid_fee_summary.csv", index=False)
    _plot_assignment5_metric(results, "objective_eur", "Objective by Grid Fee", output_path / "assignment5_objective_by_fee.png")
    _plot_assignment5_metric(
        results,
        "total_grid_exchange_kwh",
        "Grid Exchange by Grid Fee",
        output_path / "assignment5_grid_exchange_by_fee.png",
    )
    _plot_assignment5_metric(
        results,
        "self_consumption_rate",
        "Self-Consumption Rate by Grid Fee",
        output_path / "assignment5_self_consumption_by_fee.png",
    )
    _plot_assignment5_metric(
        results,
        "hours_below_fee",
        "Hours with Price Below Fee Threshold (Export Suppressed)",
        output_path / "assignment5_hours_below_fee.png",
    )
    _plot_assignment5_metric(
        results,
        "equivalent_cycles",
        "Equivalent Full Cycles by Grid Fee",
        output_path / "assignment5_equivalent_cycles_by_fee.png",
    )
    _plot_assignment5_metric(
        results,
        "total_import_kwh",
        "Total Grid Import by Grid Fee",
        output_path / "assignment5_import_by_fee.png",
    )
    _plot_assignment5_metric(
        results,
        "total_export_kwh",
        "Total Grid Export by Grid Fee",
        output_path / "assignment5_export_by_fee.png",
    )
    if not summary.empty:
        _plot_assignment5_metric(
            summary,
            "behavioural_cost_eur",
            "Behavioural Cost by Grid Fee",
            output_path / "assignment5_behavioural_cost_by_fee.png",
        )

    return {
        "assignment5_grid_fee_sweep": results,
        "assignment5_grid_fee_summary": summary,
    }


def _plot_assignment5_metric(results: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    valid = results[(results["status"] == "OPTIMAL") & results[metric].notna()].copy()
    if valid.empty:
        return
    pivot = valid.pivot(index="grid_fee_eur_per_kwh", columns="formulation", values=metric)
    ax = pivot.plot(marker="o", figsize=(8, 4))
    ax.set_title(title)
    ax.set_xlabel("Grid fee (EUR/kWh)")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=150)
    plt.close(ax.figure)
