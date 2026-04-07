from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import Assignment6Config, CaseConfig, ScenarioOverrideConfig
from .common import apply_scenario_overrides, load_analysis_data, safe_solve


def run_assignment6_degradation_study(case_config: CaseConfig, assignment6_config: Assignment6Config) -> pd.DataFrame:
    base_data = load_analysis_data(case_config, assignment6_config.dataset_scope)
    records = []

    for formulation in assignment6_config.formulations:
        for degradation_cost in assignment6_config.degradation_cost_values:
            scenario = ScenarioOverrideConfig(
                degradation_cost_eur_per_kwh_throughput=degradation_cost,
            )
            scenario_case, scenario_data = apply_scenario_overrides(case_config, base_data, scenario)
            result, error = safe_solve(scenario_case, formulation, prepared_data=scenario_data)
            if error is not None:
                records.append(
                    {
                        "formulation": formulation.value,
                        "degradation_cost_eur_per_kwh_throughput": degradation_cost,
                        "status": error,
                        "periods": len(scenario_data),
                    }
                )
                continue

            records.append(
                {
                    "formulation": formulation.value,
                    "degradation_cost_eur_per_kwh_throughput": degradation_cost,
                    "status": result.status_name,
                    "periods": len(scenario_data),
                    "objective_eur": result.objective_value_eur,
                    "runtime_seconds": result.runtime_seconds,
                    "throughput_kwh": result.summary["throughput_kwh"],
                    "equivalent_cycles": result.summary["equivalent_cycles"],
                    "total_import_kwh": result.summary["total_import_kwh"],
                    "total_export_kwh": result.summary["total_export_kwh"],
                    "average_soc_utilization": result.summary["average_soc_utilization"],
                    "implied_degradation_cost_eur": (
                        degradation_cost * result.summary["throughput_kwh"]
                    ),
                }
            )

    return pd.DataFrame(records)


def summarize_assignment6(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    valid = results[results["status"] == "OPTIMAL"].copy()
    if valid.empty:
        return pd.DataFrame()
    baseline = valid[
        valid["degradation_cost_eur_per_kwh_throughput"]
        == valid["degradation_cost_eur_per_kwh_throughput"].min()
    ][["formulation", "objective_eur", "throughput_kwh", "equivalent_cycles"]].rename(
        columns={
            "objective_eur": "baseline_objective_eur",
            "throughput_kwh": "baseline_throughput_kwh",
            "equivalent_cycles": "baseline_equivalent_cycles",
        }
    )
    summary = valid.merge(baseline, on="formulation", how="left")
    summary["objective_change_from_baseline_eur"] = (
        summary["objective_eur"] - summary["baseline_objective_eur"]
    )
    summary["throughput_change_from_baseline_kwh"] = (
        summary["throughput_kwh"] - summary["baseline_throughput_kwh"]
    )
    summary["equivalent_cycles_change_from_baseline"] = (
        summary["equivalent_cycles"] - summary["baseline_equivalent_cycles"]
    )
    return summary


def run_assignment6_suite(
    case_config: CaseConfig,
    assignment6_config: Assignment6Config,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = run_assignment6_degradation_study(case_config, assignment6_config)
    summary = summarize_assignment6(results)

    results.to_csv(output_path / "assignment6_degradation_study.csv", index=False)
    summary.to_csv(output_path / "assignment6_degradation_summary.csv", index=False)
    _plot_assignment6_metric(
        results,
        "objective_eur",
        "Objective by Degradation Cost",
        output_path / "assignment6_objective_by_degradation.png",
    )
    _plot_assignment6_metric(
        results,
        "throughput_kwh",
        "Battery Throughput by Degradation Cost",
        output_path / "assignment6_throughput_by_degradation.png",
    )

    return {
        "assignment6_degradation_study": results,
        "assignment6_degradation_summary": summary,
    }


def _plot_assignment6_metric(results: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    valid = results[(results["status"] == "OPTIMAL") & results[metric].notna()].copy()
    if valid.empty:
        return
    pivot = valid.pivot(
        index="degradation_cost_eur_per_kwh_throughput",
        columns="formulation",
        values=metric,
    )
    _BW_LINE_STYLES = [
        {"marker": "o", "markersize": 8, "linestyle": "-",  "linewidth": 2.0},
        {"marker": "s", "markersize": 8, "linestyle": "--", "linewidth": 1.5,
         "markerfacecolor": "none", "markeredgewidth": 1.5},
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, style in zip(pivot.columns, _BW_LINE_STYLES):
        ax.plot(pivot.index, pivot[col], label=col, **style)
    ax.set_title(title)
    ax.set_xlabel("Degradation cost (EUR/kWh throughput)")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
