from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Assignment5Config, CaseConfig, ScenarioOverrideConfig
from .common import apply_scenario_overrides, load_analysis_data, safe_solve

_SERIES_STYLES: dict[str, dict] = {
    "basic":   {"marker": "o", "markersize": 8, "linestyle": "-",  "linewidth": 2.0},
    "tighter": {"marker": "s", "markersize": 8, "linestyle": "--", "linewidth": 1.5,
                "markerfacecolor": "none", "markeredgewidth": 1.5},
}


def _compute_no_battery_objective(data: pd.DataFrame, grid_fee: float) -> dict:
    d = data["net_demand_kwh"].values
    lam = data["price_eur_per_kwh"].values
    passive_import = np.clip(d, 0, None)
    passive_export = np.clip(-d, 0, None)
    no_bat_obj = (-(lam + grid_fee) * passive_import + (lam - grid_fee) * passive_export).sum()
    return {
        "no_battery_objective_eur": no_bat_obj,
        "passive_import_kwh": float(passive_import.sum()),
        "passive_export_kwh": float(passive_export.sum()),
    }


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

            no_bat = _compute_no_battery_objective(scenario_data, grid_fee)

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
                        **no_bat,
                    }
                )
                continue

            battery_value_eur = result.objective_value_eur - no_bat["no_battery_objective_eur"]

            records.append(
                {
                    "formulation": formulation.value,
                    "grid_fee_eur_per_kwh": grid_fee,
                    "status": result.status_name,
                    "periods": len(scenario_data),
                    "objective_eur": result.objective_value_eur,
                    **no_bat,
                    "battery_value_eur": battery_value_eur,
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

    t_wall_start = time.perf_counter()
    results = run_assignment5_grid_fee_sweep(case_config, assignment5_config)
    t_wall_end = time.perf_counter()
    summary = summarize_assignment5(results)

    # Print runtime table
    print("\nAssignment 5 solve times:")
    if "runtime_seconds" in results.columns:
        time_cols = results[["formulation", "grid_fee_eur_per_kwh", "runtime_seconds"]].copy()
        print(time_cols.to_string(index=False))
        total_solver = results["runtime_seconds"].sum()
        print(f"  Total solver time: {total_solver:.2f} s")
    print(f"  Wall time:         {t_wall_end - t_wall_start:.2f} s\n")

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
        "no_battery_objective_eur",
        "No-Battery Objective by Grid Fee",
        output_path / "assignment5_no_battery_objective_by_fee.png",
    )
    _plot_assignment5_metric(
        results,
        "battery_value_eur",
        "Battery Value (Incremental Profit) by Grid Fee",
        output_path / "assignment5_battery_value_by_fee.png",
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


_FORMULATION_STYLES: dict[str, dict] = {
    "basic": {"marker": "o", "linestyle": "-", "linewidth": 2.0, "markersize": 8, "color": "#1f77b4"},
    "tighter": {"marker": "s", "linestyle": "--", "linewidth": 2.0, "markersize": 7, "color": "#d62728"},
}


def _plot_assignment5_metric(results: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    valid = results[(results["status"] == "OPTIMAL") & results[metric].notna()].copy()
    if valid.empty:
        return
    pivot = valid.pivot(index="grid_fee_eur_per_kwh", columns="formulation", values=metric)
<<<<<<< Updated upstream
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in pivot.columns:
        style = _FORMULATION_STYLES.get(col, {"marker": "^", "linestyle": "-.", "linewidth": 2.0, "markersize": 7})
        ax.plot(pivot.index, pivot[col], label=col, **style)
    ax.set_title(title)
    ax.set_xlabel("Grid fee (EUR/kWh)")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)
=======

    has_diff_panel = "basic" in pivot.columns and "tighter" in pivot.columns
    if has_diff_panel:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(8, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
    else:
        fig, ax_top = plt.subplots(figsize=(8, 4))
        ax_bot = None

    # Top panel — absolute values
    for col in pivot.columns:
        kw = _SERIES_STYLES.get(col, {"marker": "s", "markersize": 8, "linestyle": "-.", "color": "black",
                                      "markerfacecolor": "none", "markeredgewidth": 1.5})
        ax_top.plot(pivot.index, pivot[col], label=col, **kw)
    ax_top.set_title(title)
    ax_top.set_ylabel(metric)
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # Bottom panel — basic minus tighter difference
    if has_diff_panel:
        diff = pivot["basic"] - pivot["tighter"]
        bar_colors = ["0.4" if d >= 0 else "white" for d in diff]
        ax_bot.bar(pivot.index, diff, width=0.006, color=bar_colors, edgecolor="black")
        ax_bot.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_bot.set_ylabel("basic − tighter (Δ)")
        ax_bot.grid(True, axis="y", alpha=0.3)
        if diff.abs().max() < 1e-6:
            ax_bot.set_title("Formulations produce identical values", fontsize=8)
        ax_bot.set_xlabel("Grid fee (EUR/kWh)")

>>>>>>> Stashed changes
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
