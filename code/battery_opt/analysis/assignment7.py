from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    t_wall_start = time.perf_counter()
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
    schedules: dict[tuple[str, str], pd.DataFrame] = {}

    for formulation in FORMULATIONS:
        for scenario in FORESIGHT_SCENARIOS:
            if scenario["window"] is None:
                record, schedule = _run_perfect_foresight(rh_config, formulation, base_data, scenario["name"])
            else:
                record, schedule = _run_rolling(
                    rh_config, formulation, base_data,
                    scenario["name"], scenario["window"], scenario["step"],
                    "hard_floor",
                )
            records.append(record)
            schedules[(scenario["name"], formulation.value)] = schedule

        record_soft, schedule_soft = _run_rolling(
            rh_config, formulation, base_data,
            "two_day_soft_valuation", 48, 24,
            "soft_valuation",
        )
        records.append(record_soft)
        schedules[("two_day_soft_valuation", formulation.value)] = schedule_soft

        # EPEX day-ahead scenario: decision epoch at hour 13, window=35, step=24
        record_epex, schedule_epex = _run_rolling(
            rh_config, formulation, base_data,
            "epex_day_ahead", 35, 24,
            "hard_floor",
            first_window=13, first_step=13,
        )
        records.append(record_epex)
        schedules[("epex_day_ahead", formulation.value)] = schedule_epex

    results = pd.DataFrame(records)

    # Add terminal SOC for each scenario
    for idx, row in results.iterrows():
        key = (row["scenario"], row["formulation"])
        sched = schedules.get(key, pd.DataFrame())
        results.loc[idx, "terminal_soc_kwh"] = float(sched["soc_kwh"].iloc[-1]) if not sched.empty else np.nan

    perfect = results[results["scenario"] == "perfect_foresight"][
        ["formulation", "total_profit_eur"]
    ].rename(columns={"total_profit_eur": "perfect_profit_eur"})
    results = results.merge(perfect, on="formulation", how="left")
    results["vpi_eur"] = results["perfect_profit_eur"] - results["total_profit_eur"]
    results["vpi_pct"] = results["vpi_eur"] / results["perfect_profit_eur"].abs() * 100

    # ── Terminal-SoC comparability check ──────────────────────────────────────
    # Scenarios that end with a different SoC than perfect foresight (e.g.
    # soft_valuation) have an unfair VPI comparison.  We add three helper
    # columns so the CSV is self-documenting:
    #   terminal_soc_adj_eur  – EUR value of the SoC difference (at mean price)
    #   vpi_comparable        – False when |diff| >= 0.1 kWh
    #   vpi_adjusted_eur      – VPI corrected for the terminal-SoC difference
    #   terminal_soc_note     – plain-English explanation when not comparable
    mean_price = float(base_data["price_eur_per_kwh"].mean())
    pf_terminal = (
        results[results["scenario"] == "perfect_foresight"]
        .set_index("formulation")["terminal_soc_kwh"]
    )

    def _soc_adj(row: pd.Series) -> tuple[float, bool, str]:
        if row["scenario"] == "perfect_foresight":
            return 0.0, True, ""
        pf_soc = pf_terminal.get(row["formulation"], np.nan)
        diff = float(pf_soc - row["terminal_soc_kwh"])   # +ve → PF kept more energy
        adj = mean_price * diff
        comparable = abs(diff) < 0.1
        note = (
            ""
            if comparable
            else (
                f"Terminal SoC differs from perfect_foresight by {diff:+.2f} kWh; "
                f"raw VPI is not directly comparable. "
                f"vpi_adjusted_eur corrects for this ({adj:+.2f} EUR adjustment)."
            )
        )
        return adj, comparable, note

    adj_results = results.apply(_soc_adj, axis=1, result_type="expand")
    adj_results.columns = ["terminal_soc_adj_eur", "vpi_comparable", "terminal_soc_note"]
    results = pd.concat([results, adj_results], axis=1)
    results["vpi_adjusted_eur"] = results["vpi_eur"] + results["terminal_soc_adj_eur"]

    results.to_csv(output_path / "assignment7_rolling_horizon.csv", index=False)

    # ── Runtime summary ────────────────────────────────────────────────────────
    print("\nAssignment 7 scenario runtimes:")
    basic_times = results[results["formulation"] == "basic"][
        ["scenario", "total_solve_time_seconds"]
    ].copy()
    print(basic_times.to_string(index=False))
    total_solver = results["total_solve_time_seconds"].sum()
    print(f"  Total solver time (all formulations): {total_solver:.2f} s")

    _plot_profit_comparison(results, output_path)
    _plot_vpi(results, output_path)
    _plot_solve_time(results, output_path)
    _plot_soc_trajectory(schedules, output_path)
    _decompose_vpi(results, rh_config, output_path)

    # Grid fee sensitivity: re-run at phi=0
    sensitivity_results = _run_grid_fee_sensitivity(case_config, base_data, output_path)

    # Integration with assignment 5: VPI across grid fee sweep
    vpi_fee_results = _run_vpi_vs_grid_fee(case_config, base_data, output_path)

    # Integration with assignment 6: VPI across degradation cost sweep
    vpi_deg_results = _run_vpi_vs_degradation(case_config, base_data, output_path)

    t_wall_end = time.perf_counter()
    print(f"  Wall time (total):                     {t_wall_end - t_wall_start:.2f} s\n")

    return {
        "assignment7_rolling_horizon": results,
        "assignment7_grid_fee_sensitivity": sensitivity_results,
        "assignment7_vpi_vs_grid_fee": vpi_fee_results,
        "assignment7_vpi_vs_degradation": vpi_deg_results,
    }


def _run_perfect_foresight(
    case_config: CaseConfig,
    formulation: Formulation,
    data: pd.DataFrame,
    scenario_name: str,
) -> tuple[dict, pd.DataFrame]:
    result = solve_case(
        case_config, formulation, prepared_data=data,
        terminal_soc_kwh=case_config.battery.initial_soc_kwh,
    )
    schedule = result.schedule

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
    }, schedule


def _run_rolling(
    case_config: CaseConfig,
    formulation: Formulation,
    data: pd.DataFrame,
    scenario_name: str,
    window: int,
    step: int,
    terminal_strategy: str,
    first_window: int | None = None,
    first_step: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    rh = solve_rolling_horizon(
        case_config, formulation, data,
        window=window, step=step,
        terminal_strategy=terminal_strategy,
        first_window=first_window, first_step=first_step,
    )
    schedule = rh["schedule"]
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
    }, schedule


# ---------- Analysis 1: VPI decomposition ----------

def _decompose_vpi(results: pd.DataFrame, case_config: CaseConfig, output_path: Path) -> None:
    """Split day-ahead VPI into extra degradation cost vs lost arbitrage profit."""
    deg_cost = case_config.battery.degradation_cost_eur_per_kwh_throughput

    decomp_rows = []
    for formulation in results["formulation"].unique():
        form_df = results[results["formulation"] == formulation]
        pf = form_df[form_df["scenario"] == "perfect_foresight"].iloc[0]

        for _, row in form_df.iterrows():
            if row["scenario"] == "perfect_foresight":
                continue
            if row["terminal_strategy"] not in ("hard_floor", "n/a"):
                continue

            throughput_diff = row["throughput_kwh"] - pf["throughput_kwh"]
            degradation_loss = deg_cost * throughput_diff
            total_vpi = row["vpi_eur"]
            arbitrage_loss = total_vpi - degradation_loss

            decomp_rows.append({
                "scenario": row["scenario"],
                "formulation": formulation,
                "vpi_eur": total_vpi,
                "degradation_loss_eur": degradation_loss,
                "arbitrage_loss_eur": arbitrage_loss,
                "extra_throughput_kwh": throughput_diff,
            })

    if not decomp_rows:
        return

    decomp = pd.DataFrame(decomp_rows)
    decomp.to_csv(output_path / "assignment7_vpi_decomposition.csv", index=False)

    # Plot stacked bar for basic formulation
    basic = decomp[decomp["formulation"] == "basic"].copy()
    if basic.empty:
        return

    order = ["weekly", "two_day", "day_ahead", "epex_day_ahead"]
    basic = basic.set_index("scenario")
    basic = basic.reindex([s for s in order if s in basic.index])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(basic)))
    degs = basic["degradation_loss_eur"].to_numpy()
    arbs = basic["arbitrage_loss_eur"].to_numpy()
    ax.bar(x, degs, label="Extra degradation cost")
    ax.bar(x, arbs, bottom=degs, label="Suboptimal arbitrage timing")
    ax.set_xticks(x)
    ax.set_xticklabels(list(basic.index), rotation=15)
    ax.set_title("VPI Decomposition: Degradation vs Arbitrage Loss")
    ax.set_ylabel("Loss (EUR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "assignment7_vpi_decomposition.png", dpi=150)
    plt.close(fig)


# ---------- Analysis 2: SoC trajectory plot ----------

def _plot_soc_trajectory(
    schedules: dict[tuple[str, str], pd.DataFrame],
    output_path: Path,
) -> None:
    """Overlay average daily SoC profile for selected scenarios (basic formulation)."""
    # weekly is included (dashed, low zorder) to verify it overlaps two-day
    scenarios_to_plot = ["weekly", "perfect_foresight", "two_day", "day_ahead", "epex_day_ahead"]
    labels = {
        "weekly": "Weekly (W=168, dashed — overlaps two-day)",
        "perfect_foresight": "Perfect Foresight",
        "two_day": "Two-Day (W=48)",
        "day_ahead": "Day-Ahead (W=24)",
        "epex_day_ahead": "EPEX Day-Ahead (W=35)",
    }
    styles = {
        "perfect_foresight": {"linestyle": "-",         "linewidth": 2.5, "marker": "o", "markersize": 7, "zorder": 3},
        "weekly":            {"linestyle": ":",         "linewidth": 1.2, "marker": "x", "markersize": 7, "zorder": 1, "alpha": 0.5},
        "two_day":           {"linestyle": "--",        "linewidth": 2.0, "marker": "s", "markersize": 7,
                              "markerfacecolor": "none", "markeredgewidth": 1.5, "zorder": 2},
        "day_ahead":         {"linestyle": "-.",        "linewidth": 2.0, "marker": "^", "markersize": 7, "zorder": 2},
        "epex_day_ahead":    {"linestyle": (0, (5, 1)), "linewidth": 2.0, "marker": "D", "markersize": 7,
                              "markerfacecolor": "none", "markeredgewidth": 1.5, "zorder": 2},
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    for scenario in scenarios_to_plot:
        key = (scenario, "basic")
        sched = schedules.get(key)
        if sched is None or sched.empty:
            continue

        sched = sched.copy()
        sched["hour"] = pd.to_datetime(sched["start"]).dt.hour
        avg_soc = sched.groupby("hour")["soc_kwh"].mean()
        kw = styles.get(scenario, {"linestyle": "-", "linewidth": 1.8, "marker": "o", "markersize": 7, "zorder": 2})
        ax.plot(avg_soc.index, avg_soc.values, label=labels.get(scenario, scenario), **kw)

    ax.set_title(
        "Average Daily SoC Profile by Foresight Scenario\n"
        "(Perfect Foresight line may be hidden behind weekly/two-day — profiles near-identical)"
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average SoC (kWh)")
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "assignment7_soc_trajectory.png", dpi=150)
    plt.close(fig)


# ---------- Analysis 3: Grid fee sensitivity ----------

def _run_grid_fee_sensitivity(
    case_config: CaseConfig,
    data: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Re-run scenarios at grid_fee=0 to test if foresight value is price- or fee-driven."""
    no_fee_config = replace(
        case_config,
        battery=replace(
            case_config.battery,
            grid_fee_eur_per_kwh=0.0,
            degradation_cost_eur_per_kwh_throughput=DEGRADATION_COST,
        ),
    )

    scenarios = [
        {"name": "perfect_foresight", "window": None},
        {"name": "two_day", "window": 48, "step": 24},
        {"name": "day_ahead", "window": 24, "step": 24},
    ]

    records = []
    formulation = Formulation.BASIC

    for scenario in scenarios:
        if scenario["window"] is None:
            record, _ = _run_perfect_foresight(no_fee_config, formulation, data, scenario["name"])
        else:
            record, _ = _run_rolling(
                no_fee_config, formulation, data,
                scenario["name"], scenario["window"], scenario["step"],
                "hard_floor",
            )
        records.append(record)

    results = pd.DataFrame(records)
    pf_profit = results.loc[results["scenario"] == "perfect_foresight", "total_profit_eur"].iloc[0]
    results["vpi_eur"] = pf_profit - results["total_profit_eur"]
    results["vpi_pct"] = results["vpi_eur"] / abs(pf_profit) * 100
    results["grid_fee"] = 0.0

    results.to_csv(output_path / "assignment7_grid_fee_sensitivity.csv", index=False)

    return results


# ---------- Analysis 4: VPI vs grid fee sweep (integration with assignment 5) ----------

GRID_FEE_VALUES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
DEGRADATION_COST_VALUES = [0.0, 0.005, 0.01, 0.02, 0.03]


def _run_vpi_vs_grid_fee(
    case_config: CaseConfig,
    data: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """VPI of day-ahead scenario across the assignment 5 grid fee sweep."""
    formulation = Formulation.BASIC
    rows = []

    for fee in GRID_FEE_VALUES:
        cfg = replace(
            case_config,
            battery=replace(
                case_config.battery,
                grid_fee_eur_per_kwh=fee,
                degradation_cost_eur_per_kwh_throughput=DEGRADATION_COST,
            ),
        )
        pf_rec, _ = _run_perfect_foresight(cfg, formulation, data, "perfect_foresight")
        da_rec, _ = _run_rolling(cfg, formulation, data, "day_ahead", 24, 24, "hard_floor")
        vpi = pf_rec["total_profit_eur"] - da_rec["total_profit_eur"]
        denom = abs(pf_rec["total_profit_eur"])
        rows.append({
            "grid_fee_eur_per_kwh": fee,
            "pf_profit_eur": pf_rec["total_profit_eur"],
            "da_profit_eur": da_rec["total_profit_eur"],
            "vpi_eur": vpi,
            "vpi_pct": vpi / denom * 100 if denom > 0 else float("nan"),
        })

    results = pd.DataFrame(rows)
    results.to_csv(output_path / "assignment7_vpi_vs_grid_fee.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(results["grid_fee_eur_per_kwh"], results["vpi_eur"], marker="o")
    ax1.set_title("VPI (EUR) vs Grid Fee")
    ax1.set_xlabel("Grid fee (EUR/kWh)")
    ax1.set_ylabel("VPI (EUR)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(results["grid_fee_eur_per_kwh"], results["vpi_pct"], marker="o", color="tab:orange")
    ax2.set_title("VPI (%) vs Grid Fee")
    ax2.set_xlabel("Grid fee (EUR/kWh)")
    ax2.set_ylabel("VPI (%)")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Day-Ahead VPI across Grid Fee Levels (Assignment 5 integration)")
    fig.tight_layout()
    fig.savefig(output_path / "assignment7_vpi_vs_grid_fee.png", dpi=150)
    plt.close(fig)

    return results


# ---------- Analysis 5: VPI vs degradation cost sweep (integration with assignment 6) ----------

def _run_vpi_vs_degradation(
    case_config: CaseConfig,
    data: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """VPI of day-ahead scenario across the assignment 6 degradation cost sweep."""
    formulation = Formulation.BASIC
    rows = []

    for deg in DEGRADATION_COST_VALUES:
        cfg = replace(
            case_config,
            battery=replace(
                case_config.battery,
                grid_fee_eur_per_kwh=GRID_FEE,
                degradation_cost_eur_per_kwh_throughput=deg,
            ),
        )
        pf_rec, _ = _run_perfect_foresight(cfg, formulation, data, "perfect_foresight")
        da_rec, _ = _run_rolling(cfg, formulation, data, "day_ahead", 24, 24, "hard_floor")
        vpi = pf_rec["total_profit_eur"] - da_rec["total_profit_eur"]
        denom = abs(pf_rec["total_profit_eur"])
        rows.append({
            "degradation_cost_eur_per_kwh": deg,
            "pf_profit_eur": pf_rec["total_profit_eur"],
            "da_profit_eur": da_rec["total_profit_eur"],
            "vpi_eur": vpi,
            "vpi_pct": vpi / denom * 100 if denom > 0 else float("nan"),
        })

    results = pd.DataFrame(rows)
    results.to_csv(output_path / "assignment7_vpi_vs_degradation.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(results["degradation_cost_eur_per_kwh"], results["vpi_eur"], marker="o")
    ax1.set_title("VPI (EUR) vs Degradation Cost")
    ax1.set_xlabel("Degradation cost (EUR/kWh throughput)")
    ax1.set_ylabel("VPI (EUR)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(results["degradation_cost_eur_per_kwh"], results["vpi_pct"], marker="o", color="tab:orange")
    ax2.set_title("VPI (%) vs Degradation Cost")
    ax2.set_xlabel("Degradation cost (EUR/kWh throughput)")
    ax2.set_ylabel("VPI (%)")
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Day-Ahead VPI across Degradation Cost Levels (Assignment 6 integration)")
    fig.tight_layout()
    fig.savefig(output_path / "assignment7_vpi_vs_degradation.png", dpi=150)
    plt.close(fig)

    return results


# ---------- Existing plots ----------

def _plot_profit_comparison(results: pd.DataFrame, output_path: Path) -> None:
    hard_floor = results[results["terminal_strategy"].isin(["n/a", "hard_floor"])].copy()
    pivot = hard_floor.pivot(index="scenario", columns="formulation", values="total_profit_eur")
    order = ["perfect_foresight", "weekly", "two_day", "day_ahead", "epex_day_ahead"]
    pivot = pivot.reindex([s for s in order if s in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Total Profit by Foresight Scenario\n(negative values = net cost; mandatory household load)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Profit (EUR)")
    ax.grid(True, alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2, fontsize=7)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path / "assignment7_profit_comparison.png", dpi=150)
    plt.close(ax.figure)


def _plot_vpi(results: pd.DataFrame, output_path: Path) -> None:
    hard_floor = results[results["terminal_strategy"].isin(["n/a", "hard_floor"])].copy()
    rh_only = hard_floor[hard_floor["scenario"] != "perfect_foresight"]
    if rh_only.empty:
        return
    pivot = rh_only.pivot(index="scenario", columns="formulation", values="vpi_eur")
    order = ["weekly", "two_day", "day_ahead", "epex_day_ahead"]
    pivot = pivot.reindex([s for s in order if s in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Value of Perfect Information (VPI)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("VPI (EUR)")
    ax.grid(True, alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=2, fontsize=7)
    ax.annotate(
        "Weekly and two-day bars at or near zero indicate negligible foresight value",
        xy=(0.5, -0.22), xycoords="axes fraction", ha="center", fontsize=8, color="grey",
    )
    ax.figure.tight_layout()
    ax.figure.subplots_adjust(bottom=0.18)
    ax.figure.savefig(output_path / "assignment7_vpi.png", dpi=150)
    plt.close(ax.figure)


def _plot_solve_time(results: pd.DataFrame, output_path: Path) -> None:
    hard_floor = results[results["terminal_strategy"].isin(["n/a", "hard_floor"])].copy()
    pivot = hard_floor.pivot(index="scenario", columns="formulation", values="total_solve_time_seconds")
    order = ["perfect_foresight", "weekly", "two_day", "day_ahead", "epex_day_ahead"]
    pivot = pivot.reindex([s for s in order if s in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Total Solve Time by Foresight Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Time (seconds)")
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path / "assignment7_solve_time.png", dpi=150)
    plt.close(ax.figure)
