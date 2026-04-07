from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from formulations.basic import Basic
from formulations.tighter import Tighter
from formulations.params import PARAMS

from . import scenarios
from .wrapper import solve_model

sizes = [0.1, 0.2, 0.5, 0.75]
periods = ["start", "middle", "end"]
output_dir = Path("outputs/assignment4")
results_path = output_dir / "assignment4_horizon_scaling_results.csv"
gap_results_path = output_dir / "assignment4_horizon_scaling_gaps.csv"
plot_path = output_dir / "assignment4_horizon_scaling_gap_comparison.png"


def run_horizon_scaling(data: pd.DataFrame) -> pd.DataFrame:
    base_data = data.copy().reset_index(drop=True)
    results = []
    for size in sizes:
        for period in periods:
            scaled_data = scenarios.horizon_scale(base_data, size, period)
            scenario_label = f"{size:.0%} - {period}"
            for formulation in [Basic, Tighter]:
                for lp in [False, True]:
                    results.append(
                        {
                            "size": size,
                            "period": period,
                            "scenario": scenario_label,
                            "formulation": formulation.__name__,
                            "model_variant": "LP" if lp else "MIP",
                            "lp_relaxation": lp,
                            "profit": solve_model(
                                formulation, scaled_data, PARAMS, lp=lp
                            ),
                        }
                    )

    results_df = pd.DataFrame(results)
    gap_df = _build_gap_table(results_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    gap_df.to_csv(gap_results_path, index=False)
    _plot_horizon_scaling_gaps(gap_df)

    print(gap_df.set_index("scenario")[["basic_gap", "tighter_gap"]])
    return results_df


def _build_gap_table(results_df: pd.DataFrame) -> pd.DataFrame:
    pivot_df = results_df.pivot(
        index=["size", "period", "scenario"],
        columns=["formulation", "model_variant"],
        values="profit",
    ).reset_index()
    pivot_df["basic_gap"] = pivot_df[("Basic", "LP")] - pivot_df[("Basic", "MIP")]
    pivot_df["tighter_gap"] = (
        pivot_df[("Tighter", "LP")] - pivot_df[("Tighter", "MIP")]
    )

    gap_df = pd.DataFrame(
        {
            "size": pivot_df["size"],
            "period": pivot_df["period"],
            "scenario": pivot_df["scenario"],
            "basic_mip": pivot_df[("Basic", "MIP")],
            "basic_lp": pivot_df[("Basic", "LP")],
            "basic_gap": pivot_df["basic_gap"],
            "tighter_mip": pivot_df[("Tighter", "MIP")],
            "tighter_lp": pivot_df[("Tighter", "LP")],
            "tighter_gap": pivot_df["tighter_gap"],
        }
    )
    gap_df["period_order"] = gap_df["period"].map(
        {period_name: idx for idx, period_name in enumerate(periods)}
    )
    gap_df = gap_df.sort_values(["size", "period_order"]).drop(
        columns=["period_order"]
    )
    return gap_df.reset_index(drop=True)


def _plot_horizon_scaling_gaps(gap_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    scenario_labels = list(gap_df["scenario"])
    x = list(range(len(scenario_labels)))
    width = 0.35

    ax.bar(
        [position - width / 2 for position in x],
        gap_df["basic_gap"],
        width=width,
        label="Basic Gap",
    )
    ax.bar(
        [position + width / 2 for position in x],
        gap_df["tighter_gap"],
        width=width,
        label="Tighter Gap",
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("LP - MIP Gap")
    ax.set_title("Horizon Scaling LP-MIP Gaps by Formulation")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
