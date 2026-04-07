from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import pulp
from formulations.basic import Basic
from formulations.params import PARAMS
from formulations.tighter import Tighter

from . import scenarios


def solve_model(formulation_cls, data, params, lp=False):
    model = pulp.LpProblem(
        f"{formulation_cls.__name__}_{'LP' if lp else 'MIP'}",
        pulp.LpMaximize,
    )
    formulation = formulation_cls(model, data.copy(), params.copy(), lp=lp, msg=False)
    formulation.run_model()
    return pulp.value(model.objective)


def count_fractional_nodes(formulation, tol=1e-6):
    return sum(
        1
        for var_name, var in formulation.dec_vars.items()
        if var_name.startswith("mode_")
        and var.varValue is not None
        and tol < var.varValue < 1 - tol
    )


def solve_lp_model(formulation_cls, data, params):
    model = pulp.LpProblem(
        f"{formulation_cls.__name__}_LP",
        pulp.LpMaximize,
    )
    formulation = formulation_cls(model, data.copy(), params.copy(), lp=True, msg=False)
    formulation.run_model()
    return pulp.value(model.objective), count_fractional_nodes(formulation)


class assign4wrapper:

    def __init__(
        self,
        data: pd.DataFrame,
        price_scale: float = 1.5,
        low_efficiency: float = 0.85,
        tight_power_limit: float = 0.5,
        loose_power_limit: float = 5.0,
        small_capacity: float = 0.5,
        large_capacity: float = 10.0,
    ):

        self.data = data
        self.price_scale = price_scale
        self.low_efficiency = low_efficiency
        self.tight_power_limit = tight_power_limit
        self.loose_power_limit = loose_power_limit
        self.small_capacity = small_capacity
        self.large_capacity = large_capacity

    def run(self) -> pd.DataFrame:
        flat_prices_data = scenarios.flat_prices(self.data.copy())
        price_scale_data = scenarios.price_scale(
            self.data.copy(), scale=self.price_scale
        )
        perfect_efficiency_params = scenarios.perfect_efficiency(PARAMS.copy())
        low_efficiency_params = scenarios.low_efficiency(
            PARAMS.copy(), charge=self.low_efficiency, discharge=self.low_efficiency
        )
        tight_power_limits_params = scenarios.tight_power_limits(
            PARAMS.copy(),
            charge=self.tight_power_limit,
            discharge=self.tight_power_limit,
        )
        loose_power_limits_params = scenarios.loose_power_limits(
            PARAMS.copy(),
            charge=self.loose_power_limit,
            discharge=self.loose_power_limit,
        )
        small_capacity_params = scenarios.small_capacity(
            PARAMS.copy(), value=self.small_capacity
        )
        large_capacity_params = scenarios.large_capacity(
            PARAMS.copy(), value=self.large_capacity
        )

        results = {}

        results["baseline"] = self.run_baseline()
        results["flat_prices"] = self.run_flat_prices()
        results["price_scale"] = self.run_price_scale()
        results["perfect_efficiency"] = self.run_perfect_efficiency()
        results["low_efficiency"] = self.run_low_efficiency()
        results["tight_power_limits"] = self.run_tight_power_limits()
        results["loose_power_limits"] = self.run_loose_power_limits()
        results["small_capacity"] = self.run_small_capacity()
        results["large_capacity"] = self.run_large_capacity()

        # Create histogram comparing gaps across scenarios
        fig, ax = plt.subplots(figsize=(12, 6))

        scenarios_list = list(results.keys())
        x = range(len(scenarios_list))
        width = 0.35

        basic_gaps = [results[s]["basic_gap"] for s in scenarios_list]
        tighter_gaps = [results[s]["tighter_gap"] for s in scenarios_list]

        ax.bar([i - width / 2 for i in x], basic_gaps, width, label="Basic Gap")
        ax.bar([i + width / 2 for i in x], tighter_gaps, width, label="Tighter Gap")

        ax.set_xlabel("Scenario")
        ax.set_ylabel("Gap")
        ax.set_title("LP-MIP Gaps by Scenario and Formulation")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios_list, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            "outputs/assignment4/gaps_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        results_df = pd.DataFrame(results).T
        output_path = Path("outputs/assignment4") / "assignment4_scenarios_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index_label="scenario")
        return results_df

    def run_baseline(self) -> dict:
        # baseline
        basic_mip = solve_model(Basic, self.data, PARAMS)
        basic_lp, basic_lp_fractional_nodes = solve_lp_model(Basic, self.data, PARAMS)
        basic_gap = basic_lp - basic_mip
        tighter_mip = solve_model(Tighter, self.data, PARAMS)
        tighter_lp, tighter_lp_fractional_nodes = solve_lp_model(
            Tighter, self.data, PARAMS
        )
        tighter_gap = tighter_lp - tighter_mip

        return {
            "basic_mip": basic_mip,
            "basic_lp": basic_lp,
            "basic_gap": basic_gap,
            "basic_lp_fractional_nodes": basic_lp_fractional_nodes,
            "tighter_mip": tighter_mip,
            "tighter_lp": tighter_lp,
            "tighter_gap": tighter_gap,
            "tighter_lp_fractional_nodes": tighter_lp_fractional_nodes,
        }

    def run_flat_prices(self) -> dict:
        flat_prices_data = scenarios.flat_prices(self.data.copy())
        flat_prices_basicmip = solve_model(Basic, flat_prices_data, PARAMS)
        flat_prices_basiclp, flat_prices_basic_fractional_nodes = solve_lp_model(
            Basic, flat_prices_data, PARAMS
        )
        flat_prices_basicgap = flat_prices_basiclp - flat_prices_basicmip
        flat_prices_tightermip = solve_model(Tighter, flat_prices_data, PARAMS)
        flat_prices_tighterlp, flat_prices_tighter_fractional_nodes = solve_lp_model(
            Tighter, flat_prices_data, PARAMS
        )
        flat_prices_tightergap = flat_prices_tighterlp - flat_prices_tightermip

        return {
            "basic_mip": flat_prices_basicmip,
            "basic_lp": flat_prices_basiclp,
            "basic_gap": flat_prices_basicgap,
            "basic_lp_fractional_nodes": flat_prices_basic_fractional_nodes,
            "tighter_mip": flat_prices_tightermip,
            "tighter_lp": flat_prices_tighterlp,
            "tighter_gap": flat_prices_tightergap,
            "tighter_lp_fractional_nodes": flat_prices_tighter_fractional_nodes,
        }

    def run_price_scale(self) -> dict:
        price_scale_data = scenarios.price_scale(
            self.data.copy(), scale=self.price_scale
        )
        price_scale_basicmip = solve_model(Basic, price_scale_data, PARAMS)
        price_scale_basiclp, price_scale_basic_fractional_nodes = solve_lp_model(
            Basic, price_scale_data, PARAMS
        )
        price_scale_basicgap = price_scale_basiclp - price_scale_basicmip
        price_scale_tightermip = solve_model(Tighter, price_scale_data, PARAMS)
        price_scale_tighterlp, price_scale_tighter_fractional_nodes = solve_lp_model(
            Tighter, price_scale_data, PARAMS
        )
        price_scale_tightergap = price_scale_tighterlp - price_scale_tightermip

        return {
            "basic_mip": price_scale_basicmip,
            "basic_lp": price_scale_basiclp,
            "basic_gap": price_scale_basicgap,
            "basic_lp_fractional_nodes": price_scale_basic_fractional_nodes,
            "tighter_mip": price_scale_tightermip,
            "tighter_lp": price_scale_tighterlp,
            "tighter_gap": price_scale_tightergap,
            "tighter_lp_fractional_nodes": price_scale_tighter_fractional_nodes,
        }

    def run_perfect_efficiency(self) -> dict:
        perfect_efficiency_params = scenarios.perfect_efficiency(PARAMS.copy())
        perfect_efficiency_basicmip = solve_model(
            Basic, self.data.copy(), perfect_efficiency_params
        )
        perfect_efficiency_basiclp, perfect_efficiency_basic_fractional_nodes = (
            solve_lp_model(Basic, self.data.copy(), perfect_efficiency_params)
        )
        perfect_efficiency_basicgap = (
            perfect_efficiency_basiclp - perfect_efficiency_basicmip
        )
        perfect_efficiency_tightermip = solve_model(
            Tighter, self.data.copy(), perfect_efficiency_params
        )
        perfect_efficiency_tighterlp, perfect_efficiency_tighter_fractional_nodes = (
            solve_lp_model(Tighter, self.data.copy(), perfect_efficiency_params)
        )
        perfect_efficiency_tightergap = (
            perfect_efficiency_tighterlp - perfect_efficiency_tightermip
        )

        return {
            "basic_mip": perfect_efficiency_basicmip,
            "basic_lp": perfect_efficiency_basiclp,
            "basic_gap": perfect_efficiency_basicgap,
            "basic_lp_fractional_nodes": perfect_efficiency_basic_fractional_nodes,
            "tighter_mip": perfect_efficiency_tightermip,
            "tighter_lp": perfect_efficiency_tighterlp,
            "tighter_gap": perfect_efficiency_tightergap,
            "tighter_lp_fractional_nodes": perfect_efficiency_tighter_fractional_nodes,
        }

    def run_low_efficiency(self) -> dict:
        low_efficiency_params = scenarios.low_efficiency(PARAMS.copy())
        low_efficiency_basicmip = solve_model(
            Basic, self.data.copy(), low_efficiency_params
        )
        low_efficiency_basiclp, low_efficiency_basic_fractional_nodes = solve_lp_model(
            Basic, self.data.copy(), low_efficiency_params
        )
        low_efficiency_basicgap = low_efficiency_basiclp - low_efficiency_basicmip
        low_efficiency_tightermip = solve_model(
            Tighter, self.data.copy(), low_efficiency_params
        )
        low_efficiency_tighterlp, low_efficiency_tighter_fractional_nodes = (
            solve_lp_model(Tighter, self.data.copy(), low_efficiency_params)
        )
        low_efficiency_tightergap = low_efficiency_tighterlp - low_efficiency_tightermip

        return {
            "basic_mip": low_efficiency_basicmip,
            "basic_lp": low_efficiency_basiclp,
            "basic_gap": low_efficiency_basicgap,
            "basic_lp_fractional_nodes": low_efficiency_basic_fractional_nodes,
            "tighter_mip": low_efficiency_tightermip,
            "tighter_lp": low_efficiency_tighterlp,
            "tighter_gap": low_efficiency_tightergap,
            "tighter_lp_fractional_nodes": low_efficiency_tighter_fractional_nodes,
        }

    def run_tight_power_limits(self) -> dict:
        tight_power_limits_params = scenarios.tight_power_limits(PARAMS.copy())
        tight_power_limits_basicmip = solve_model(
            Basic, self.data.copy(), tight_power_limits_params
        )
        tight_power_limits_basiclp, tight_power_limits_basic_fractional_nodes = (
            solve_lp_model(Basic, self.data.copy(), tight_power_limits_params)
        )
        tight_power_limits_basicgap = (
            tight_power_limits_basiclp - tight_power_limits_basicmip
        )
        tight_power_limits_tightermip = solve_model(
            Tighter, self.data.copy(), tight_power_limits_params
        )
        tight_power_limits_tighterlp, tight_power_limits_tighter_fractional_nodes = (
            solve_lp_model(Tighter, self.data.copy(), tight_power_limits_params)
        )
        tight_power_limits_tightergap = (
            tight_power_limits_tighterlp - tight_power_limits_tightermip
        )

        return {
            "basic_mip": tight_power_limits_basicmip,
            "basic_lp": tight_power_limits_basiclp,
            "basic_gap": tight_power_limits_basicgap,
            "basic_lp_fractional_nodes": tight_power_limits_basic_fractional_nodes,
            "tighter_mip": tight_power_limits_tightermip,
            "tighter_lp": tight_power_limits_tighterlp,
            "tighter_gap": tight_power_limits_tightergap,
            "tighter_lp_fractional_nodes": tight_power_limits_tighter_fractional_nodes,
        }

    def run_loose_power_limits(self) -> dict:
        loose_power_limits_params = scenarios.loose_power_limits(PARAMS.copy())
        loose_power_limits_basicmip = solve_model(
            Basic, self.data.copy(), loose_power_limits_params
        )
        loose_power_limits_basiclp, loose_power_limits_basic_fractional_nodes = (
            solve_lp_model(Basic, self.data.copy(), loose_power_limits_params)
        )
        loose_power_limits_basicgap = (
            loose_power_limits_basiclp - loose_power_limits_basicmip
        )
        loose_power_limits_tightermip = solve_model(
            Tighter, self.data.copy(), loose_power_limits_params
        )
        loose_power_limits_tighterlp, loose_power_limits_tighter_fractional_nodes = (
            solve_lp_model(Tighter, self.data.copy(), loose_power_limits_params)
        )
        loose_power_limits_tightergap = (
            loose_power_limits_tighterlp - loose_power_limits_tightermip
        )

        return {
            "basic_mip": loose_power_limits_basicmip,
            "basic_lp": loose_power_limits_basiclp,
            "basic_gap": loose_power_limits_basicgap,
            "basic_lp_fractional_nodes": loose_power_limits_basic_fractional_nodes,
            "tighter_mip": loose_power_limits_tightermip,
            "tighter_lp": loose_power_limits_tighterlp,
            "tighter_gap": loose_power_limits_tightergap,
            "tighter_lp_fractional_nodes": loose_power_limits_tighter_fractional_nodes,
        }

    def run_small_capacity(self) -> dict:
        small_capacity_params = scenarios.small_capacity(PARAMS.copy())
        small_capacity_basicmip = solve_model(
            Basic, self.data.copy(), small_capacity_params
        )
        small_capacity_basiclp, small_capacity_basic_fractional_nodes = solve_lp_model(
            Basic, self.data.copy(), small_capacity_params
        )
        small_capacity_basicgap = small_capacity_basiclp - small_capacity_basicmip
        small_capacity_tightermip = solve_model(
            Tighter, self.data.copy(), small_capacity_params
        )
        small_capacity_tighterlp, small_capacity_tighter_fractional_nodes = (
            solve_lp_model(Tighter, self.data.copy(), small_capacity_params)
        )
        small_capacity_tightergap = small_capacity_tighterlp - small_capacity_tightermip

        return {
            "basic_mip": small_capacity_basicmip,
            "basic_lp": small_capacity_basiclp,
            "basic_gap": small_capacity_basicgap,
            "basic_lp_fractional_nodes": small_capacity_basic_fractional_nodes,
            "tighter_mip": small_capacity_tightermip,
            "tighter_lp": small_capacity_tighterlp,
            "tighter_gap": small_capacity_tightergap,
            "tighter_lp_fractional_nodes": small_capacity_tighter_fractional_nodes,
        }

    def run_large_capacity(self) -> dict:
        large_capacity_params = scenarios.large_capacity(PARAMS.copy())
        large_capacity_basicmip = solve_model(
            Basic, self.data.copy(), large_capacity_params
        )
        large_capacity_basiclp, large_capacity_basic_fractional_nodes = solve_lp_model(
            Basic, self.data.copy(), large_capacity_params
        )
        large_capacity_basicgap = large_capacity_basiclp - large_capacity_basicmip
        large_capacity_tightermip = solve_model(
            Tighter, self.data.copy(), large_capacity_params
        )
        large_capacity_tighterlp, large_capacity_tighter_fractional_nodes = (
            solve_lp_model(Tighter, self.data.copy(), large_capacity_params)
        )
        large_capacity_tightergap = large_capacity_tighterlp - large_capacity_tightermip

        return {
            "basic_mip": large_capacity_basicmip,
            "basic_lp": large_capacity_basiclp,
            "basic_gap": large_capacity_basicgap,
            "basic_lp_fractional_nodes": large_capacity_basic_fractional_nodes,
            "tighter_mip": large_capacity_tightermip,
            "tighter_lp": large_capacity_tighterlp,
            "tighter_gap": large_capacity_tightergap,
            "tighter_lp_fractional_nodes": large_capacity_tighter_fractional_nodes,
        }
