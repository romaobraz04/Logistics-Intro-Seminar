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


def negative_price_fractional_mode_share(
    formulation, negative_price_count: int, tol=1e-6
):
    if negative_price_count == 0:
        return 0.0

    negative_price_fractional_count = sum(
        1
        for t in range(formulation.periods)
        if formulation.price.iloc[t] < 0
        and formulation.dec_vars[f"mode_{t}"].varValue is not None
        and tol < formulation.dec_vars[f"mode_{t}"].varValue < 1 - tol
    )
    return negative_price_fractional_count / negative_price_count


def solve_lp_model(formulation_cls, data, params, negative_price_count: int):
    model = pulp.LpProblem(
        f"{formulation_cls.__name__}_LP",
        pulp.LpMaximize,
    )
    formulation = formulation_cls(model, data.copy(), params.copy(), lp=True, msg=False)
    formulation.run_model()
    return (
        pulp.value(model.objective),
        count_fractional_nodes(formulation),
        negative_price_fractional_mode_share(formulation, negative_price_count),
    )


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
        self.negative_price_count = int((self.data["Price (EUR/kWh)"] < 0).sum())

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

        ax.bar(
            [i - width / 2 for i in x],
            basic_gaps,
            width,
            label="Basic Gap",
            hatch="//",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x],
            tighter_gaps,
            width,
            label="Tighter Gap",
            hatch="xx",
            edgecolor="black",
            linewidth=0.8,
        )

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
        return self._run_scenario(self.data, PARAMS)

    def run_flat_prices(self) -> dict:
        flat_prices_data = scenarios.flat_prices(self.data.copy())
        return self._run_scenario(flat_prices_data, PARAMS)

    def run_price_scale(self) -> dict:
        price_scale_data = scenarios.price_scale(
            self.data.copy(), scale=self.price_scale
        )
        return self._run_scenario(price_scale_data, PARAMS)

    def run_perfect_efficiency(self) -> dict:
        perfect_efficiency_params = scenarios.perfect_efficiency(PARAMS.copy())
        return self._run_scenario(self.data.copy(), perfect_efficiency_params)

    def run_low_efficiency(self) -> dict:
        low_efficiency_params = scenarios.low_efficiency(PARAMS.copy())
        return self._run_scenario(self.data.copy(), low_efficiency_params)

    def run_tight_power_limits(self) -> dict:
        tight_power_limits_params = scenarios.tight_power_limits(PARAMS.copy())
        return self._run_scenario(self.data.copy(), tight_power_limits_params)

    def run_loose_power_limits(self) -> dict:
        loose_power_limits_params = scenarios.loose_power_limits(PARAMS.copy())
        return self._run_scenario(self.data.copy(), loose_power_limits_params)

    def run_small_capacity(self) -> dict:
        small_capacity_params = scenarios.small_capacity(PARAMS.copy())
        return self._run_scenario(self.data.copy(), small_capacity_params)

    def run_large_capacity(self) -> dict:
        large_capacity_params = scenarios.large_capacity(PARAMS.copy())
        return self._run_scenario(self.data.copy(), large_capacity_params)

    def _run_scenario(self, data: pd.DataFrame, params: dict) -> dict:
        basic_mip = solve_model(Basic, data, params)
        (
            basic_lp,
            basic_lp_fractional_nodes,
            basic_negative_price_fractional_share,
        ) = solve_lp_model(Basic, data, params, self.negative_price_count)
        tighter_mip = solve_model(Tighter, data, params)
        (
            tighter_lp,
            tighter_lp_fractional_nodes,
            tighter_negative_price_fractional_share,
        ) = solve_lp_model(Tighter, data, params, self.negative_price_count)

        return {
            "basic_mip": basic_mip,
            "basic_lp": basic_lp,
            "basic_gap": basic_lp - basic_mip,
            "basic_lp_fractional_nodes": basic_lp_fractional_nodes,
            "basic_negative_price_fractional_share": basic_negative_price_fractional_share,
            "tighter_mip": tighter_mip,
            "tighter_lp": tighter_lp,
            "tighter_gap": tighter_lp - tighter_mip,
            "tighter_lp_fractional_nodes": tighter_lp_fractional_nodes,
            "tighter_negative_price_fractional_share": tighter_negative_price_fractional_share,
        }
