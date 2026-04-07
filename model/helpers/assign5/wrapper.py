from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pulp

from formulations.basic import Basic
from formulations.params import PARAMS
from formulations.tighter import Tighter

from ..metrics import (
    behavioural_cost,
    equivalent_full_cycles,
    self_comsumption_rate,
    total_grid_exchange,
)

FEE_LEVELS = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
FORMULATIONS = {
    "basic": Basic,
    "tighter": Tighter,
}
FORMULATION_ORDER = ["no_battery", "basic", "tighter"]


def _sum_prefixed_variables(dec_vars: dict, prefix: str, delta: float) -> float:
    total = sum(value for key, value in dec_vars.items() if key.startswith(prefix))
    return total * delta


class assign5wrapper:
    def __init__(
        self,
        data: pd.DataFrame,
        fee_levels: list[float] | None = None,
        output_dir: str | Path = "outputs/assignment5",
    ):
        self.data = data.copy().reset_index(drop=True)
        self.fee_levels = (
            fee_levels.copy() if fee_levels is not None else FEE_LEVELS.copy()
        )
        self.output_dir = Path(output_dir)

    def run(self) -> dict[str, pd.DataFrame]:
        records = []
        for fee in self.fee_levels:
            for formulation_name, formulation_cls in FORMULATIONS.items():
                records.append(
                    self._solve_case(
                        formulation_name=formulation_name,
                        formulation_cls=formulation_cls,
                        params=PARAMS.copy(),
                        grid_fee_eur_kwh=fee,
                    )
                )

            records.append(
                self._solve_case(
                    formulation_name="no_battery",
                    formulation_cls=Basic,
                    params=self._no_battery_params(),
                    grid_fee_eur_kwh=fee,
                )
            )

        results_df = pd.DataFrame(records)
        results_df = self._add_behavioural_cost(results_df)
        results_df = results_df.sort_values(
            by=["grid_fee_eur_per_kwh", "formulation"],
            key=lambda series: (
                series.map({name: idx for idx, name in enumerate(FORMULATION_ORDER)})
                if series.name == "formulation"
                else series
            ),
        ).reset_index(drop=True)

        export_df = results_df.drop(columns=["_dec_vars", "_delta"], errors="ignore")
        summary_df = self._build_summary(export_df)
        incremental_value_df = self._build_battery_incremental_value(summary_df)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(
            self.output_dir / "assignment5_grid_fee_sweep.csv", index=False
        )
        summary_df.to_csv(
            self.output_dir / "assignment5_grid_fee_summary.csv", index=False
        )
        incremental_value_df.to_csv(
            self.output_dir / "assignment5_battery_incremental_value.csv", index=False
        )

        self._plot_metric(
            export_df,
            metric="objective_eur",
            ylabel="Objective (EUR)",
            title="Objective by Grid Fee",
            filename="assignment5_objective_by_fee.png",
        )
        self._plot_metric(
            export_df,
            metric="total_grid_exchange_kwh",
            ylabel="Grid Exchange (kWh)",
            title="Grid Exchange by Grid Fee",
            filename="assignment5_grid_exchange_by_fee.png",
        )
        self._plot_metric(
            export_df,
            metric="self_consumption_rate",
            ylabel="Self-Consumption Rate",
            title="Self-Consumption Rate by Grid Fee",
            filename="assignment5_self_consumption_by_fee.png",
        )
        self._plot_metric(
            export_df,
            metric="equivalent_full_cycles",
            ylabel="Equivalent Full Cycles",
            title="Equivalent Full Cycles by Grid Fee",
            filename="assignment5_equivalent_full_cycles_by_fee.png",
        )
        self._plot_metric(
            export_df,
            metric="behavioural_cost_eur",
            ylabel="Behavioural Cost (EUR)",
            title="Behavioural Cost by Grid Fee",
            filename="assignment5_behavioural_cost_by_fee.png",
        )
        self._plot_metric(
            incremental_value_df,
            metric="battery_incremental_value_eur",
            ylabel="Battery Incremental Value (EUR)",
            title="Battery Incremental Value by Grid Fee",
            filename="assignment5_battery_incremental_value_by_fee.png",
            formulations=list(FORMULATIONS.keys()),
        )

        return {
            "assignment5_grid_fee_sweep": export_df,
            "assignment5_grid_fee_summary": summary_df,
            "assignment5_battery_incremental_value": incremental_value_df,
        }

    def _solve_case(
        self,
        formulation_name: str,
        formulation_cls,
        params: dict,
        grid_fee_eur_kwh: float,
    ) -> dict:
        model = pulp.LpProblem(
            f"{formulation_name}_fee_{grid_fee_eur_kwh:.3f}",
            pulp.LpMaximize,
        )
        formulation = formulation_cls(
            model,
            self.data.copy(),
            params.copy(),
            lp=False,
            grid_fee_eur_kwh=grid_fee_eur_kwh,
            msg=False,
        )
        status_code, objective_value, runtime_seconds, _ = formulation.run_model()
        status_name = str(pulp.LpStatus[status_code]).upper()
        dec_vars = {
            var_name: float(var.varValue) if var.varValue is not None else 0.0
            for var_name, var in formulation.dec_vars.items()
        }

        record = {
            "formulation": formulation_name,
            "grid_fee_eur_per_kwh": grid_fee_eur_kwh,
            "status": status_name,
            "periods": formulation.periods,
            "objective_eur": (
                float(objective_value) if objective_value is not None else pd.NA
            ),
            "runtime_seconds": (
                float(runtime_seconds) if runtime_seconds is not None else pd.NA
            ),
            "_dec_vars": dec_vars,
            "_delta": formulation.delta,
        }

        if status_name != "OPTIMAL":
            record.update(
                {
                    "total_grid_fee_paid_eur": pd.NA,
                    "total_grid_exchange_kwh": pd.NA,
                    "total_charge_kwh": pd.NA,
                    "total_discharge_kwh": pd.NA,
                    "equivalent_full_cycles": pd.NA,
                    "self_consumption_rate": pd.NA,
                }
            )
            return record

        total_exchange = total_grid_exchange(dec_vars, formulation.delta)
        record.update(
            {
                "total_grid_fee_paid_eur": grid_fee_eur_kwh * total_exchange,
                "total_grid_exchange_kwh": total_exchange,
                "total_charge_kwh": _sum_prefixed_variables(
                    dec_vars, "charge_power_", formulation.delta
                ),
                "total_discharge_kwh": _sum_prefixed_variables(
                    dec_vars, "discharge_power_", formulation.delta
                ),
                "equivalent_full_cycles": equivalent_full_cycles(
                    dec_vars,
                    formulation.delta,
                    params["min_soc"],
                    params["max_soc"],
                ),
                "self_consumption_rate": self_comsumption_rate(
                    dec_vars,
                    formulation.delta,
                    formulation.net_demand,
                ),
            }
        )
        return record

    def _add_behavioural_cost(self, results_df: pd.DataFrame) -> pd.DataFrame:
        baseline_fee = min(self.fee_levels)
        optimal_results = results_df[results_df["status"] == "OPTIMAL"]
        baseline_objectives = (
            optimal_results[optimal_results["grid_fee_eur_per_kwh"] == baseline_fee]
            .set_index("formulation")["objective_eur"]
            .to_dict()
        )

        behavioural_costs = []
        for _, row in results_df.iterrows():
            if row["status"] != "OPTIMAL":
                behavioural_costs.append(pd.NA)
                continue

            baseline_objective = baseline_objectives.get(row["formulation"])
            if baseline_objective is None:
                behavioural_costs.append(pd.NA)
                continue

            behavioural_costs.append(
                behavioural_cost(
                    row["_dec_vars"],
                    row["_delta"],
                    baseline_objective,
                    row["objective_eur"],
                    row["grid_fee_eur_per_kwh"],
                )
            )

        results_df["behavioural_cost_eur"] = behavioural_costs
        return results_df

    def _build_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        valid = results_df[results_df["status"] == "OPTIMAL"].copy()
        if valid.empty:
            return pd.DataFrame()

        baseline_fee = min(self.fee_levels)
        zero_fee = valid[valid["grid_fee_eur_per_kwh"] == baseline_fee][
            [
                "formulation",
                "objective_eur",
                "total_grid_exchange_kwh",
                "self_consumption_rate",
            ]
        ].rename(
            columns={
                "objective_eur": "zero_fee_objective_eur",
                "total_grid_exchange_kwh": "zero_fee_grid_exchange_kwh",
                "self_consumption_rate": "zero_fee_self_consumption_rate",
            }
        )
        no_battery = valid[valid["formulation"] == "no_battery"][
            [
                "grid_fee_eur_per_kwh",
                "objective_eur",
                "total_grid_exchange_kwh",
                "self_consumption_rate",
            ]
        ].rename(
            columns={
                "objective_eur": "no_battery_objective_eur",
                "total_grid_exchange_kwh": "no_battery_grid_exchange_kwh",
                "self_consumption_rate": "no_battery_self_consumption_rate",
            }
        )

        summary = valid.merge(zero_fee, on="formulation", how="left")
        summary = summary.merge(no_battery, on="grid_fee_eur_per_kwh", how="left")
        summary["objective_change_from_zero_fee_eur"] = (
            summary["objective_eur"] - summary["zero_fee_objective_eur"]
        )
        summary["grid_exchange_change_from_zero_fee_kwh"] = (
            summary["total_grid_exchange_kwh"] - summary["zero_fee_grid_exchange_kwh"]
        )
        summary["self_consumption_change_from_zero_fee"] = (
            summary["self_consumption_rate"] - summary["zero_fee_self_consumption_rate"]
        )
        summary["objective_gain_vs_no_battery_eur"] = (
            summary["objective_eur"] - summary["no_battery_objective_eur"]
        )
        summary["grid_exchange_reduction_vs_no_battery_kwh"] = (
            summary["no_battery_grid_exchange_kwh"] - summary["total_grid_exchange_kwh"]
        )
        summary["self_consumption_gain_vs_no_battery"] = (
            summary["self_consumption_rate"]
            - summary["no_battery_self_consumption_rate"]
        )
        summary["battery_incremental_value_eur"] = (
            summary["objective_eur"] - summary["no_battery_objective_eur"]
        )
        summary["objective_gain_vs_no_battery_eur"] = summary[
            "battery_incremental_value_eur"
        ]
        return summary.sort_values(
            by=["grid_fee_eur_per_kwh", "formulation"],
            key=lambda series: (
                series.map({name: idx for idx, name in enumerate(FORMULATION_ORDER)})
                if series.name == "formulation"
                else series
            ),
        ).reset_index(drop=True)

    def _build_battery_incremental_value(
        self, summary_df: pd.DataFrame
    ) -> pd.DataFrame:
        if summary_df.empty:
            return pd.DataFrame(
                columns=[
                    "grid_fee_eur_per_kwh",
                    "formulation",
                    "objective_eur",
                    "no_battery_objective_eur",
                    "battery_incremental_value_eur",
                ]
            )

        incremental_value_df = summary_df[summary_df["formulation"] != "no_battery"][
            [
                "grid_fee_eur_per_kwh",
                "formulation",
                "objective_eur",
                "no_battery_objective_eur",
                "battery_incremental_value_eur",
            ]
        ].copy()
        return incremental_value_df.reset_index(drop=True)

    def _plot_metric(
        self,
        results_df: pd.DataFrame,
        metric: str,
        ylabel: str,
        title: str,
        filename: str,
        formulations: list[str] | None = None,
    ) -> None:
        valid = (
            results_df[
                (results_df["status"] == "OPTIMAL") & results_df[metric].notna()
            ].copy()
            if "status" in results_df.columns
            else results_df[results_df[metric].notna()].copy()
        )
        if valid.empty:
            return

        formulations = (
            formulations if formulations is not None else list(FORMULATIONS.keys())
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        for formulation_name in formulations:
            subset = valid[valid["formulation"] == formulation_name].sort_values(
                "grid_fee_eur_per_kwh"
            )
            if subset.empty:
                continue

            ax.plot(
                subset["grid_fee_eur_per_kwh"],
                subset[metric],
                marker="o",
                linewidth=2,
                label=formulation_name.replace("_", " "),
            )

        ax.set_xlabel("Grid Fee (EUR/kWh)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(self.fee_levels)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _no_battery_params() -> dict:
        params = PARAMS.copy()
        params["init_soc"] = 0.0
        params["min_soc"] = 0.0
        params["max_soc"] = 0.0
        return params
