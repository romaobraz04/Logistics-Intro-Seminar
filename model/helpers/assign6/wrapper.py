from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pulp

from formulations.basic import Basic
from formulations.params import PARAMS
from formulations.tighter import Tighter
from helpers.metrics import equivalent_full_cycles
from helpers.assign5.wrapper import FORMULATION_ORDER, _sum_prefixed_variables

DEG_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03]
FORMULATIONS = {
    "basic": Basic,
    "tighter": Tighter,
}


class assign6wrapper:
    def __init__(
        self,
        data: pd.DataFrame,
        deg_levels: list[float] | None = None,
        output_dir: str | Path = "outputs/assignment6",
    ):
        self.data = data.copy().reset_index(drop=True)
        self.deg_levels = (
            deg_levels.copy() if deg_levels is not None else DEG_LEVELS.copy()
        )
        self.output_dir = Path(output_dir)

    def run(self) -> pd.DataFrame:
        records = []
        for deg in self.deg_levels:
            for formulation_name, formulation_cls in FORMULATIONS.items():
                records.append(
                    self._solve_case(
                        formulation_name=formulation_name,
                        formulation_cls=formulation_cls,
                        params=PARAMS.copy(),
                        degradation_cost_eur_per_kwh_discharge=deg,
                    )
                )
        results_df = pd.DataFrame(records)
        results_df = results_df.sort_values(
            by=["degradation_cost_eur_per_kwh_discharge", "formulation"],
            key=lambda series: (
                series.map({name: idx for idx, name in enumerate(FORMULATION_ORDER)})
                if series.name == "formulation"
                else series
            ),
        ).reset_index(drop=True)
        export_df = results_df.drop(columns=["_dec_vars", "_delta"], errors="ignore")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(
            self.output_dir / "assignment6_degradation_cost_sweep.csv", index=False
        )

        self._plot_metric(
            export_df,
            metric="objective_eur",
            ylabel="Objective (EUR)",
            title="Objective by Degradation Cost",
            filename="assignment6_objective_by_degradation_cost.png",
        )

        self._plot_metric(
            export_df,
            metric="equivalent_full_cycles",
            ylabel="Equivalent Full Cycles",
            title="Equivalent Full Cycles by Degradation Cost",
            filename="assignment6_equivalent_full_cycles_by_degradation_cost.png",
        )

        return export_df

    def _solve_case(
        self,
        formulation_name: str,
        formulation_cls,
        params: dict,
        degradation_cost_eur_per_kwh_discharge: float,
    ) -> dict:
        model = pulp.LpProblem(
            f"{formulation_name}_deg_{degradation_cost_eur_per_kwh_discharge:.3f}",
            pulp.LpMaximize,
        )
        formulation = formulation_cls(
            model,
            self.data.copy(),
            params.copy(),
            lp=False,
            degradation_cost_eur_per_kwh_discharge=degradation_cost_eur_per_kwh_discharge,
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
            "degradation_cost_eur_per_kwh_discharge": degradation_cost_eur_per_kwh_discharge,
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
                    "total_charge_kwh": pd.NA,
                    "total_discharge_kwh": pd.NA,
                    "equivalent_full_cycles": pd.NA,
                }
            )
            return record

        record.update(
            {
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
            }
        )
        return record

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

        formulations = formulations if formulations is not None else FORMULATION_ORDER
        fig, ax = plt.subplots(figsize=(9, 5))
        for formulation_name in formulations:
            subset = valid[valid["formulation"] == formulation_name].sort_values(
                "degradation_cost_eur_per_kwh_discharge"
            )
            if subset.empty:
                continue

            ax.plot(
                subset["degradation_cost_eur_per_kwh_discharge"],
                subset[metric],
                marker="o",
                linewidth=2,
                label=formulation_name.replace("_", " "),
            )

        ax.set_xlabel("Grid Fee (EUR/kWh)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(self.deg_levels)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
