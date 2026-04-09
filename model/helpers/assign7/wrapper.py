from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

from formulations.basic import Basic
from formulations.params import PARAMS
from formulations.tighter import Tighter
from helpers.metrics import equivalent_full_cycles
from helpers.assign5.wrapper import FORMULATION_ORDER
from helpers.assign6.wrapper import LINE_PLOT_STYLES, _line_plot_offsets
from helpers.assign7 import scenarios as sc

FORMULATIONS = {"basic": Basic, "tighter": Tighter}

SOC_STYLES = {
    "perfect_foresight":    {"color": "black",      "linestyle": "-",  "linewidth": 2.5, "zorder": 6},
    "weekly":               {"color": "tab:blue",   "linestyle": "-",  "linewidth": 2,   "marker": "o", "markersize": 4, "markevery": 4, "zorder": 5},
    "two_day":              {"color": "tab:orange",  "linestyle": "--", "linewidth": 2,   "marker": "s", "markersize": 4, "markevery": 4, "zorder": 4},
    "day_ahead":            {"color": "tab:green",  "linestyle": "-.", "linewidth": 2,   "marker": "^", "markersize": 4, "markevery": 4, "zorder": 3},
    "epex_day_ahead":       {"color": "tab:red",    "linestyle": ":",  "linewidth": 2,   "marker": "v", "markersize": 4, "markevery": 4, "zorder": 3},
    "day_ahead_naive":      {"color": "tab:purple", "linestyle": "--", "linewidth": 2,   "marker": "D", "markersize": 4, "markevery": 6, "zorder": 2},
    "epex_day_ahead_naive": {"color": "tab:brown",  "linestyle": "-.", "linewidth": 2,   "marker": "x", "markersize": 5, "markevery": 6, "zorder": 2},
}


class assign7wrapper:
    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str | Path = "outputs/assignment7",
    ):
        self.data = data.copy().reset_index(drop=True)
        self.output_dir = Path(output_dir)
        self.S0 = PARAMS["init_soc"]

    def run(self) -> dict[str, pd.DataFrame]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for formulation_name, formulation_cls in FORMULATIONS.items():
            print(f"[{formulation_name}] perfect foresight ...")
            records.append(
                self.run_perfect_foresight(formulation_name, formulation_cls)
            )

            print(f"[{formulation_name}] weekly ...")
            records.append(self.run_weekly(formulation_name, formulation_cls))

            print(f"[{formulation_name}] two-day ...")
            records.append(self.run_two_day(formulation_name, formulation_cls))

            print(f"[{formulation_name}] day-ahead ...")
            records.append(self.run_day_ahead(formulation_name, formulation_cls))

            print(f"[{formulation_name}] EPEX day-ahead ...")
            records.append(self.run_epex_day_ahead(formulation_name, formulation_cls))

            print(f"[{formulation_name}] two-day soft valuation ...")
            records.append(self.run_two_day_soft_val(formulation_name, formulation_cls))

            print(f"[{formulation_name}] day-ahead naïve ...")
            records.append(self.run_day_ahead_naive(formulation_name, formulation_cls))

            print(f"[{formulation_name}] EPEX day-ahead naïve ...")
            records.append(
                self.run_epex_day_ahead_naive(formulation_name, formulation_cls)
            )

        results = pd.DataFrame(records)

        # VPI
        perfect = (
            results[results["scenario"] == "perfect_foresight"]
            .set_index("formulation")[["total_profit_eur", "throughput_kwh"]]
            .rename(
                columns={
                    "total_profit_eur": "perfect_profit_eur",
                    "throughput_kwh": "perfect_throughput_kwh",
                }
            )
        )
        results = results.join(perfect, on="formulation")
        results["vpi_eur"] = results["perfect_profit_eur"] - results["total_profit_eur"]
        results["vpi_pct"] = (
            results["vpi_eur"] / results["perfect_profit_eur"].abs() * 100
        )

        # VPI decomposition
        results["extra_throughput_kwh"] = (
            results["throughput_kwh"] - results["perfect_throughput_kwh"]
        ).clip(lower=0)
        results["vpi_degradation_eur"] = (
            sc.DEGRADATION_COST * results["extra_throughput_kwh"]
        )
        results["vpi_arbitrage_eur"] = (
            results["vpi_eur"] - results["vpi_degradation_eur"]
        )

        # Sensitivity analyses
        print("Running grid-fee sensitivity ...")
        fee_df = self._run_fee_sensitivity()

        print("Running degradation-cost sensitivity ...")
        deg_df = self._run_deg_sensitivity()

        # Export
        export = results.drop(
            columns=["perfect_profit_eur", "perfect_throughput_kwh"], errors="ignore"
        )
        export.to_csv(self.output_dir / "assignment7_rolling_horizon.csv", index=False)
        fee_df.to_csv(self.output_dir / "assignment7_fee_sensitivity.csv", index=False)
        deg_df.to_csv(self.output_dir / "assignment7_deg_sensitivity.csv", index=False)

        # SoC profiles
        print("Collecting SoC profiles ...")
        soc_profiles = self._collect_soc_profiles()

        # Plots
        self._plot_profit(results)
        self._plot_vpi(results)
        self._plot_soc_profile(soc_profiles)
        self._plot_solve_time(results)
        self._plot_fee_sensitivity(fee_df)
        self._plot_deg_sensitivity(deg_df)

        return {
            "assignment7_rolling_horizon": export,
            "assignment7_fee_sensitivity": fee_df,
            "assignment7_deg_sensitivity": deg_df,
        }

    def run_perfect_foresight(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_perfect_foresight(
            formulation_cls, sc.GRID_FEE, sc.DEGRADATION_COST
        )
        return self._make_record(sc.PERFECT_FORESIGHT, formulation_name, metrics)

    def run_weekly(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_rolling(formulation_cls, **self._rh_kwargs(sc.WEEKLY))
        return self._make_record(sc.WEEKLY, formulation_name, metrics)

    def run_two_day(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_rolling(formulation_cls, **self._rh_kwargs(sc.TWO_DAY))
        return self._make_record(sc.TWO_DAY, formulation_name, metrics)

    def run_day_ahead(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_rolling(formulation_cls, **self._rh_kwargs(sc.DAY_AHEAD))
        return self._make_record(sc.DAY_AHEAD, formulation_name, metrics)

    def run_epex_day_ahead(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_rolling(
            formulation_cls, **self._rh_kwargs(sc.EPEX_DAY_AHEAD)
        )
        return self._make_record(sc.EPEX_DAY_AHEAD, formulation_name, metrics)

    def run_two_day_soft_val(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_rolling(
            formulation_cls, **self._rh_kwargs(sc.TWO_DAY_SOFT_VAL)
        )
        return self._make_record(sc.TWO_DAY_SOFT_VAL, formulation_name, metrics)

    def run_day_ahead_naive(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_rolling(
            formulation_cls, **self._rh_kwargs(sc.DAY_AHEAD_NAIVE)
        )
        return self._make_record(sc.DAY_AHEAD_NAIVE, formulation_name, metrics)

    def run_epex_day_ahead_naive(self, formulation_name: str, formulation_cls) -> dict:
        metrics = self._run_rolling(
            formulation_cls, **self._rh_kwargs(sc.EPEX_DAY_AHEAD_NAIVE)
        )
        return self._make_record(sc.EPEX_DAY_AHEAD_NAIVE, formulation_name, metrics)

    def _make_record(
        self, scenario: dict, formulation_name: str, metrics: dict
    ) -> dict:
        """Flatten a scenario dict + solver metrics into one DataFrame row."""
        return {
            "formulation": formulation_name,
            "scenario": scenario["name"],
            "terminal_strategy": scenario["terminal"] or "hard_floor",
            "window": scenario["window"] or len(self.data),
            "step": scenario["step"] or len(self.data),
            "epoch_offset": scenario["epoch_offset"],
            **{k: v for k, v in metrics.items() if k != "soc_profile"},
        }

    def _rh_kwargs(self, scenario: dict) -> dict:
        """Extract rolling-horizon keyword arguments from a scenario dict."""
        return {
            "window": scenario["window"],
            "step": scenario["step"],
            "terminal_strategy": scenario["terminal"],
            "epoch_offset": scenario["epoch_offset"],
            "grid_fee": sc.GRID_FEE,
            "deg_cost": sc.DEGRADATION_COST,
            "forecast_lag": scenario.get("forecast_lag", 0),
        }

    def _solve_window(
        self,
        formulation_cls,
        window_data: pd.DataFrame,
        initial_soc: float,
        terminal_strategy: str | None,
        grid_fee: float,
        deg_cost: float,
        label: str = "window",
        realized_prices: np.ndarray | None = None,
    ) -> dict:
        """Solve one optimisation window; return per-period details."""
        params = PARAMS.copy()
        params["init_soc"] = initial_soc

        model = pulp.LpProblem(label, pulp.LpMaximize)
        f = formulation_cls(
            model,
            window_data.copy().reset_index(drop=True),
            params,
            lp=False,
            grid_fee_eur_kwh=grid_fee,
            degradation_cost_eur_per_kwh_discharge=deg_cost,
            msg=False,
        )
        f.create_variables()
        f.objective_function()
        f.add_constraints()

        n = f.periods

        if terminal_strategy == "hard_floor":
            model += (f.dec_vars[f"soc_{n}"] >= self.S0, "Terminal_SOC_Floor")
        elif terminal_strategy == "soft_valuation":
            mean_price = window_data["Price (EUR/kWh)"].mean()
            model.objective += mean_price * f.dec_vars[f"soc_{n}"]

        status_code = model.solve(pulp.HiGHS(msg=False))

        dec_vars = {
            k: float(v.varValue) if v.varValue is not None else 0.0
            for k, v in f.dec_vars.items()
        }

        prices = (
            realized_prices
            if realized_prices is not None
            else window_data["Price (EUR/kWh)"].reset_index(drop=True).values
        )
        period_profits = [
            f.delta
            * prices[t]
            * (
                dec_vars.get(f"electricity_sell_{t}", 0.0)
                - dec_vars.get(f"electricity_buy_{t}", 0.0)
            )
            - f.delta
            * grid_fee
            * (
                dec_vars.get(f"electricity_sell_{t}", 0.0)
                + dec_vars.get(f"electricity_buy_{t}", 0.0)
            )
            - f.delta * deg_cost * dec_vars.get(f"discharge_power_{t}", 0.0)
            for t in range(min(n, len(prices)))
        ]

        return {
            "status": str(pulp.LpStatus[status_code]).upper(),
            "runtime": model.solutionTime,
            "period_profits": period_profits,
            "soc_profile": [dec_vars.get(f"soc_{t}", 0.0) for t in range(n + 1)],
            "discharge_per_period": [
                dec_vars.get(f"discharge_power_{t}", 0.0) for t in range(n)
            ],
            "buy_per_period": [
                dec_vars.get(f"electricity_buy_{t}", 0.0) for t in range(n)
            ],
            "sell_per_period": [
                dec_vars.get(f"electricity_sell_{t}", 0.0) for t in range(n)
            ],
            "dec_vars": dec_vars,
            "delta": f.delta,
            "params": params,
        }

    def _run_perfect_foresight(
        self, formulation_cls, grid_fee: float, deg_cost: float
    ) -> dict:
        sol = self._solve_window(
            formulation_cls,
            self.data,
            initial_soc=self.S0,
            terminal_strategy="hard_floor",
            grid_fee=grid_fee,
            deg_cost=deg_cost,
            label="perfect_foresight",
        )
        throughput = sum(sol["discharge_per_period"]) * sol["delta"]
        return {
            "total_profit_eur": sum(sol["period_profits"]),
            "total_runtime_seconds": sol["runtime"],
            "n_subproblems": 1,
            "throughput_kwh": throughput,
            "import_kwh": sum(sol["buy_per_period"]) * sol["delta"],
            "export_kwh": sum(sol["sell_per_period"]) * sol["delta"],
            "equivalent_cycles": equivalent_full_cycles(
                sol["dec_vars"], sol["delta"], PARAMS["min_soc"], PARAMS["max_soc"]
            ),
            "terminal_soc_kwh": sol["soc_profile"][-1],
            "soc_profile": sol["soc_profile"],
        }

    def _run_rolling(
        self,
        formulation_cls,
        window: int,
        step: int,
        terminal_strategy: str,
        epoch_offset: int,
        grid_fee: float,
        deg_cost: float,
        forecast_lag: int = 0,
    ) -> dict:
        T = len(self.data)
        hat_s = self.S0
        total_profit = 0.0
        total_runtime = 0.0
        n_subproblems = 0
        all_discharge, all_buy, all_sell = [], [], []
        full_soc = [hat_s]

        # Pre-epoch window (EPEX only): handle hours 0 → epoch_offset-1.
        # Skipped when forecast_lag > 0 to avoid index underflow on lagged prices.
        if epoch_offset > 0 and forecast_lag == 0:
            sol = self._solve_window(
                formulation_cls,
                self.data.iloc[:epoch_offset],
                hat_s,
                terminal_strategy="hard_floor",
                grid_fee=grid_fee,
                deg_cost=deg_cost,
                label="pre_epoch",
            )
            total_profit += sum(sol["period_profits"])
            total_runtime += sol["runtime"]
            n_subproblems += 1
            all_discharge.extend(sol["discharge_per_period"])
            all_buy.extend(sol["buy_per_period"])
            all_sell.extend(sol["sell_per_period"])
            full_soc.extend(sol["soc_profile"][1:])
            hat_s = sol["soc_profile"][-1]

        # First valid epoch: when forecast_lag > 0, skip the initial hours where
        # no prior-week data exists.
        t = max(epoch_offset, forecast_lag)
        while t < T:
            n_impl = min(step, T - t)
            window_data = self.data.iloc[t : t + window]

            if forecast_lag == 0:
                forecast_data = window_data
                realized_prices = None
            else:
                lag_start = t - forecast_lag
                lagged = self.data.iloc[lag_start : lag_start + window].reset_index(
                    drop=True
                )
                # True day-ahead prices (known) + naïve net demand (same hour last week).
                # Decisions optimised against (d_{t-168}, λ_t); profit evaluated at (d_t, λ_t).
                forecast_data = window_data.copy().reset_index(drop=True)
                forecast_data["Volume (kWh)"] = lagged["Volume (kWh)"].values[
                    : len(forecast_data)
                ]
                realized_prices = None

            sol = self._solve_window(
                formulation_cls,
                forecast_data,
                hat_s,
                terminal_strategy,
                grid_fee=grid_fee,
                deg_cost=deg_cost,
                label=f"rh_t{t}",
                realized_prices=realized_prices,
            )

            total_profit += sum(sol["period_profits"][:n_impl])
            total_runtime += sol["runtime"]
            n_subproblems += 1
            all_discharge.extend(sol["discharge_per_period"][:n_impl])
            all_buy.extend(sol["buy_per_period"][:n_impl])
            all_sell.extend(sol["sell_per_period"][:n_impl])
            full_soc.extend(sol["soc_profile"][1 : n_impl + 1])
            hat_s = sol["soc_profile"][n_impl]
            t += step

        throughput = sum(all_discharge)
        usable = PARAMS["max_soc"] - PARAMS["min_soc"]
        return {
            "total_profit_eur": total_profit,
            "total_runtime_seconds": total_runtime,
            "n_subproblems": n_subproblems,
            "throughput_kwh": throughput,
            "import_kwh": sum(all_buy),
            "export_kwh": sum(all_sell),
            "equivalent_cycles": throughput / usable if usable > 0 else 0.0,
            "terminal_soc_kwh": full_soc[-1] if full_soc else self.S0,
            "soc_profile": full_soc,
        }

    def _collect_soc_profiles(self) -> dict[str, list]:
        profiles = {}
        for scenario in sc.ALL_SCENARIOS:
            if scenario["name"] == "two_day_soft_val":
                continue
            if scenario["window"] is None:
                metrics = self._run_perfect_foresight(
                    Basic, sc.GRID_FEE, sc.DEGRADATION_COST
                )
            else:
                metrics = self._run_rolling(Basic, **self._rh_kwargs(scenario))
            profiles[scenario["name"]] = metrics["soc_profile"]
        return profiles

    def _run_fee_sensitivity(self) -> pd.DataFrame:
        records = []
        for fee in sc.GRID_FEE_LEVELS:
            for formulation_name, formulation_cls in FORMULATIONS.items():
                pf = self._run_perfect_foresight(
                    formulation_cls, fee, sc.DEGRADATION_COST
                )
                da = self._run_rolling(
                    formulation_cls,
                    window=sc.DAY_AHEAD["window"],
                    step=sc.DAY_AHEAD["step"],
                    terminal_strategy=sc.DAY_AHEAD["terminal"],
                    epoch_offset=sc.DAY_AHEAD["epoch_offset"],
                    grid_fee=fee,
                    deg_cost=sc.DEGRADATION_COST,
                )
                vpi_eur = pf["total_profit_eur"] - da["total_profit_eur"]
                records.append(
                    {
                        "formulation": formulation_name,
                        "grid_fee_eur_per_kwh": fee,
                        "perfect_profit_eur": pf["total_profit_eur"],
                        "day_ahead_profit_eur": da["total_profit_eur"],
                        "perfect_eq_cycles": pf["equivalent_cycles"],
                        "day_ahead_eq_cycles": da["equivalent_cycles"],
                        "perfect_grid_exchange_kwh": pf["import_kwh"]
                        + pf["export_kwh"],
                        "day_ahead_grid_exchange_kwh": da["import_kwh"]
                        + da["export_kwh"],
                        "vpi_eur": vpi_eur,
                        "vpi_pct": vpi_eur / abs(pf["total_profit_eur"]) * 100,
                    }
                )
        return pd.DataFrame(records)

    def _run_deg_sensitivity(self) -> pd.DataFrame:
        records = []
        for deg in sc.DEG_LEVELS:
            for formulation_name, formulation_cls in FORMULATIONS.items():
                pf = self._run_perfect_foresight(formulation_cls, sc.GRID_FEE, deg)
                da = self._run_rolling(
                    formulation_cls,
                    window=sc.DAY_AHEAD["window"],
                    step=sc.DAY_AHEAD["step"],
                    terminal_strategy=sc.DAY_AHEAD["terminal"],
                    epoch_offset=sc.DAY_AHEAD["epoch_offset"],
                    grid_fee=sc.GRID_FEE,
                    deg_cost=deg,
                )
                vpi_eur = pf["total_profit_eur"] - da["total_profit_eur"]
                records.append(
                    {
                        "formulation": formulation_name,
                        "degradation_cost_eur_per_kwh": deg,
                        "perfect_profit_eur": pf["total_profit_eur"],
                        "day_ahead_profit_eur": da["total_profit_eur"],
                        "perfect_eq_cycles": pf["equivalent_cycles"],
                        "day_ahead_eq_cycles": da["equivalent_cycles"],
                        "vpi_eur": vpi_eur,
                        "vpi_pct": vpi_eur / abs(pf["total_profit_eur"]) * 100,
                    }
                )
        return pd.DataFrame(records)

    def _plot_profit(self, results: pd.DataFrame) -> None:
        """Figure 1a: total profit by scenario × formulation."""
        main = results[results["scenario"].isin(sc.MAIN_SCENARIO_NAMES)].copy()
        pivot = main.pivot(
            index="scenario", columns="formulation", values="total_profit_eur"
        )
        pivot = pivot.reindex([s for s in sc.MAIN_SCENARIO_NAMES if s in pivot.index])
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        hatches = ["//", "xx"]
        for container, hatch in zip(ax.containers, hatches):
            for patch in container:
                patch.set_hatch(hatch)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
        ax.set_title(
            "Total Profit by Foresight Scenario\n"
            "(negative values = net cost; mandatory household load)"
        )
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Profit (EUR)")
        ax.set_xticklabels([s.replace("_", "\n") for s in pivot.index], rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="formulation")
        fig.tight_layout()
        fig.savefig(self.output_dir / "assignment7_profit_comparison.png", dpi=300)
        plt.close(fig)

    def _plot_vpi(self, results: pd.DataFrame) -> None:
        """Figure 1b: VPI (EUR) by scenario × formulation."""
        rh_only = results[
            results["scenario"].isin(sc.MAIN_SCENARIO_NAMES)
            & (results["scenario"] != "perfect_foresight")
        ].copy()
        order = [s for s in sc.MAIN_SCENARIO_NAMES if s != "perfect_foresight"]
        pivot = rh_only.pivot(index="scenario", columns="formulation", values="vpi_eur")
        pivot = pivot.reindex([s for s in order if s in pivot.index])
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        hatches = ["//", "xx"]
        for container, hatch in zip(ax.containers, hatches):
            for patch in container:
                patch.set_hatch(hatch)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
        ax.set_title("Value of Perfect Information (VPI)")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("VPI (EUR)")
        ax.set_xticklabels([s.replace("_", "\n") for s in pivot.index], rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="formulation")
        fig.tight_layout()
        fig.savefig(self.output_dir / "assignment7_vpi.png", dpi=300)
        plt.close(fig)

    def _plot_soc_profile(self, profiles: dict[str, list]) -> None:
        """Figure 2b: average daily SoC profile by scenario."""
        fig, ax = plt.subplots(figsize=(13, 5))
        hours = np.arange(24)
        for scenario_name, soc in profiles.items():
            if len(soc) < 24:
                continue
            soc_arr = np.array(soc[:-1])
            n_days = len(soc_arr) // 24
            if n_days == 0:
                continue
            avg = soc_arr[: n_days * 24].reshape(n_days, 24).mean(axis=0)
            style = SOC_STYLES.get(scenario_name, {"linewidth": 2})
            ax.plot(hours, avg, label=scenario_name.replace("_", " "), **style)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average SoC (kWh)")
        ax.set_title("Average Daily SoC Profile by Foresight Scenario")
        ax.set_xticks(hours)
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig.tight_layout()
        fig.savefig(self.output_dir / "assignment7_soc_profile.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_solve_time(self, results: pd.DataFrame) -> None:
        """Figure 4: total solve time by scenario × formulation."""
        main = results[results["scenario"].isin(sc.MAIN_SCENARIO_NAMES)].copy()
        pivot = main.pivot(
            index="scenario", columns="formulation", values="total_runtime_seconds"
        )
        pivot = pivot.reindex([s for s in sc.MAIN_SCENARIO_NAMES if s in pivot.index])
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        hatches = ["//", "xx"]
        for container, hatch in zip(ax.containers, hatches):
            for patch in container:
                patch.set_hatch(hatch)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
        ax.set_title("Total Solve Time by Foresight Scenario")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticklabels([s.replace("_", "\n") for s in pivot.index], rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=7, padding=2)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="formulation")
        fig.tight_layout()
        fig.savefig(self.output_dir / "assignment7_solve_time.png", dpi=300)
        plt.close(fig)

    def _plot_fee_sensitivity(self, fee_df: pd.DataFrame) -> None:
        """Figure 3a: day-ahead VPI vs grid fee (two panels: EUR and %)."""
        formulation_names = [f for f in FORMULATION_ORDER if f in fee_df["formulation"].unique()]
        x_offsets = _line_plot_offsets(fee_df["grid_fee_eur_per_kwh"], formulation_names)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for formulation_name in formulation_names:
            sub = fee_df[fee_df["formulation"] == formulation_name].sort_values(
                "grid_fee_eur_per_kwh"
            )
            label = formulation_name.replace("_", " ")
            style = LINE_PLOT_STYLES.get(formulation_name, {})
            x = sub["grid_fee_eur_per_kwh"] + x_offsets.get(formulation_name, 0.0)
            ax1.plot(x, sub["vpi_eur"], label=label, **style)
            ax2.plot(x, sub["vpi_pct"], label=label, **style)
        for ax, ylabel in [(ax1, "VPI (EUR)"), (ax2, "VPI (%)")]:
            ax.set_xlabel("Grid Fee (EUR/kWh)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(sc.GRID_FEE_LEVELS)
            ax.set_axisbelow(True)
            ax.legend()
            ax.grid(True, alpha=0.3)
        ax1.set_title("Day-Ahead VPI vs Grid Fee Level")
        ax2.set_title("VPI (%) vs Grid Fee")
        fig.suptitle(
            "Day-Ahead VPI across Grid Fee Levels (Assignment 5 Integration)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(self.output_dir / "assignment7_vpi_vs_grid_fee.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_deg_sensitivity(self, deg_df: pd.DataFrame) -> None:
        """Figure 3b: day-ahead VPI vs degradation cost (two panels: EUR and %)."""
        formulation_names = [f for f in FORMULATION_ORDER if f in deg_df["formulation"].unique()]
        x_offsets = _line_plot_offsets(deg_df["degradation_cost_eur_per_kwh"], formulation_names)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for formulation_name in formulation_names:
            sub = deg_df[deg_df["formulation"] == formulation_name].sort_values(
                "degradation_cost_eur_per_kwh"
            )
            label = formulation_name.replace("_", " ")
            style = LINE_PLOT_STYLES.get(formulation_name, {})
            x = sub["degradation_cost_eur_per_kwh"] + x_offsets.get(formulation_name, 0.0)
            ax1.plot(x, sub["vpi_eur"], label=label, **style)
            ax2.plot(x, sub["vpi_pct"], label=label, **style)
        for ax, ylabel in [(ax1, "VPI (EUR)"), (ax2, "VPI (%)")]:
            ax.set_xlabel("Degradation Cost (EUR/kWh throughput)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(sc.DEG_LEVELS)
            ax.set_axisbelow(True)
            ax.legend()
            ax.grid(True, alpha=0.3)
        ax1.set_title("Day-Ahead VPI vs Degradation Cost")
        ax2.set_title("VPI (%) vs Degradation Cost")
        fig.suptitle(
            "Day-Ahead VPI across Degradation Cost Levels (Assignment 6 Integration)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "assignment7_vpi_vs_degradation_cost.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
