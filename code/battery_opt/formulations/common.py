from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pulp

from ..config import CaseConfig, Formulation


@dataclass
class ModelArtifacts:
    problem: pulp.LpProblem
    formulation: Formulation
    data: pd.DataFrame
    time_step_hours: float
    soc: dict
    charge_kw: dict
    discharge_kw: dict
    mode: dict
    net_grid_kwh: dict


def build_common_model(
    case_config: CaseConfig,
    data: pd.DataFrame,
    formulation: Formulation,
    *,
    relax_binaries: bool,
) -> ModelArtifacts:
    battery = case_config.battery
    dt = case_config.time_step_hours or _infer_dt(data)
    periods = range(len(data))

    problem = pulp.LpProblem(f"{case_config.name}_{formulation.value}", pulp.LpMaximize)

    soc = pulp.LpVariable.dicts("soc_kwh", periods, lowBound=0.0)
    charge_kw = pulp.LpVariable.dicts("charge_kw", periods, lowBound=0.0)
    discharge_kw = pulp.LpVariable.dicts("discharge_kw", periods, lowBound=0.0)
    net_grid_kwh = pulp.LpVariable.dicts("net_grid_kwh", periods, lowBound=None)
    mode_cat = pulp.LpContinuous if relax_binaries else pulp.LpBinary
    mode = pulp.LpVariable.dicts("mode", periods, lowBound=0.0, upBound=1.0, cat=mode_cat)

    fee_penalty = 0
    if battery.grid_fee_eur_per_kwh > 0.0:
        grid_import_kwh = pulp.LpVariable.dicts("grid_import_kwh", periods, lowBound=0.0)
        grid_export_kwh = pulp.LpVariable.dicts("grid_export_kwh", periods, lowBound=0.0)
        for t in periods:
            problem += (
                net_grid_kwh[t] == grid_import_kwh[t] - grid_export_kwh[t],
                f"net_grid_split_{t}",
            )
        fee_penalty = pulp.lpSum(
            battery.grid_fee_eur_per_kwh * (grid_import_kwh[t] + grid_export_kwh[t])
            for t in periods
        )

    for t in periods:
        previous_soc = battery.initial_soc_kwh if t == 0 else soc[t - 1]
        problem += (
            soc[t]
            == previous_soc
            + battery.charge_efficiency * charge_kw[t] * dt
            - discharge_kw[t] * dt / battery.discharge_efficiency,
            f"soc_balance_{t}",
        )
        problem += (soc[t] >= battery.soc_min_kwh, f"soc_min_{t}")
        problem += (soc[t] <= battery.soc_max_kwh, f"soc_max_{t}")
        problem += (
            charge_kw[t] <= battery.charge_power_limit_kw * mode[t],
            f"charge_limit_{t}",
        )
        problem += (
            discharge_kw[t] <= battery.discharge_power_limit_kw * (1.0 - mode[t]),
            f"discharge_limit_{t}",
        )
        problem += (
            net_grid_kwh[t]
            == float(data.loc[t, "net_demand_kwh"]) + charge_kw[t] * dt - discharge_kw[t] * dt,
            f"grid_balance_{t}",
        )

    degradation_penalty = pulp.lpSum(
        battery.degradation_cost_eur_per_kwh_throughput * (charge_kw[t] + discharge_kw[t]) * dt
        for t in periods
    )
    energy_profit = pulp.lpSum(
        -float(data.loc[t, "price_eur_per_kwh"]) * net_grid_kwh[t]
        for t in periods
    )
    problem += energy_profit - fee_penalty - degradation_penalty

    return ModelArtifacts(
        problem=problem,
        formulation=formulation,
        data=data.copy(),
        time_step_hours=dt,
        soc=soc,
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        mode=mode,
        net_grid_kwh=net_grid_kwh,
    )


def previous_soc_value(artifacts: ModelArtifacts, t: int, initial_soc_kwh: float) -> Any:
    return initial_soc_kwh if t == 0 else artifacts.soc[t - 1]


def _infer_dt(data: pd.DataFrame) -> float:
    delta = data.loc[0, "end"] - data.loc[0, "start"]
    return delta.total_seconds() / 3600.0
