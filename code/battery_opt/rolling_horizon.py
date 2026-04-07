from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import pandas as pd
import pulp

from .config import BatteryConfig, CaseConfig, Formulation
from .formulations import build_basic_model, build_tighter_model
from .formulations.common import ModelArtifacts
from .metrics import summarize_schedule

BUILDERS = {
    Formulation.BASIC: build_basic_model,
    Formulation.TIGHTER: build_tighter_model,
}


def solve_rolling_horizon(
    case_config: CaseConfig,
    formulation: Formulation | str,
    data: pd.DataFrame,
    window: int,
    step: int,
    terminal_strategy: str = "hard_floor",
    first_window: int | None = None,
    first_step: int | None = None,
) -> dict[str, Any]:
    formulation = Formulation(formulation)
    battery = case_config.battery
    T = len(data)
    dt = case_config.time_step_hours or 1.0

    implemented_rows: list[dict] = []
    current_soc = battery.initial_soc_kwh
    total_solve_time = 0.0
    n_subproblems = 0
    is_first = True

    t_start = 0
    while t_start < T:
        cur_window = first_window if (is_first and first_window is not None) else window
        cur_step = first_step if (is_first and first_step is not None) else step
        t_end = min(t_start + cur_window, T)
        window_data = data.iloc[t_start:t_end].copy().reset_index(drop=True)
        W_len = len(window_data)

        window_config = replace(
            case_config,
            battery=replace(battery, initial_soc_kwh=current_soc),
        )

        artifacts = BUILDERS[formulation](window_config, window_data, relax_binaries=False)

        if terminal_strategy == "hard_floor":
            last_idx = W_len - 1
            artifacts.problem += (
                artifacts.soc[last_idx] >= battery.initial_soc_kwh,
                "terminal_floor",
            )
        elif terminal_strategy == "soft_valuation":
            mean_price = float(window_data["price_eur_per_kwh"].mean())
            last_idx = W_len - 1
            artifacts.problem.objective += mean_price * artifacts.soc[last_idx]

        solver = pulp.HiGHS(
            msg=0,
            timeLimit=case_config.solver.time_limit_seconds,
            gapRel=case_config.solver.mip_gap,
            threads=case_config.solver.threads,
        )
        t0 = time.perf_counter()
        artifacts.problem.solve(solver)
        elapsed = time.perf_counter() - t0
        total_solve_time += elapsed
        n_subproblems += 1

        if artifacts.problem.status != pulp.constants.LpStatusOptimal:
            break

        implement_end = min(cur_step, W_len)
        for t_local in range(implement_end):
            t_global = t_start + t_local
            row = {
                "start": data.loc[t_global, "start"],
                "end": data.loc[t_global, "end"],
                "net_demand_kwh": float(data.loc[t_global, "net_demand_kwh"]),
                "price_eur_per_kwh": float(data.loc[t_global, "price_eur_per_kwh"]),
                "charge_kw": artifacts.charge_kw[t_local].varValue,
                "discharge_kw": artifacts.discharge_kw[t_local].varValue,
                "soc_kwh": artifacts.soc[t_local].varValue,
                "mode": artifacts.mode[t_local].varValue,
                "net_grid_kwh": artifacts.net_grid_kwh[t_local].varValue,
            }
            implemented_rows.append(row)

        last_implemented = implement_end - 1
        current_soc = artifacts.soc[last_implemented].varValue
        t_start += cur_step
        is_first = False

    schedule = pd.DataFrame(implemented_rows)
    if not schedule.empty:
        schedule["charge_kwh"] = schedule["charge_kw"] * dt
        schedule["discharge_kwh"] = schedule["discharge_kw"] * dt
        schedule["grid_import_kwh"] = schedule["net_grid_kwh"].clip(lower=0.0)
        schedule["grid_export_kwh"] = (-schedule["net_grid_kwh"]).clip(lower=0.0)

    summary = summarize_schedule(schedule, battery)

    prices = schedule["price_eur_per_kwh"] if not schedule.empty else pd.Series(dtype=float)
    net_grid = schedule["net_grid_kwh"] if not schedule.empty else pd.Series(dtype=float)
    total_profit = float((-prices * net_grid).sum()) if not schedule.empty else 0.0

    grid_fee = battery.grid_fee_eur_per_kwh
    if grid_fee > 0.0 and not schedule.empty:
        total_profit -= grid_fee * summary["total_grid_exchange_kwh"]

    deg_cost = battery.degradation_cost_eur_per_kwh_throughput
    if deg_cost > 0.0 and not schedule.empty:
        total_profit -= deg_cost * float(
            (schedule["charge_kwh"] + schedule["discharge_kwh"]).sum()
        )

    return {
        "schedule": schedule,
        "summary": summary,
        "total_profit_eur": total_profit,
        "total_solve_time_seconds": total_solve_time,
        "n_subproblems": n_subproblems,
        "periods_implemented": len(schedule),
    }
