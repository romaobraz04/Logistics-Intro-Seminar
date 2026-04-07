from __future__ import annotations

import time
from typing import Callable

import highspy
import pandas as pd
import pulp

from .config import CaseConfig, Formulation
from .data import load_timeseries
from .formulations import build_basic_model, build_tighter_model
from .formulations.common import ModelArtifacts
from .metrics import summarize_schedule
from .results import OptimizationResult

BUILDERS: dict[Formulation, Callable[..., ModelArtifacts]] = {
    Formulation.BASIC: build_basic_model,
    Formulation.TIGHTER: build_tighter_model,
}

STATUS_NAMES = {
    pulp.constants.LpStatusOptimal: "OPTIMAL",
    pulp.constants.LpStatusNotSolved: "NOT_SOLVED",
    pulp.constants.LpStatusInfeasible: "INFEASIBLE",
    pulp.constants.LpStatusUnbounded: "UNBOUNDED",
    pulp.constants.LpStatusUndefined: "UNDEFINED",
}


def solve_case(
    case_config: CaseConfig,
    formulation: Formulation | str,
    *,
    relax_binaries: bool = False,
    prepared_data: pd.DataFrame | None = None,
    terminal_soc_kwh: float | None = None,
) -> OptimizationResult:
    formulation = Formulation(formulation)
    if prepared_data is None:
        data, inferred_dt = load_timeseries(case_config.data)
    else:
        data = prepared_data.copy().reset_index(drop=True)
        inferred_dt = case_config.time_step_hours
        if inferred_dt is None:
            delta = data.loc[0, "end"] - data.loc[0, "start"]
            inferred_dt = delta.total_seconds() / 3600.0

    effective_config = case_config if case_config.time_step_hours else _with_time_step(case_config, inferred_dt)
    artifacts = BUILDERS[formulation](effective_config, data, relax_binaries=relax_binaries)

    if terminal_soc_kwh is not None:
        last_t = len(data) - 1
        artifacts.problem += (
            artifacts.soc[last_t] >= terminal_soc_kwh,
            "terminal_soc_floor",
        )

    solver = pulp.HiGHS(
        msg=0,
        timeLimit=effective_config.solver.time_limit_seconds,
        gapRel=effective_config.solver.mip_gap,
        threads=effective_config.solver.threads,
    )
    start_time = time.perf_counter()
    artifacts.problem.solve(solver)
    elapsed = time.perf_counter() - start_time

    schedule = _extract_schedule(artifacts)
    summary = summarize_schedule(schedule, effective_config.battery)
    summary["time_step_hours"] = artifacts.time_step_hours

    objective = pulp.value(artifacts.problem.objective)
    mip_gap, best_bound, node_count = _extract_highs_stats(artifacts.problem)

    return OptimizationResult(
        formulation=formulation.value,
        relaxed=relax_binaries,
        status_code=artifacts.problem.status,
        status_name=STATUS_NAMES.get(artifacts.problem.status, str(artifacts.problem.status)),
        objective_value_eur=objective,
        runtime_seconds=elapsed,
        mip_gap=mip_gap,
        best_bound=best_bound,
        node_count=node_count,
        schedule=schedule,
        summary=summary,
    )


def _extract_highs_stats(
    problem: pulp.LpProblem,
) -> tuple[float | None, float | None, float | None]:
    try:
        sol = problem.solverModel
        if sol is None:
            return None, None, None
        info = sol.getInfoValue
        mip_gap = info("mip_gap")[1]
        best_bound = info("mip_dual_bound")[1]
        node_count = float(info("mip_node_count")[1])
        return mip_gap, best_bound, node_count
    except Exception:
        return None, None, None


def _extract_schedule(artifacts: ModelArtifacts) -> pd.DataFrame:
    if artifacts.problem.status != pulp.constants.LpStatusOptimal:
        return pd.DataFrame()

    df = artifacts.data.copy()
    dt = artifacts.time_step_hours
    df["charge_kw"] = [artifacts.charge_kw[t].varValue for t in range(len(df))]
    df["discharge_kw"] = [artifacts.discharge_kw[t].varValue for t in range(len(df))]
    df["soc_kwh"] = [artifacts.soc[t].varValue for t in range(len(df))]
    df["mode"] = [artifacts.mode[t].varValue for t in range(len(df))]
    df["net_grid_kwh"] = [artifacts.net_grid_kwh[t].varValue for t in range(len(df))]
    df["charge_kwh"] = df["charge_kw"] * dt
    df["discharge_kwh"] = df["discharge_kw"] * dt
    df["grid_import_kwh"] = df["net_grid_kwh"].clip(lower=0.0)
    df["grid_export_kwh"] = (-df["net_grid_kwh"]).clip(lower=0.0)
    return df


def _with_time_step(case_config: CaseConfig, time_step_hours: float) -> CaseConfig:
    return CaseConfig(
        name=case_config.name,
        data=case_config.data,
        battery=case_config.battery,
        solver=case_config.solver,
        time_step_hours=time_step_hours,
    )
