from __future__ import annotations

import pandas as pd

from ..config import CaseConfig, Formulation
from .common import ModelArtifacts, build_common_model, previous_soc_value


def build_model(
    case_config: CaseConfig,
    data: pd.DataFrame,
    *,
    relax_binaries: bool = False,
) -> ModelArtifacts:
    artifacts = build_common_model(
        case_config,
        data,
        Formulation.TIGHTER,
        relax_binaries=relax_binaries,
    )
    battery = case_config.battery
    dt = artifacts.time_step_hours

    for t in range(len(data)):
        previous_soc = previous_soc_value(artifacts, t, battery.initial_soc_kwh)
        artifacts.problem += (
            previous_soc >= battery.soc_min_kwh + dt * artifacts.discharge_kw[t] / battery.discharge_efficiency,
            f"tight_discharge_soc_{t}",
        )
        artifacts.problem += (
            previous_soc <= battery.soc_max_kwh - battery.charge_efficiency * dt * artifacts.charge_kw[t],
            f"tight_charge_soc_{t}",
        )

    return artifacts
