from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from ..config import CaseConfig, DatasetScope, Formulation, FractionWindowConfig, ScenarioOverrideConfig, WindowPosition
from ..data import apply_timeseries_modifiers, load_timeseries
from ..solve import solve_case


@dataclass(frozen=True)
class DataWindow:
    label: str
    fraction: float
    position: WindowPosition
    start_index: int
    end_index: int
    data: pd.DataFrame


def load_analysis_data(case_config: CaseConfig, dataset_scope: DatasetScope | str) -> pd.DataFrame:
    scope = DatasetScope(dataset_scope)
    if scope == DatasetScope.FULL_DATASET:
        data_config = replace(case_config.data, periods=None)
    else:
        data_config = case_config.data
    data, _ = load_timeseries(data_config)
    return data


def build_fraction_windows(
    data: pd.DataFrame,
    horizon_windows: list[FractionWindowConfig] | tuple[FractionWindowConfig, ...],
) -> list[DataWindow]:
    windows: list[DataWindow] = []
    seen: set[tuple[int, int]] = set()
    total_periods = len(data)

    for window in horizon_windows:
        base_periods = max(1, int(round(total_periods * window.fraction)))
        if window.max_periods is not None:
            base_periods = min(base_periods, window.max_periods)
        base_periods = min(base_periods, total_periods)

        for position in window.positions:
            start_index = _window_start_index(total_periods, base_periods, position)
            end_index = start_index + base_periods
            dedupe_key = (start_index, end_index)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            label = f"{window.fraction:.0%}_{position.value}"
            windows.append(
                DataWindow(
                    label=label,
                    fraction=window.fraction,
                    position=position,
                    start_index=start_index,
                    end_index=end_index,
                    data=data.iloc[start_index:end_index].copy().reset_index(drop=True),
                )
            )

    return windows


def apply_scenario_overrides(
    case_config: CaseConfig,
    data: pd.DataFrame,
    overrides: ScenarioOverrideConfig,
) -> tuple[CaseConfig, pd.DataFrame]:
    updated_case = replace(
        case_config,
        battery=replace(
            case_config.battery,
            initial_soc_kwh=(
                overrides.initial_soc_kwh
                if overrides.initial_soc_kwh is not None
                else case_config.battery.initial_soc_kwh
            ),
            soc_min_kwh=(
                overrides.soc_min_kwh
                if overrides.soc_min_kwh is not None
                else case_config.battery.soc_min_kwh
            ),
            soc_max_kwh=(
                overrides.soc_max_kwh
                if overrides.soc_max_kwh is not None
                else case_config.battery.soc_max_kwh
            ),
            charge_power_limit_kw=(
                overrides.charge_power_limit_kw
                if overrides.charge_power_limit_kw is not None
                else case_config.battery.charge_power_limit_kw
            ),
            discharge_power_limit_kw=(
                overrides.discharge_power_limit_kw
                if overrides.discharge_power_limit_kw is not None
                else case_config.battery.discharge_power_limit_kw
            ),
            charge_efficiency=(
                overrides.charge_efficiency
                if overrides.charge_efficiency is not None
                else case_config.battery.charge_efficiency
            ),
            discharge_efficiency=(
                overrides.discharge_efficiency
                if overrides.discharge_efficiency is not None
                else case_config.battery.discharge_efficiency
            ),
            grid_fee_eur_per_kwh=(
                overrides.grid_fee_eur_per_kwh
                if overrides.grid_fee_eur_per_kwh is not None
                else case_config.battery.grid_fee_eur_per_kwh
            ),
            degradation_cost_eur_per_kwh_throughput=(
                overrides.degradation_cost_eur_per_kwh_throughput
                if overrides.degradation_cost_eur_per_kwh_throughput is not None
                else case_config.battery.degradation_cost_eur_per_kwh_throughput
            ),
        ),
    )
    updated_data = apply_timeseries_modifiers(
        data,
        price_scale=overrides.price_scale,
        price_shift=overrides.price_shift,
        flatten_prices=overrides.flatten_prices,
        demand_scale=overrides.demand_scale,
        demand_shift=overrides.demand_shift,
    )
    return updated_case, updated_data


def safe_solve(case_config: CaseConfig, formulation: Formulation, **kwargs):
    try:
        return solve_case(case_config, formulation, **kwargs), None
    except Exception as exc:
        return None, f"SOLVER_ERROR: {exc}"


def window_for_fraction(
    data: pd.DataFrame,
    fraction: float,
    position: WindowPosition | str,
) -> DataWindow:
    position = WindowPosition(position)
    periods = max(1, int(round(len(data) * fraction)))
    periods = min(periods, len(data))
    start_index = _window_start_index(len(data), periods, position)
    end_index = start_index + periods
    return DataWindow(
        label=f"{fraction:.0%}_{position.value}",
        fraction=fraction,
        position=position,
        start_index=start_index,
        end_index=end_index,
        data=data.iloc[start_index:end_index].copy().reset_index(drop=True),
    )


def _window_start_index(total_periods: int, window_periods: int, position: WindowPosition) -> int:
    if position == WindowPosition.START:
        return 0
    if position == WindowPosition.MIDDLE:
        return max(0, (total_periods - window_periods) // 2)
    if position == WindowPosition.END:
        return max(0, total_periods - window_periods)
    raise ValueError(f"Unsupported window position: {position}")
