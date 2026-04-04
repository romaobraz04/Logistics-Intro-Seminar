from __future__ import annotations

from typing import Any

import pandas as pd

from .config import BatteryConfig


def summarize_schedule(schedule: pd.DataFrame, battery: BatteryConfig) -> dict[str, Any]:
    if schedule.empty:
        return {
            "total_charge_kwh": 0.0,
            "total_discharge_kwh": 0.0,
            "throughput_kwh": 0.0,
            "total_import_kwh": 0.0,
            "total_export_kwh": 0.0,
            "total_grid_exchange_kwh": 0.0,
            "average_soc_kwh": 0.0,
            "average_soc_utilization": 0.0,
            "local_surplus_kwh": 0.0,
            "local_self_consumed_kwh": 0.0,
            "surplus_exported_kwh": 0.0,
            "self_consumption_rate": 0.0,
            "fractional_mode_count": 0,
            "max_mode_fractionality": 0.0,
        }

    total_charge_kwh = float(schedule["charge_kwh"].sum())
    total_discharge_kwh = float(schedule["discharge_kwh"].sum())
    total_import_kwh = float(schedule["grid_import_kwh"].sum())
    total_export_kwh = float(schedule["grid_export_kwh"].sum())
    total_grid_exchange_kwh = total_import_kwh + total_export_kwh
    throughput_kwh = total_charge_kwh + total_discharge_kwh
    usable_capacity = max(battery.soc_max_kwh - battery.soc_min_kwh, 1e-9)
    equivalent_cycles = total_discharge_kwh / usable_capacity
    average_soc_kwh = float(schedule["soc_kwh"].mean())
    average_soc_utilization = (average_soc_kwh - battery.soc_min_kwh) / usable_capacity

    local_surplus = (-schedule["net_demand_kwh"]).clip(lower=0.0)
    surplus_exported = pd.concat([local_surplus, schedule["grid_export_kwh"]], axis=1).min(axis=1)
    local_self_consumed = local_surplus - surplus_exported
    local_surplus_kwh = float(local_surplus.sum())
    surplus_exported_kwh = float(surplus_exported.sum())
    local_self_consumed_kwh = float(local_self_consumed.sum())
    self_consumption_rate = (
        local_self_consumed_kwh / local_surplus_kwh if local_surplus_kwh > 0.0 else 1.0
    )

    mode = schedule["mode"]
    distance_to_integer = mode.map(lambda value: min(abs(value), abs(1.0 - value)))
    fractional_mask = distance_to_integer > 1e-6

    return {
        "total_charge_kwh": total_charge_kwh,
        "total_discharge_kwh": total_discharge_kwh,
        "throughput_kwh": throughput_kwh,
        "total_import_kwh": total_import_kwh,
        "total_export_kwh": total_export_kwh,
        "total_grid_exchange_kwh": total_grid_exchange_kwh,
        "equivalent_cycles": equivalent_cycles,
        "average_soc_kwh": average_soc_kwh,
        "average_soc_utilization": average_soc_utilization,
        "local_surplus_kwh": local_surplus_kwh,
        "local_self_consumed_kwh": local_self_consumed_kwh,
        "surplus_exported_kwh": surplus_exported_kwh,
        "self_consumption_rate": self_consumption_rate,
        "min_soc_kwh": float(schedule["soc_kwh"].min()),
        "max_soc_kwh": float(schedule["soc_kwh"].max()),
        "fractional_mode_count": int(fractional_mask.sum()),
        "max_mode_fractionality": float(distance_to_integer.max()),
    }
