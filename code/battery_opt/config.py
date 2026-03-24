from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Formulation(str, Enum):
    BASIC = "basic"
    TIGHTER = "tighter"


class DatasetScope(str, Enum):
    CONFIGURED_CASE = "configured_case"
    FULL_DATASET = "full_dataset"


class WindowPosition(str, Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"


@dataclass(frozen=True)
class DataConfig:
    csv_path: Path
    start_index: int = 0
    periods: int | None = None


@dataclass(frozen=True)
class BatteryConfig:
    initial_soc_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    charge_power_limit_kw: float
    discharge_power_limit_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    grid_fee_eur_per_kwh: float = 0.0
    degradation_cost_eur_per_kwh_throughput: float = 0.0


@dataclass(frozen=True)
class SolverConfig:
    time_limit_seconds: float | None = None
    mip_gap: float | None = None
    threads: int | None = None
    output_flag: int = 0


@dataclass(frozen=True)
class CaseConfig:
    name: str
    data: DataConfig
    battery: BatteryConfig
    solver: SolverConfig = field(default_factory=SolverConfig)
    time_step_hours: float | None = None


@dataclass(frozen=True)
class FractionWindowConfig:
    fraction: float
    positions: tuple[WindowPosition, ...] = (WindowPosition.START,)
    max_periods: int | None = None


@dataclass(frozen=True)
class ScenarioOverrideConfig:
    price_scale: float = 1.0
    price_shift: float = 0.0
    flatten_prices: bool = False
    demand_scale: float = 1.0
    demand_shift: float = 0.0
    initial_soc_kwh: float | None = None
    soc_min_kwh: float | None = None
    soc_max_kwh: float | None = None
    charge_power_limit_kw: float | None = None
    discharge_power_limit_kw: float | None = None
    charge_efficiency: float | None = None
    discharge_efficiency: float | None = None
    grid_fee_eur_per_kwh: float | None = None
    degradation_cost_eur_per_kwh_throughput: float | None = None


@dataclass(frozen=True)
class LPScenarioConfig(ScenarioOverrideConfig):
    name: str = "scenario"


@dataclass(frozen=True)
class Assignment4Config:
    benchmark_full_horizon: bool = True
    window_dataset_scope: DatasetScope = DatasetScope.CONFIGURED_CASE
    horizon_windows: tuple[FractionWindowConfig, ...] = ()
    lp_scenarios: tuple[LPScenarioConfig, ...] = ()
    objective_tolerance: float = 1e-6
    schedule_tolerance: float = 1e-6


@dataclass(frozen=True)
class Assignment5Config:
    dataset_scope: DatasetScope = DatasetScope.CONFIGURED_CASE
    formulations: tuple[Formulation, ...] = (Formulation.TIGHTER,)
    grid_fee_values: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)


@dataclass(frozen=True)
class Assignment6Config:
    dataset_scope: DatasetScope = DatasetScope.CONFIGURED_CASE
    formulations: tuple[Formulation, ...] = (Formulation.TIGHTER,)
    degradation_cost_values: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.03)


@dataclass(frozen=True)
class Assignment7ScenarioConfig(ScenarioOverrideConfig):
    name: str = "extension_scenario"
    description: str = ""
    formulation: Formulation = Formulation.TIGHTER
    window_fraction: float | None = None
    window_position: WindowPosition = WindowPosition.START
    save_schedule: bool = False
    custom_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Assignment7Config:
    dataset_scope: DatasetScope = DatasetScope.CONFIGURED_CASE
    scenarios: tuple[Assignment7ScenarioConfig, ...] = ()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_case_config(path: str | Path) -> CaseConfig:
    config_path = Path(path)
    raw = _read_json(config_path)
    return CaseConfig(
        name=raw["name"],
        data=DataConfig(
            csv_path=_resolve_path(raw["data"]["csv_path"], config_path.parent),
            start_index=raw["data"].get("start_index", 0),
            periods=raw["data"].get("periods"),
        ),
        battery=BatteryConfig(
            initial_soc_kwh=raw["battery"]["initial_soc_kwh"],
            soc_min_kwh=raw["battery"]["soc_min_kwh"],
            soc_max_kwh=raw["battery"]["soc_max_kwh"],
            charge_power_limit_kw=raw["battery"]["charge_power_limit_kw"],
            discharge_power_limit_kw=raw["battery"]["discharge_power_limit_kw"],
            charge_efficiency=raw["battery"]["charge_efficiency"],
            discharge_efficiency=raw["battery"]["discharge_efficiency"],
            grid_fee_eur_per_kwh=raw["battery"].get("grid_fee_eur_per_kwh", 0.0),
            degradation_cost_eur_per_kwh_throughput=raw["battery"].get(
                "degradation_cost_eur_per_kwh_throughput",
                0.0,
            ),
        ),
        solver=SolverConfig(
            time_limit_seconds=raw.get("solver", {}).get("time_limit_seconds"),
            mip_gap=raw.get("solver", {}).get("mip_gap"),
            threads=raw.get("solver", {}).get("threads"),
            output_flag=raw.get("solver", {}).get("output_flag", 0),
        ),
        time_step_hours=raw.get("time_step_hours"),
    )


def load_assignment4_config(path: str | Path) -> Assignment4Config:
    raw = _read_json(path)
    windows = tuple(
        FractionWindowConfig(
            fraction=window["fraction"],
            positions=tuple(
                WindowPosition(position)
                for position in window.get("positions", [WindowPosition.START.value])
            ),
            max_periods=window.get("max_periods"),
        )
        for window in raw.get("horizon_windows", [])
    )
    scenarios = tuple(
        LPScenarioConfig(
            name=scenario["name"],
            **_scenario_override_kwargs(scenario),
        )
        for scenario in raw.get("lp_scenarios", [])
    )
    return Assignment4Config(
        benchmark_full_horizon=raw.get("benchmark_full_horizon", True),
        window_dataset_scope=DatasetScope(raw.get("window_dataset_scope", DatasetScope.CONFIGURED_CASE.value)),
        horizon_windows=windows,
        lp_scenarios=scenarios,
        objective_tolerance=raw.get("objective_tolerance", 1e-6),
        schedule_tolerance=raw.get("schedule_tolerance", 1e-6),
    )


def load_assignment5_config(path: str | Path) -> Assignment5Config:
    raw = _read_json(path)
    return Assignment5Config(
        dataset_scope=DatasetScope(raw.get("dataset_scope", DatasetScope.CONFIGURED_CASE.value)),
        formulations=_load_formulations(raw.get("formulations")),
        grid_fee_values=tuple(raw.get("grid_fee_values", [])),
    )


def load_assignment6_config(path: str | Path) -> Assignment6Config:
    raw = _read_json(path)
    return Assignment6Config(
        dataset_scope=DatasetScope(raw.get("dataset_scope", DatasetScope.CONFIGURED_CASE.value)),
        formulations=_load_formulations(raw.get("formulations")),
        degradation_cost_values=tuple(raw.get("degradation_cost_values", [])),
    )


def load_assignment7_config(path: str | Path) -> Assignment7Config:
    raw = _read_json(path)
    scenarios = tuple(
        Assignment7ScenarioConfig(
            name=scenario["name"],
            description=scenario.get("description", ""),
            formulation=Formulation(scenario.get("formulation", Formulation.TIGHTER.value)),
            window_fraction=scenario.get("window_fraction"),
            window_position=WindowPosition(scenario.get("window_position", WindowPosition.START.value)),
            save_schedule=scenario.get("save_schedule", False),
            custom_parameters=scenario.get("custom_parameters", {}),
            **_scenario_override_kwargs(scenario),
        )
        for scenario in raw.get("scenarios", [])
    )
    return Assignment7Config(
        dataset_scope=DatasetScope(raw.get("dataset_scope", DatasetScope.CONFIGURED_CASE.value)),
        scenarios=scenarios,
    )


def _load_formulations(raw: list[str] | None) -> tuple[Formulation, ...]:
    if raw is None:
        return (Formulation.TIGHTER,)
    return tuple(Formulation(item) for item in raw)


def _scenario_override_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "price_scale": raw.get("price_scale", 1.0),
        "price_shift": raw.get("price_shift", 0.0),
        "flatten_prices": raw.get("flatten_prices", False),
        "demand_scale": raw.get("demand_scale", 1.0),
        "demand_shift": raw.get("demand_shift", 0.0),
        "initial_soc_kwh": raw.get("initial_soc_kwh"),
        "soc_min_kwh": raw.get("soc_min_kwh"),
        "soc_max_kwh": raw.get("soc_max_kwh"),
        "charge_power_limit_kw": raw.get("charge_power_limit_kw"),
        "discharge_power_limit_kw": raw.get("discharge_power_limit_kw"),
        "charge_efficiency": raw.get("charge_efficiency"),
        "discharge_efficiency": raw.get("discharge_efficiency"),
        "grid_fee_eur_per_kwh": raw.get("grid_fee_eur_per_kwh"),
        "degradation_cost_eur_per_kwh_throughput": raw.get("degradation_cost_eur_per_kwh_throughput"),
    }


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()
