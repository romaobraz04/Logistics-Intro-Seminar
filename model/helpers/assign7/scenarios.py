"""
Foresight scenario configurations for Assignment 7.

Each dict describes one rolling-horizon foresight scenario:
  name            identifier used in DataFrames and filenames
  window          look-ahead window W (hours); None = full horizon (perfect foresight)
  step            decision step S (hours); None = full horizon
  epoch_offset    period index of the first decision epoch (0 = midnight, 12 = 13:00 EPEX)
  terminal        terminal boundary strategy: "hard_floor" | "soft_valuation" | None
"""


GRID_FEE = 0.02           
DEGRADATION_COST = 0.01   

# Sensitivity sweep levels
GRID_FEE_LEVELS = [0, 0.01, 0.02, 0.03, 0.04, 0.05]  
DEG_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03]           


PERFECT_FORESIGHT = {
    "name": "perfect_foresight",
    "window": None,
    "step": None,
    "epoch_offset": 0,
    "terminal": None,
    "forecast_lag": 0,
}

WEEKLY = {
    "name": "weekly",
    "window": 168,
    "step": 24,
    "epoch_offset": 0,
    "terminal": "hard_floor",
    "forecast_lag": 0,
}

TWO_DAY = {
    "name": "two_day",
    "window": 48,
    "step": 24,
    "epoch_offset": 0,
    "terminal": "hard_floor",
    "forecast_lag": 0,
}

DAY_AHEAD = {
    "name": "day_ahead",
    "window": 24,
    "step": 24,
    "epoch_offset": 0,
    "terminal": "hard_floor",
    "forecast_lag": 0,
}

EPEX_DAY_AHEAD = {
    "name": "epex_day_ahead",
    "window": 35,
    "step": 24,
    "epoch_offset": 12,
    "terminal": "hard_floor",
    "forecast_lag": 0,
}

TWO_DAY_SOFT_VAL = {
    "name": "two_day_soft_val",
    "window": 48,
    "step": 24,
    "epoch_offset": 0,
    "terminal": "soft_valuation",
    "forecast_lag": 0,
}

DAY_AHEAD_NAIVE = {
    "name": "day_ahead_naive",
    "window": 24,
    "step": 24,
    "epoch_offset": 0,
    "terminal": "hard_floor",
    "forecast_lag": 168,
}

EPEX_DAY_AHEAD_NAIVE = {
    "name": "epex_day_ahead_naive",
    "window": 35,
    "step": 24,
    "epoch_offset": 12,
    "terminal": "hard_floor",
    "forecast_lag": 168,
}


ALL_SCENARIOS = [
    PERFECT_FORESIGHT,
    WEEKLY,
    TWO_DAY,
    DAY_AHEAD,
    EPEX_DAY_AHEAD,
    TWO_DAY_SOFT_VAL,
    DAY_AHEAD_NAIVE,
    EPEX_DAY_AHEAD_NAIVE,
]

MAIN_SCENARIO_NAMES = [
    s["name"]
    for s in [
        PERFECT_FORESIGHT, WEEKLY, TWO_DAY, DAY_AHEAD, EPEX_DAY_AHEAD,
        DAY_AHEAD_NAIVE, EPEX_DAY_AHEAD_NAIVE,
    ]
]