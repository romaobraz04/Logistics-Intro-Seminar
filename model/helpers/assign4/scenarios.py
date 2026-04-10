import pandas as pd


def flat_prices(data: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
    data: project data
    """
    price = data["Price (EUR/kWh)"]
    data["Price (EUR/kWh)"] = price.mean()
    return data


def price_scale(data: pd.DataFrame, scale: float) -> pd.DataFrame:
    """
    Args:
    data: project data
    scale: factor by which to scale the prices, e.g. 1.1 for 10% increase, 0.5 for 50% decrease
    """
    data["Price (EUR/kWh)"] = data["Price (EUR/kWh)"] * scale
    return data


def perfect_efficiency(params: dict) -> dict:
    """
    Args:
    data: project data
    """
    params["charge_efficiency"] = 1.0
    params["discharge_efficiency"] = 1.0
    return params


def low_efficiency(params: dict, charge: float = 0.85, discharge: float = 0.85) -> dict:
    """
    Args:
    data: project data
    """
    params["charge_efficiency"] = charge
    params["discharge_efficiency"] = discharge
    return params


def tight_power_limits(
    params: dict, charge: float = 0.5, discharge: float = 0.5
) -> dict:
    """
    Args:
    data: project data
    """
    params["charge_limit"] = charge
    params["discharge_limit"] = discharge
    return params


def loose_power_limits(
    params: dict, charge: float = 5.0, discharge: float = 5.0
) -> dict:
    """
    Args:
    data: project data
    """
    params["charge_limit"] = charge
    params["discharge_limit"] = discharge
    return params


def small_capacity(params: dict, value: float = 2.0) -> dict:
    """
    Args:
    data: project data
    """
    params["max_soc"] = value
    return params


def large_capacity(params: dict, value: float = 10.0) -> dict:
    """
    Args:
    data: project data
    """
    params["max_soc"] = value
    return params


def horizon_scale(data: pd.DataFrame, size: float, period: str) -> pd.DataFrame:
    """
    Args:
    data: project data
    size: proportion of the dataset to be studied, e.g. 0.1 for 10%, 0.75 for 75%
    period: period to be studied - 'first', 'middle', or 'last'
    """
    period_aliases = {"start": "first", "middle": "middle", "end": "last"}
    normalized_period = period_aliases.get(period, period)

    if normalized_period == "first":
        return data.head(int(data.shape[0] * size))
    elif normalized_period == "middle":
        start_index = int((data.shape[0] - (data.shape[0] * size)) / 2)
        end_index = start_index + int(data.shape[0] * size)
        return data.iloc[start_index:end_index]
    elif normalized_period == "last":
        return data.tail(int(data.shape[0] * size))
    else:
        raise ValueError("Invalid period. Choose 'first', 'middle', or 'last'.")
