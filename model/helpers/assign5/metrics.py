import pandas as pd
import pulp
import numpy as np


def _count_time_steps(dec_vars: dict, prefix: str) -> int:
    return sum(1 for key in dec_vars if key.startswith(prefix))


def equivalent_full_cycles(
    dec_vars: dict, delta: float, min_soc: float, max_soc: float
) -> float:
    """
    Calculate the equivalent full cycles based on the decision variables.

    Parameters:
    - dec_vars: A dictionary containing the decision variables for discharging.
    - delta: The time step duration in hours.
    - min_soc: The minimum state of charge (SOC) of the battery.
    - max_soc: The maximum state of charge (SOC) of the battery.

    Returns:
    - The equivalent full cycles as a float.
    """
    usable_capacity = max_soc - min_soc
    if usable_capacity <= 0:
        return 0.0

    total_discharge = sum(
        dec_vars.get(f"discharge_power_{t}", 0.0)
        for t in range(_count_time_steps(dec_vars, "discharge_power_"))
    )

    # Calculate the net energy throughput
    net_energy = total_discharge * delta

    # Calculate the equivalent full cycles
    equivalent_cycles = net_energy / usable_capacity

    return equivalent_cycles


def total_grid_exchange(dec_vars: dict, delta: float) -> float:
    """
    Calculate the total grid exchange based on the decision variables.

    Parameters:
    - dec_vars: A dictionary containing the decision variables for buying and selling electricity.
    - delta: The time step duration in hours.

    Returns:
    - The total grid exchange as a float.
    """
    periods = max(
        _count_time_steps(dec_vars, "electricity_buy_"),
        _count_time_steps(dec_vars, "electricity_sell_"),
    )
    total_buy = sum(dec_vars.get(f"electricity_buy_{t}", 0.0) for t in range(periods))
    total_sell = sum(
        dec_vars.get(f"electricity_sell_{t}", 0.0) for t in range(periods)
    )

    # Total grid exchange is the net energy exchanged with the grid
    total_exchange = (total_buy + total_sell) * delta

    return total_exchange


def self_comsumption_rate(dec_vars: dict, delta: float, net_demand: pd.Series) -> float:
    """
    Calculate the self-consumption rate based on the decision variables.

    Parameters:
    - dec_vars: A dictionary containing the decision variables for buying and selling electricity.
    - delta: The time step duration in hours.

    Returns:
    - The self-consumption rate as a float.
    """
    net_demand = net_demand.reset_index(drop=True)
    surplus = [max(-float(net_demand.iloc[t]), 0.0) for t in range(len(net_demand))]
    exported_surplus = sum(
        delta * min(surplus[t], dec_vars.get(f"electricity_sell_{t}", 0.0))
        for t in range(len(net_demand))
    )
    denominator = sum(delta * surplus[t] for t in range(len(net_demand)))

    # Self-consumption is the ratio of energy used from the battery to total energy consumed
    if denominator > 0:
        self_consumption_rate = (denominator - exported_surplus) / denominator
        return self_consumption_rate
    else:
        return 0.0


def behavioural_cost(
    dec_vars: dict, delta: float, obj_value_0: float, obj_value_fee: float, fee: float
) -> float:

    return (obj_value_0 - obj_value_fee) - fee * total_grid_exchange(dec_vars, delta)
