import pandas as pd
from helpers.assign4.horizon_scale import run_horizon_scaling

data = pd.read_csv("data/net_demand_and_price.csv")

if __name__ == "__main__":
    run_horizon_scaling(data)
