import pandas as pd
from helpers.assign4.wrapper import assign4wrapper

data = pd.read_csv("data/net_demand_and_price.csv")

if __name__ == "__main__":
    scenarios = assign4wrapper(data, price_scale=10)
    scenarios.run()
