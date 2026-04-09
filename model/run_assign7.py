import pandas as pd

from helpers.assign7.wrapper import assign7wrapper

data = pd.read_csv("data/net_demand_and_price.csv")

if __name__ == "__main__":
    assign7 = assign7wrapper(data)
    assign7.run()
