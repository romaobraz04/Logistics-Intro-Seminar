import pandas as pd

from helpers.assign6.wrapper import assign6wrapper

data = pd.read_csv("data/net_demand_and_price.csv")

if __name__ == "__main__":
    assign6 = assign6wrapper(data)
    assign6.run()
